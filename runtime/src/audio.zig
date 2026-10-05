//! LILA audio (manual section 4): tracker synth + 256-byte 4:1 ADPCM ring.
//!
//! Pipeline: tracker voices (square/tri/noise/saw) + SFX voices mix into a
//! 248-sample i16 block -> IMA-ADPCM encoded into a 128-byte half of the
//! static 256-byte ring (4:1 vs 16-bit PCM) -> consumer half is decoded back
//! to PCM. All integer math; no libm; <1% of a core at arcade scale.
//!
//! Sample rate 44100. Game logic runs at 60Hz; audio is PULLED by the host
//! (native WAV writer / web AudioWorklet) — sample-accurate tracker timing.

const std = @import("std");
const c = @import("core.zig");

pub const SR: u32 = 44100;
pub const BLOCK: u32 = 248; // samples per half-block (124B nibbles)
pub const HALF_BYTES: u32 = 128; // 4B header + 124B nibbles

// memory layout inside ADDR_AUDIO (0x180000)
const A_PLAYING: u32 = 0x180000;
const A_TRACK: u32 = 0x180004;
const A_STEP_SAMPLES: u32 = 0x180008; // total samples per tracker step
// tracker voices: 4 × { step_idx u8, note_vol u16, phase u32, left u32, freq_step u32, noise u32 } = 20B
const A_TVOICES: u32 = 0x180010;
const TVOICE_SIZE: u32 = 24;
// sfx voices: 4 × { active u8, wave u8, vol u8, pad, freq i32, sweep i32, decay i32, age i32, phase u32, noise u32 } = 36B
const A_SVOICES: u32 = 0x180080;
const SVOICE_SIZE: u32 = 40;
// mix staging (after sfx voices end at 0x180120)
const A_MIX: u32 = 0x180140; // i16[248]
// ADPCM ring: 256B
const A_RING: u32 = 0x180400;
const A_RING_FLAGS: u32 = 0x180500; // u32: bit0 half0 has data, bit1 half1
// decode staging + output fifo (A_DEC spans 0x180520..0x180710)
const A_DEC: u32 = 0x180520; // i16[248] decoded half
const A_DEC_LEN: u32 = 0x180720; // u32 samples available in A_DEC
const A_DEC_POS: u32 = 0x180724; // u32 consumed
// write half cursor
const A_HALF: u32 = 0x180728; // u32 next half to fill (0/1)
// v7: master music volume 0..16 (16 = unity — identical to v6 mixing);
// written by OP_MUSIC_VOL, read per tracker sample
const A_MVOL: u32 = 0x18072C;

// v8 OPTIMIZER subsystem 4: fitted-voice state (FM / additive trig-spline).
// One pool of 4 slots, parallel to the legacy sfx voices; triggerSfx picks a
// free slot across BOTH pools (stealing the oldest on saturation). The state
// mirrors opt::audio::synth_reference EXACTLY (same 4096-entry Q15 LUT,
// same wrapping u32 phase accumulators, same 1/16-quantized linear envelope)
// so the build-time residual measures what the player actually hears.
const A_FVOICES: u32 = 0x180800;
const FVOICE_SIZE: u32 = 80;
// per fitted voice:
//   fv+0 active u8, fv+1 kind u8 (4 FM, 5 ADD), fv+2 npart u8
//   fv+4 f0 i32 Hz, fv+8 sweep i32 Hz/s
//   fv+12 ratio_q4 u8, fv+13 index_q4 u8, fv+14 noise_mix u8, fv+15 vol u8
//   fv+16 decay_samples i32, fv+20 age i32
//   fv+24 carrier phase u32, fv+28 modulator phase u32, fv+32 noise LCG u32
//   fv+40+i*4 partial phase-step u32[4], fv+56+i*4 partial phase u32[4]
//   fv+72+i partial amp u8[4], fv+76+i partial phase0 u8[4]

// ---------------- comptime tables ----------------

/// note (0..95, C1-based) -> u32 phase increment = freq * 2^32 / SR.
const NOTE_STEPS: [96]u32 = blk: {
    @setEvalBranchQuota(20000);
    var t: [96]u32 = undefined;
    for (0..96) |n| {
        const f: f64 = 440.0 * std.math.pow(f64, 2.0, (@as(f64, @floatFromInt(n)) - 57.0) / 12.0);
        t[n] = @intFromFloat(@round(f * 4294967296.0 / 44100.0));
    }
    break :blk t;
};

const IMA_STEP_TAB = [89]i32{
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
    253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
    1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327,
    3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442,
    11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
    32767,
};

const IMA_IDX_TAB = [16]i8{ -1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8 };

// ---------------- state accessors ----------------

inline fn rdT(a: u32) u32 { return c.rd32(a); }
inline fn wrT(a: u32, v: u32) void { c.wr32(a, v); }

pub fn reset() void {
    @memset(c.mem[c.ADDR_AUDIO .. c.ADDR_AUDIO + c.AUDIO_BYTES], 0);
    wrT(A_RING_FLAGS, 0);
    wrT(A_DEC_LEN, 0);
    wrT(A_DEC_POS, 0);
    wrT(A_HALF, 0);
    wrT(A_MVOL, 16); // v7: full volume unless the game ducks it
}

/// v7: master music volume (0 = mute, 16 = unity). Scaled per tracker
/// sample: (out * note_vol * mvol) >> 8 — at 16 this is bit-identical to
/// the v6 mixer's (out * note_vol) >> 4.
pub fn setMusicVol(vol: u32) void {
    wrT(A_MVOL, @min(vol, 16));
}

// ---------------- tracker / music ----------------

pub fn startMusic(track: u32) void {
    if (track >= c.n_tracks) return;
    wrT(A_PLAYING, 1);
    wrT(A_TRACK, track);
    const base = c.ADDR_MUSIC + track * c.MUSIC_STRIDE;
    const bpm: u32 = c.rd16(base);
    wrT(A_STEP_SAMPLES, if (bpm == 0) 4410 else SR * 15 / bpm);
    var v: u32 = 0;
    while (v < 4) : (v += 1) {
        const vb = A_TVOICES + v * TVOICE_SIZE;
        @memset(c.mem[vb .. vb + TVOICE_SIZE], 0);
        wrT(vb + 8, 0); // phase
        wrT(vb + 12, rdT(A_STEP_SAMPLES)); // samples left in step
        c.wr16(vb + 2, 0); // note_vol 0 until first row
    }
}

pub fn stopMusic() void {
    wrT(A_PLAYING, 0);
}

inline fn voiceData(track: u32, v: u32) u32 {
    return c.ADDR_MUSIC + track * c.MUSIC_STRIDE + 4 + v * 40;
}

fn trackerVoiceSample(v: u32) i32 {
    const playing = rdT(A_PLAYING);
    if (playing == 0) return 0;
    const track = rdT(A_TRACK);
    const vd = voiceData(track, v);
    const wave = c.mem[vd];
    const vb = A_TVOICES + v * TVOICE_SIZE;
    const note_vol: u32 = c.rd16(vb + 2);
    const phase = rdT(vb + 8);
    const noise = rdT(vb + 20);

    var out: i32 = 0;
    switch (wave) {
        0 => out = if (phase >> 31 != 0) 256 else -256, // square
        1 => { // triangle
            const ti: i32 = @intCast(phase >> 24); // 0..255
            const tri: i32 = if (ti < 128) ti * 4 - 256 else 510 * 2 - ti * 4 - 256 + 2;
            out = tri;
        },
        2 => { // noise
            var ns = noise;
            ns ^= ns << 13;
            ns ^= ns >> 17;
            ns ^= ns << 5;
            wrT(vb + 20, ns);
            out = if (ns & 0x80000000 != 0) 256 else -256;
        },
        3 => { // saw
            const t: i32 = @intCast(phase >> 24);
            out = (t - 128) * 2;
        },
        else => out = 0,
    }
    // v7: per-voice volume x master music volume (mvol 16 = v6 mixer)
    const mvol: i32 = @intCast(rdT(A_MVOL));
    return ((out * @as(i32, @intCast(note_vol))) * mvol) >> 8;
}

/// advance tracker voice state by one sample
fn trackerAdvance(v: u32) void {
    if (rdT(A_PLAYING) == 0) return;
    const track = rdT(A_TRACK);
    const vd = voiceData(track, v);
    const nsteps = c.mem[vd + 2];
    if (nsteps == 0) return;
    const vb = A_TVOICES + v * TVOICE_SIZE;
    var phase = rdT(vb + 8);
    const freq_step = rdT(vb + 16);
    phase +%= freq_step;
    wrT(vb + 8, phase);

    var left = rdT(vb + 12);
    if (left == 0) {
        // next row
        var si = c.mem[vb];
        si = (si + 1) % nsteps;
        c.mem[vb] = si;
        const row = c.rd16(vd + 4 + si * 2);
        const note: u32 = row >> 4;
        const vol: u32 = row & 0xF;
        if (note == 0xFFF) {
            c.wr16(vb + 2, 0); // off
        } else if (note == 0xFFE) {
            // hold: keep pitch + volume
        } else if (note < 96) {
            wrT(vb + 16, NOTE_STEPS[note]);
            c.wr16(vb + 2, @intCast(vol));
        }
        left = rdT(A_STEP_SAMPLES);
    }
    wrT(vb + 12, left - 1);
}

// ---------------- sfx ----------------

pub fn triggerSfx(prog: u32) void {
    if (prog >= c.n_sfx) return;
    const pb = c.ADDR_SFX + prog * 8;
    const wave = c.mem[pb];
    if (wave >= 4) {
        // ---- v8 fitted voice (subsystem 4): the .wav was analyzed at BUILD
        // time; here we only load the baked f(t) parameters. Zero decoding,
        // zero buffers — O(1) synthesis with instant pitch/time transforms.
        const fb = c.ADDR_SFX_FIT + prog * c.SFX_FIT_STRIDE;
        if (c.mem[fb] != wave) return; // loader invariant: fit record present
        // allocate in the FITTED pool (parallel to the legacy one)
        var best: u32 = 0;
        var best_age: i32 = -1;
        var i: u32 = 0;
        while (i < 4) : (i += 1) {
            const active = c.mem[A_FVOICES + i * FVOICE_SIZE];
            const age: i32 = @bitCast(rdT(A_FVOICES + i * FVOICE_SIZE + 20));
            if (active == 0) { best = i; best_age = -1; break; }
            if (age > best_age) { best_age = age; best = i; }
        }
        const fv = A_FVOICES + best * FVOICE_SIZE;
        @memset(c.mem[fv .. fv + FVOICE_SIZE], 0);
        c.mem[fv] = 1;
        c.mem[fv + 1] = wave;
        const f0: i32 = @intCast(c.rd16(fb + 4));
        c.mem[fv + 2] = if (wave == 5) @min(c.mem[fb + 15], 4) else 0;
        c.wr32(fv + 4, @bitCast(f0));
        c.wr32(fv + 8, @bitCast(@as(i32, @as(i16, @bitCast(c.rd16(fb + 8)))))); // sweep (FM)
        c.mem[fv + 12] = c.mem[fb + 10]; // ratio_q4
        c.mem[fv + 13] = c.mem[fb + 11]; // index_q4
        c.mem[fv + 14] = c.mem[fb + 14]; // noise mix
        c.mem[fv + 15] = c.mem[fb + 13]; // vol
        const decay: i32 = c.mem[fb + 12];
        c.wr32(fv + 16, @bitCast(decay * (@as(i32, @intCast(SR)) / 60)));
        c.wr32(fv + 20, 0); // age
        c.wr32(fv + 24, 0); // carrier phase
        c.wr32(fv + 28, 0); // modulator phase
        c.wr32(fv + 32, 0x2545F491); // noise seed
        if (wave == 5) {
            // additive: precompute per-partial phase steps + seed phases
            const npart: u32 = c.mem[fv + 2];
            var pk: u32 = 0;
            while (pk < npart) : (pk += 1) {
                const po = fb + 16 + pk * 3; // 3B per partial — matches loader v14 (was pk*4: overflowed the 28B record)
                const ratio_q6: i32 = c.mem[po];
                const ph0: u8 = c.mem[po + 2];
                const fpart: i64 = @divTrunc(@as(i64, f0) * ratio_q6, 64);
                c.wr32(fv + 40 + pk * 4, @intCast(phaseStep(fpart)));
                c.wr32(fv + 76 + pk, ph0);
            }
        }
        return;
    }
    // ---- legacy wave program (unchanged v1..v7 path) ----
    // find free or oldest legacy slot (original v6 policy)
    var best: u32 = 0;
    var best_age: i32 = -1;
    var i: u32 = 0;
    while (i < 4) : (i += 1) {
        const sb2 = A_SVOICES + i * SVOICE_SIZE;
        const active = c.mem[sb2];
        const age: i32 = @bitCast(rdT(sb2 + 24));
        if (active == 0) { best = i; best_age = -1; break; }
        if (age > best_age) { best_age = age; best = i; }
    }
    const sb = A_SVOICES + best * SVOICE_SIZE;
    const freq: i32 = @intCast(c.rd16(pb + 1));
    const sweep: i32 = @as(i16, @bitCast(c.rd16(pb + 3)));
    const decay: i32 = c.mem[pb + 5];
    const vol: i32 = c.mem[pb + 6];
    c.mem[sb] = 1;
    c.mem[sb + 1] = wave;
    c.mem[sb + 2] = @intCast(@as(u8, @truncate(@as(u32, @bitCast(vol)))));
    wrT(sb + 8, @bitCast(freq));
    wrT(sb + 12, @bitCast(sweep));
    wrT(sb + 16, @bitCast(decay * (@as(i32, @intCast(SR)) / 60)));
    wrT(sb + 24, 0); // age
    wrT(sb + 28, 0); // phase
    wrT(sb + 32, 0x2545F491); // noise seed
}

/// (pool allocation is per-pool inside triggerSfx: legacy voices use the
/// 4 legacy slots, fitted voices the 4 fitted slots — v6 policy preserved
/// for legacy programs, so legacy mixing is bit-identical)

/// u32 phase increment for a Hz value (integer, trunc — mirrors phase_step).
inline fn phaseStep(hz: i64) u32 {
    const v = @divTrunc(hz * 4294967296, SR);
    return @intCast(@max(v, 0));
}

/// Q15 sine from the engine LUT by 32-bit phase — mirrors the compiler's
/// s15() exactly (same LUT, same (ph >> 20) & 0xFFF indexing).
inline fn s15(ph: u32) i32 {
    return c.SIN_LUT[(ph >> 20) & 0xFFF];
}

fn fittedVoiceSample(i: u32) i32 {
    const fv = A_FVOICES + i * FVOICE_SIZE;
    if (c.mem[fv] == 0) return 0;
    const kind = c.mem[fv + 1];
    const vol: i32 = c.mem[fv + 15];
    const decay_samples: i32 = @bitCast(rdT(fv + 16));
    const age: i32 = @bitCast(rdT(fv + 20));
    if (decay_samples <= 0 or age >= decay_samples) {
        c.mem[fv] = 0;
        return 0;
    }
    // 1/16-quantized linear envelope — the engine's voice convention
    const env16: i32 = @intCast(@divTrunc(@as(i64, decay_samples - age) * 16, decay_samples));
    var out: i32 = 0;
    if (kind == 4) {
        // FM: f(t) = env(t)·sin(phi_c + I·sin(phi_m)) [ + noise mix ]
        const f0: i32 = @bitCast(rdT(fv + 4));
        const sweep: i32 = @bitCast(rdT(fv + 8));
        const ratio_q4: i32 = c.mem[fv + 12];
        const index_q4: i32 = c.mem[fv + 13];
        const noise_mix: i32 = c.mem[fv + 14];
        var pc = rdT(fv + 24);
        var pm = rdT(fv + 28);
        // instantaneous freqs (integer; matches the build-time mirror)
        const f_now: i64 = @as(i64, f0) + @divTrunc(@as(i64, sweep) * age, SR);
        const fmod: i64 = @max(@divTrunc(f_now * ratio_q4, 16), 0);
        pc +%= phaseStep(f_now);
        pm +%= phaseStep(fmod);
        wrT(fv + 24, pc);
        wrT(fv + 28, pm);
        // modulation: dev is Q15 radians; 20861 = 2^32/(2pi·2^15) converts
        // it EXACTLY once into carrier phase units (the constant the
        // compiler's ground-truth loop validates against).
        const dev: i64 = @divTrunc(@as(i64, s15(pm)) * index_q4, 16);
        const prod: i64 = dev * 20861;
        const off: u32 = @truncate(@as(u64, @bitCast(prod)));
        const carr = pc +% off;
        out = s15(carr) >> 7; // Q15 -> ±256 wave units
        if (noise_mix > 0) {
            var ns = rdT(fv + 32);
            ns ^= ns << 13;
            ns ^= ns >> 17;
            ns ^= ns << 5;
            wrT(fv + 32, ns);
            const nz: i32 = if (ns & 0x80000000 != 0) 1 else -1;
            out = @divTrunc(out * (16 - noise_mix) + nz * 16 * noise_mix, 16);
        }
        // contribution = out·env16·vol / 256 (max ±240, legacy parity)
        return @intCast(@divTrunc(@as(i64, out) * env16 * vol, 256));
    }
    // ADD: f(t) = env(t)·sum a_i·sin(2pi r_i f0 t + phi_i) — 2-4 phase
    // accumulators, one LUT lookup + MAC each. No buffers, O(1) per sample.
    const npart: u32 = @min(c.mem[fv + 2], 4);
    var acc: i64 = 0;
    var pk: u32 = 0;
    while (pk < npart) : (pk += 1) {
        var ph = rdT(fv + 56 + pk * 4);
        ph +%= rdT(fv + 40 + pk * 4);
        if (age == 0) ph +%= @as(u32, c.mem[fv + 76 + pk]) << 24;
        wrT(fv + 56 + pk * 4, ph);
        acc += @as(i64, s15(ph) >> 7) * c.mem[fv + 72 + pk];
    }
    return @intCast(@divTrunc(acc * env16 * vol, 4096));
}

fn sfxVoiceSample(i: u32) i32 {
    const sb = A_SVOICES + i * SVOICE_SIZE;
    if (c.mem[sb] == 0) return 0;
    const wave = c.mem[sb + 1];
    const vol: i32 = c.mem[sb + 2];
    const freq: i32 = @bitCast(rdT(sb + 8));
    const sweep: i32 = @bitCast(rdT(sb + 12));
    const decay_samples: i32 = @bitCast(rdT(sb + 16));
    const age: i32 = @bitCast(rdT(sb + 24));
    const phase = rdT(sb + 28);
    const noise = rdT(sb + 32);

    if (decay_samples <= 0 or age >= decay_samples) {
        c.mem[sb] = 0;
        return 0;
    }
    // frequency sweep: freq + sweep * age / SR
    const f_now: i64 = @as(i64, freq) + @divTrunc(@as(i64, sweep) * age, SR);
    const step: u32 = @intCast(@max(@divTrunc(f_now * 4294967296, SR), 0));
    // volume decay envelope (linear)
    const env: i32 = @intCast(@divTrunc(@as(i64, decay_samples - age) * 16, decay_samples));
    const v = @min(@divTrunc(vol * env, 16), 15);

    var out: i32 = 0;
    switch (wave) {
        0 => out = if (phase >> 31 != 0) 256 else -256,
        1 => {
            const ti: i32 = @intCast(phase >> 24);
            out = if (ti < 128) ti * 4 - 256 else 510 * 2 - ti * 4 - 256 + 2;
        },
        2 => {
            var ns = noise;
            ns ^= ns << 13;
            ns ^= ns >> 17;
            ns ^= ns << 5;
            wrT(sb + 32, ns);
            out = if (ns & 0x80000000 != 0) 256 else -256;
        },
        3 => {
            const t: i32 = @intCast(phase >> 24);
            out = (t - 128) * 2;
        },
        else => out = 0,
    }
    // advance
    wrT(sb + 28, phase +% step);
    wrT(sb + 24, @bitCast(age + 1));
    return (out * v) >> 4;
}

// ---------------- mixing + ADPCM ring ----------------

fn mixBlock() void {
    // mix BLOCK samples of tracker + sfx voices into A_MIX (i16)
    var s: u32 = 0;
    while (s < BLOCK) : (s += 1) {
        var acc: i32 = 0;
        var v: u32 = 0;
        while (v < 4) : (v += 1) {
            acc += trackerVoiceSample(v);
            trackerAdvance(v);
        }
        var i: u32 = 0;
        while (i < 4) : (i += 1) {
            acc += sfxVoiceSample(i);
        }
        var f: u32 = 0;
        while (f < 4) : (f += 1) {
            acc += fittedVoiceSample(f);
        }
        // soft clip to i16
        if (acc > 32760) acc = 32760;
        if (acc < -32760) acc = -32760;
        c.wr16(A_MIX + s * 2, @bitCast(@as(i16, @intCast(acc))));
    }
}

fn encodeHalf() void {
    // IMA-encode BLOCK samples from A_MIX into ring half A_HALF
    const half = rdT(A_HALF);
    const dst = A_RING + half * HALF_BYTES;
    var predictor: i32 = c.rd16(A_MIX); // first sample
    var idx: i32 = 0;
    // header: predictor i16, idx u8, pad u8
    c.wr16(dst, @intCast(predictor));
    c.mem[dst + 2] = 0;
    c.mem[dst + 3] = 0;
    var nib: u32 = 0;
    while (nib < 124) : (nib += 1) {
        const smp: i32 = if (nib * 2 + 1 < BLOCK) c.rd16(A_MIX + (nib * 2 + 1) * 2) else 0;
        const smp2: i32 = if (nib * 2 + 2 < BLOCK) c.rd16(A_MIX + (nib * 2 + 2) * 2) else 0;
        // encode two samples into one byte
        const b0: u8 = @intCast(imaEncodeNib(&predictor, &idx, smp));
        const b1: u8 = @intCast(imaEncodeNib(&predictor, &idx, smp2));
        c.mem[dst + 4 + nib] = b0 | (b1 << 4);
    }
    // mark half full; advance cursor
    wrT(A_RING_FLAGS, rdT(A_RING_FLAGS) | (@as(u32, 1) << @intCast(half)));
    wrT(A_HALF, 1 - half);
}

fn imaEncodeNib(predictor: *i32, idx: *i32, smp: i32) u4 {
    const step = IMA_STEP_TAB[@intCast(@min(@max(idx.*, 0), 88))];
    var diff: i32 = smp - predictor.*;
    var sign: u4 = 0;
    if (diff < 0) {
        sign = 8;
        diff = -diff;
    }
    var nib: u4 = 0;
    if (diff >= step) { nib |= 4; diff -= step; }
    if (diff >= @divTrunc(step, 2)) { nib |= 2; diff -= @divTrunc(step, 2); }
    if (diff >= @divTrunc(step, 4)) { nib |= 1; }
    nib |= sign;
    // reconstruct
    var d: i32 = 0;
    if (nib & 4 != 0) d += step;
    if (nib & 2 != 0) d += @divTrunc(step, 2);
    if (nib & 1 != 0) d += @divTrunc(step, 4);
    if (nib & 8 != 0) d = -d;
    predictor.* += d;
    idx.* += IMA_IDX_TAB[nib];
    idx.* = @min(@max(idx.*, 0), 88);
    return nib;
}

fn imaDecodeNib(predictor: *i32, idx: *i32, nib: u4) i32 {
    const step = IMA_STEP_TAB[@intCast(@min(@max(idx.*, 0), 88))];
    var d: i32 = 0;
    if (nib & 4 != 0) d += step;
    if (nib & 2 != 0) d += @divTrunc(step, 2);
    if (nib & 1 != 0) d += @divTrunc(step, 4);
    if (nib & 8 != 0) d = -d;
    predictor.* += d;
    idx.* += IMA_IDX_TAB[nib];
    idx.* = @min(@max(idx.*, 0), 88);
    return predictor.*;
}

fn decodeHalf() void {
    // decode the OLDEST full half into A_DEC
    var flags = rdT(A_RING_FLAGS);
    if (flags == 0) return;
    // halves fill in order 0,1,0,1... decode the one NOT just written (oldest)
    const just_wrote: u32 = 1 - rdT(A_HALF);
    const oldest: u32 = just_wrote; // since we decode right after encode
    const src = A_RING + oldest * HALF_BYTES;
    var predictor: i32 = @as(i16, @bitCast(c.rd16(src)));
    var idx: i32 = c.mem[src + 2];
    var s: u32 = 0;
    while (s < 248) : (s += 1) {
        const byte = c.mem[src + 4 + s / 2];
        const nib: u4 = if (s & 1 == 0) @intCast(byte & 0xF) else @intCast(byte >> 4);
        const v = imaDecodeNib(&predictor, &idx, nib);
        c.wr16(A_DEC + s * 2, @bitCast(@as(i16, @intCast(@min(@max(v, -32768), 32767)))));
    }
    // clear half-full flag
    flags = rdT(A_RING_FLAGS);
    wrT(A_RING_FLAGS, flags & ~(@as(u32, 1) << @intCast(oldest)));
    wrT(A_DEC_LEN, 248);
    wrT(A_DEC_POS, 0);
}

/// Pull n samples of decoded PCM into dst (i16). Host calls this.
pub fn pull(dst: []i16) u32 {
    var written: u32 = 0;
    while (written < dst.len) {
        if (rdT(A_DEC_LEN) > rdT(A_DEC_POS)) {
            const pos = rdT(A_DEC_POS);
            dst[written] = @bitCast(c.rd16(A_DEC + pos * 2));
            wrT(A_DEC_POS, pos + 1);
            written += 1;
        } else {
            // produce more: mix -> encode -> decode
            mixBlock();
            encodeHalf();
            decodeHalf();
        }
    }
    return written;
}

