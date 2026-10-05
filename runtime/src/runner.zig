//! LILA native runner: loads a .libyte, runs the game with a scripted
//! autopilot (deterministic), dumps PNG frames + a WAV of the audio path.
//!
//! v9 — the Deterministic Dream Stack. Because the whole engine state is
//! one flat memory image, these are all the SAME primitive at different
//! points in time:
//!   --rewind N        bit-exact time travel + timeline fork (Braid/GGPO)
//!   --record FILE     input + per-frame state-hash chain (a "movie")
//!   --replay FILE     re-simulate a movie, verify the hash chain (TAS/netcode)
//!   --patch FILE@F    deterministic mid-run hot-swap, state preserved (live coding)
//!   --snapshot F@F    full 2MB save-state file (save anywhere, instantly)
//!   --load FILE       boot from a save-state and continue
//!   --profile         per-system wall-clock profiler (Insights-lite)
//!   --watch           interactive polling hot-reload (unchanged)
//!   --zbuffer/--no-zbuffer   3D depth knob (painter's vs per-pixel Z)
//!   --gamma/--no-gamma      v15 blend knob (sRGB vs linear-light mixing)

const std = @import("std");
const Io = std.Io;
const c = @import("core.zig");
const cfg = @import("engine_config.zig");
const engine = @import("engine.zig");
const ent = @import("entity.zig");
const audio = @import("audio.zig");
const raster = @import("raster.zig");
const tt = @import("timetravel.zig");

/// Human-readable loader error names (mirrors core.zig error codes).
fn errName(rc: i32) []const u8 {
    return switch (rc) {
        0 => "ok",
        -1 => "bad magic (not a .libyte)",
        -2 => "unsupported .libyte version",
        -3 => "file larger than the staging buffer",
        -4 => "truncated file",
        -5 => "corrupt huffman stream",
        -6 => "bad header",
        -7 => "LIMIT: game needs more entity slots than this engine build provides (capacity > MAX_ENT)",
        -8 => "LIMIT: too many functions",
        -9 => "LIMIT: too many dispatch tables",
        -10 => "LIMIT: too many scenes",
        -11 => "LIMIT: too many texts",
        -12 => "LIMIT: too many sprites",
        -13 => "LIMIT: sprite too dense (>128 verts)",
        -14 => "LIMIT: too many music tracks",
        -15 => "LIMIT: too many sfx programs",
        -16 => "LIMIT: too many anims",
        -17 => "LIMIT: too many typed fields",
        -18 => "LIMIT: constant pool exhausted",
        -19 => "LIMIT: instruction budget exceeded",
        -20 => "LIMIT: spawn tf-list pool exhausted",
        -21 => "LIMIT: more entity types than this build provides (raise max_types)",
        -22 => "LIMIT: more FSM tables than this build carries (v8)",
        -23 => "LIMIT: arrays exceed this build's planes (v11)",
        else => "unknown error",
    };
}

const FPS: u32 = 60;
const SR: u32 = 44100;
const SAMPLES_PER_FRAME: u32 = SR / FPS; // 735

const MOVIE_MAGIC = "LIR1"; // input + hash-chain movie (12B/frame)

const ScriptFn = *const fn (frame: u32) u32;

/// Autopilot: a plausible play session — thrust, rotate, fire in bursts,
/// venturing around the arena so waves spawn and kills happen.
fn autopilot(frame: u32) u32 {
    var m: u32 = 0;
    const phase = frame % 600;
    if (phase < 240) m |= 1 << 2; // up (thrust) for 4s
    if ((frame / 45) % 2 == 0) m |= 1 << 0; // left pulses
    if ((frame / 67) % 3 == 0) m |= 1 << 1; // right pulses
    if (phase > 60 and (frame % 9) < 5) m |= 1 << 3; // fire bursts
    return m;
}

/// "file@frame" -> { path, frame } (frame omitted -> null)
const AtArg = struct { path: []const u8, frame: ?u32 };
fn parseAt(arg: []const u8) AtArg {
    if (std.mem.lastIndexOfScalar(u8, arg, '@')) |at| {
        const f = std.fmt.parseInt(u32, arg[at + 1 ..], 10) catch null;
        return .{ .path = arg[0..at], .frame = f };
    }
    return .{ .path = arg, .frame = null };
}

/// Typed-field indices SHIFT whenever an entity gains/loses a field — the
/// runner used to hardcode score's tf index and silently read Particle.hue
/// after the schema grew. Look globals up by position instead: the first
/// global tf (ent==0xFF) is `score` by declaration order.
fn firstGlobalTf() u32 {
    var tf: u32 = 0;
    while (tf < c.n_tfs) : (tf += 1) {
        if (c.tfEnt(tf) == 0xFF) return tf;
    }
    return 0;
}

/// Nth typed field (declaration order) of entity type `t`. 0xFFFFFFFF if absent.
fn entTf(t: u32, ordinal: u32) u32 {
    var tf: u32 = 0;
    var n: u32 = 0;
    while (tf < c.n_tfs) : (tf += 1) {
        if (c.tfEnt(tf) == t) {
            if (n == ordinal) return tf;
            n += 1;
        }
    }
    return 0xFFFFFFFF;
}

/// Fixed fields in a wrap world are stored in the 17-bit wrap domain
/// (negative velocities appear as ~131000 unsigned). Sign-convert for
/// human-readable kinematics.
fn signed17(v: i32) i32 {
    if (v > 65535) return v - 131072;
    return v;
}

fn wavHeader(buf: []u8, n_samples: u32) void {
    @memset(buf, 0);
    @memcpy(buf[0..4], "RIFF");
    std.mem.writeInt(u32, buf[4..8], 36 + n_samples * 2, .little);
    @memcpy(buf[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, buf[16..20], 16, .little);
    std.mem.writeInt(u16, buf[20..22], 1, .little); // PCM
    std.mem.writeInt(u16, buf[22..24], 1, .little); // mono
    std.mem.writeInt(u32, buf[24..28], SR, .little);
    std.mem.writeInt(u32, buf[28..32], SR * 2, .little);
    std.mem.writeInt(u16, buf[32..34], 2, .little);
    std.mem.writeInt(u16, buf[34..36], 16, .little);
    @memcpy(buf[36..40], "data");
    std.mem.writeInt(u32, buf[40..44], n_samples * 2, .little);
}

fn fnLabel(buf: []u8, i: u32) []const u8 {
    return switch (c.fnKind(i)) {
        c.FN_INIT => "init",
        c.FN_UPDATE0 => "update0 (global)",
        c.FN_UPDATEENT => std.fmt.bufPrint(buf, "update[type {d}]", .{c.fnEntity(i)}) catch "?",
        c.FN_DRAW0 => "draw0 (global)",
        c.FN_DRAWENT => std.fmt.bufPrint(buf, "draw[type {d}]", .{c.fnEntity(i)}) catch "?",
        else => "?",
    };
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena: std.mem.Allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var in_path: []const u8 = "game.libyte";
    var frames: u32 = 600;
    var watch = false;
    var sram_path: ?[]const u8 = null; // v7: persistence file (opt-in keeps traces deterministic)
    var rewind_n: u32 = 0;
    var record_path: ?[]const u8 = null;
    var replay_path: ?[]const u8 = null;
    var patch_path: ?[]const u8 = null;
    var patch_at: u32 = 0xFFFFFFFF;
    var snap_path: ?[]const u8 = null;
    var snap_at: u32 = 0xFFFFFFFF;
    var noauto = false;
    var load_path: ?[]const u8 = null;
    var profile = false;
    // v10: --threads 1 disables the parallel-group workers; --net2 runs the
    // GGPO-style rollback harness
    var net2_frames: u32 = 0;
    var threads_flag: ?u32 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--frames") and i + 1 < args.len) {
            i += 1;
            frames = std.fmt.parseInt(u32, args[i], 10) catch 600;
        } else if (std.mem.eql(u8, args[i], "--threads") and i + 1 < args.len) {
            i += 1;
            threads_flag = std.fmt.parseInt(u32, args[i], 10) catch 0;
        } else if (std.mem.eql(u8, args[i], "--net2") and i + 1 < args.len) {
            i += 1;
            net2_frames = std.fmt.parseInt(u32, args[i], 10) catch 300;
        } else if (std.mem.eql(u8, args[i], "--watch")) {
            watch = true;
        } else if (std.mem.eql(u8, args[i], "--sram") and i + 1 < args.len) {
            i += 1;
            sram_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--rewind") and i + 1 < args.len) {
            i += 1;
            rewind_n = std.fmt.parseInt(u32, args[i], 10) catch 0;
        } else if (std.mem.eql(u8, args[i], "--record") and i + 1 < args.len) {
            i += 1;
            record_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--replay") and i + 1 < args.len) {
            i += 1;
            replay_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--patch") and i + 1 < args.len) {
            i += 1;
            const a = parseAt(args[i]);
            patch_path = a.path;
            patch_at = a.frame orelse 0xFFFFFFFF;
        } else if (std.mem.eql(u8, args[i], "--snapshot") and i + 1 < args.len) {
            i += 1;
            const a = parseAt(args[i]);
            snap_path = a.path;
            snap_at = a.frame orelse 0;
        } else if (std.mem.eql(u8, args[i], "--load") and i + 1 < args.len) {
            i += 1;
            load_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--profile")) {
            profile = true;
        } else if (std.mem.eql(u8, args[i], "--zbuffer")) {
            // v12: 3D rasterizer profile knob — per-pixel Z-buffer variant
            c.zbuf3d = true;
        } else if (std.mem.eql(u8, args[i], "--no-zbuffer")) {
            c.zbuf3d = false;
        } else if (std.mem.eql(u8, args[i], "--gamma")) {
            // v15: gamma-correct blending knob — mix RGB in linear light
            c.gamma_blend = true;
        } else if (std.mem.eql(u8, args[i], "--no-gamma")) {
            c.gamma_blend = false;
        } else if (std.mem.eql(u8, args[i], "--noauto")) {
            // headless idle: zero input mask (game-side AI / attract demos)
            noauto = true;
        } else {
            in_path = args[i];
        }
    }

    var dir = try std.Io.Dir.cwd().openDir(io, ".", .{});
    defer dir.close(io);
    var blob_buf: [256 * 1024]u8 = undefined;
    const blob = dir.readFile(io, in_path, &blob_buf) catch |e| {
        std.debug.print("cannot read {s}: {}\n", .{ in_path, e });
        std.process.exit(1);
        return;
    };

    // v7: restore SRAM before init (engine zeroing skips the SRAM range,
    // so saved values are visible to the game's init() via saved())
    if (sram_path) |sp| {
        var sram_buf: [c.SRAM_BYTES]u8 = undefined;
        const saved = dir.readFile(io, sp, &sram_buf) catch |e| blk: {
            std.debug.print("sram: no saved file '{s}' ({}) — starting fresh\n", .{ sp, e });
            break :blk sram_buf[0..0];
        };
        @memset(c.mem[c.ADDR_SRAM .. c.ADDR_SRAM + c.SRAM_BYTES], 0);
        @memcpy(c.mem[c.ADDR_SRAM .. c.ADDR_SRAM + saved.len], saved);
        std.debug.print("sram: restored {}B from {s}\n", .{ saved.len, sp });
    }

    if (threads_flag) |tn| {
        engine.threads_enabled = tn != 1;
        std.debug.print("threads: {s}\n", .{if (tn == 1) "disabled (--threads 1)" else "enabled (compile-proven groups only)"});
    }
    std.debug.print("raster profile: {s} | blending: {s}\n", .{
        if (c.zbuf3d) "z-buffer (--zbuffer)" else "painter's sort (default)",
        if (c.gamma_blend) "linear-light (--gamma)" else "sRGB (default)",
    });
    const rc = engine.init(blob);
    if (rc != 0) {
        std.debug.print("init failed: rc={d} ({s})\n", .{ rc, errName(rc) });
        if (rc <= -7) std.debug.print(
            "  this game outgrew the engine build profile. Rebuild the engine\n" ++
            "  with a bigger profile:  scripts/build_engine.sh big\n" ++
            "  (or custom: mem_mb=4 max_ent=2048 ...)\n", .{});
        std.process.exit(1);
        return;
    }
    engine.rememberSchema();
    const tfScore = firstGlobalTf();
    // how many globals does this game declare? (trace adapts: the shooter
    // prints score/lives/wave, other games print score + type census)
    var n_globals: u32 = 0;
    {
        var tf: u32 = 0;
        while (tf < c.n_tfs) : (tf += 1) {
            if (c.tfEnt(tf) == 0xFF) n_globals += 1;
        }
    }
    // chaos-metric field handles: Enemy vy(3)/vx(2)/behavior(5), Bullet kind(5)
    const tfEnemyVy = entTf(2, 3);
    const tfEnemyVx = entTf(2, 2);
    const tfEnemyBeh = entTf(2, 5);
    const tfBulletKind = entTf(1, 5);
    const metricsOk = tfEnemyVy != 0xFFFFFFFF and tfEnemyVx != 0xFFFFFFFF and
        tfEnemyBeh != 0xFFFFFFFF and tfBulletKind != 0xFFFFFFFF;
    std.debug.print("lila-engine profile '{s}': {d}MB flat mem | {d} ent slots/type | {d} types | {d} verts/frame | {d} instrs\n", .{
        cfg.PROFILE_NAME, cfg.MEM_MB, c.MAX_ENT, c.MAX_TYPES, c.MAX_VERTS, c.MAX_INSTR,
    });
    std.debug.print("lila-engine: {s} ({d}B) — {d} instrs, {d} fns, {d} entities (score=tf[{d}])\n", .{
        in_path, blob.len, c.instr_count, c.n_fns, c.n_entities, tfScore,
    });
    if (c.game_world_w > c.game_w or c.game_world_h > c.game_h) {
        std.debug.print("world: {d}x{d} (viewport {d}x{d} — scrolling world, camera active)\n", .{
            c.game_world_w, c.game_world_h, c.game_w, c.game_h,
        });
    }

    // ---------------- replay verification mode (netcode primitive) ----------
    // Re-simulate the whole game from a recorded input log; the per-frame
    // state hash chain either matches bit-for-bit or names the first
    // divergent frame. This is exactly the guarantee lockstep/rollback
    // netcode needs — verified on the real state, not a subset.
    if (replay_path) |mp| {
        var mov_buf: [4 * 1024 * 1024]u8 = undefined;
        const m = dir.readFile(io, mp, &mov_buf) catch |e| {
            std.debug.print("replay: cannot read {s}: {}\n", .{ mp, e });
            std.process.exit(1);
            return;
        };
        if (m.len < 8 or !std.mem.eql(u8, m[0..4], MOVIE_MAGIC)) {
            std.debug.print("replay: {s} is not an LIR1 movie\n", .{mp});
            std.process.exit(1);
            return;
        }
        const n = std.mem.readInt(u32, m[4..8], .little);
        std.debug.print("\n== replay verification (rollback-netcode primitive) ==\n", .{});
        std.debug.print("movie: {s} — {d} frames, {d}B ({d}B/frame: u32 input + u64 state hash)\n", .{ mp, n, m.len, 12 });
        var sink: [SAMPLES_PER_FRAME]i16 = undefined;
        var mism: u32 = 0;
        var first_bad: u32 = 0xFFFFFFFF;
        var f: u32 = 0;
        while (f < n) : (f += 1) {
            const off = 8 + f * 12;
            const inp = std.mem.readInt(u32, m[off..][0..4], .little);
            const hexp = std.mem.readInt(u64, m[off + 4 ..][0..8], .little);
            c.wr32(c.ADDR_INPUT, inp);
            engine.frame();
            _ = audio.pull(&sink);
            const got = tt.stateHash(f);
            if (got != hexp) {
                mism += 1;
                if (first_bad == 0xFFFFFFFF) {
                    first_bad = f;
                    std.debug.print("  first divergence @ frame {d}: expected {x:0>16} got {x:0>16}\n", .{ f, hexp, got });
                }
            }
        }
        if (mism == 0) {
            std.debug.print("  hash chain: {d}/{d} frames MATCH — bit-identical timeline\n", .{ n, n });
            std.debug.print("  VERDICT: PASS — deterministic; safe for lockstep/rollback netcode\n", .{});
        } else {
            std.debug.print("  hash chain: {d}/{d} frames DIVERGED\n", .{ mism, n });
            std.debug.print("  VERDICT: FAIL — state is not a pure function of inputs\n", .{});
        }
        return;
    }

    // ---------------- save-state boot ---------------------------------------
    var start_frame: u32 = 0;
    if (load_path) |lp| {
        const lbuf = try arena.alloc(u8, 44 + c.MEM_SIZE);
        const image = dir.readFile(io, lp, lbuf) catch |e| {
            std.debug.print("save-state: cannot read {s}: {}\n", .{ lp, e });
            std.process.exit(1);
            return;
        };
        const f = tt.deserializeInto(image) catch |e| {
            std.debug.print("save-state: {s} rejected ({})\n", .{ lp, e });
            std.process.exit(1);
            return;
        };
        start_frame = f;
        tt.setSchemaSig(engine.computeSchemaSig());
        std.debug.print("save-state: restored {s} @ frame {d} — resuming ({d} frames left)\n", .{ lp, f, frames -| f });
    }

    // audio staging (header written for the frames we will actually simulate)
    var pcm: [SAMPLES_PER_FRAME]i16 = undefined;
    var wav: std.ArrayList(u8) = .empty;
    defer wav.deinit(arena);
    const n_sim = frames -| start_frame; // v14: saturating — a save-state at frame >= frames simulates nothing, never underflows
    try wav.ensureTotalCapacity(arena, 44 + n_sim * SAMPLES_PER_FRAME * 2);
    var hdr: [44]u8 = undefined;
    wavHeader(&hdr, n_sim * SAMPLES_PER_FRAME);
    try wav.appendSlice(arena, &hdr);

    // time-travel staging: per-frame hashes (record/rewind) + snapshot ring
    const want_hashes = rewind_n > 0 or record_path != null;
    var main_hashes: std.ArrayList(u64) = .empty;
    defer main_hashes.deinit(arena);
    var main_inputs: std.ArrayList(u32) = .empty;
    defer main_inputs.deinit(arena);
    if (want_hashes) try main_hashes.ensureTotalCapacity(arena, frames);

    const snap_every: u32 = @max(1, frames / 64);
    var snaps: []tt.Snapshot = &.{};
    if (rewind_n > 0) {
        const n_slots = frames / snap_every + 1;
        snaps = try arena.alloc(tt.Snapshot, n_slots);
        @memset(snaps, .{ .frame = 0, .mem = &.{} });
        std.debug.print("time travel: snapshot ring {d} slots x {d}KB = {d}MB resident\n", .{
            n_slots, c.MEM_SIZE / 1024, n_slots * c.MEM_SIZE / (1024 * 1024),
        });
    }

    // chaos metrics: concurrency ceilings + threat-speed sampling. These
    // numbers ARE the anti-chaos contract: readable bullet budgets, weaver
    // weave amplitude, hunter speed under the player's.
    var max_bullets: u32 = 0;
    var max_enemies: u32 = 0;
    var max_ebullets: u32 = 0;
    var weaver_vy_min: i32 = 32767;
    var weaver_vy_max: i32 = -32768;
    var hunter_axis_max: i32 = 0;

    engine.prof_on = profile;

    var patch_buf: [256 * 1024]u8 = undefined;

    // ---------------- rollback netcode harness (v10) ------------------------
    // GGPO-style 2-player simulation built ENTIRELY on engine determinism:
    //   p0 = local inputs (known immediately), p1 = remote (arrives D frames
    //   late; local predicts hold-last). When the real p1 differs from the
    //   prediction, the session rolls back to the divergence frame, injects
    //   the corrected input history and re-simulates — then every resimmed
    //   hash MUST equal the golden (full-information) run's hash. This is
    //   the whole algorithm; UE needs a replication graph, Unity needs
    //   third-party middleware; Lila needs @memcpy. Determinism is the
    //   netcode.
    if (net2_frames > 0) {
        const D: u32 = 4; // remote input latency in frames
        const N = @min(net2_frames, frames);
        var sink: [SAMPLES_PER_FRAME]i16 = undefined;
        std.debug.print("\n== rollback netcode harness (2 players, {d}f latency {d}, hold-last prediction) ==\n", .{ N, D });

        // p1 script: the remote player holds each command for 100 frames —
        // deliberate transitions every 100f (gap >> latency D) so prediction
        // errors are ISOLATED: after each rollback the timeline is provably
        // back on the golden path before the next error lands. p0 (autopilot)
        // bits 0-3, p1 bits 8-11.
        const p1input = struct {
            fn f(frame: u32) u32 {
                var m: u32 = 0;
                switch ((frame / 100) % 3) {
                    0 => m |= 1 << 0, // left
                    1 => m |= 1 << 1, // right
                    else => m |= 1 << 3, // fire
                }
                return m << 8;
            }
        }.f;

        // capture the INITIAL state before the golden run touches it — this
        // is the frame-0 base both sessions must start from
        const snap0 = try tt.Snapshot.capture(arena, 0);

        // ---- golden pass: both inputs known in advance ----
        var golden: std.ArrayList(u64) = .empty;
        defer golden.deinit(arena);
        try golden.ensureTotalCapacity(arena, N);
        var f: u32 = 0;
        while (f < N) : (f += 1) {
            c.wr32(c.ADDR_INPUT, autopilot(f) | p1input(f));
            engine.frame();
            _ = audio.pull(&sink);
            try golden.append(arena, tt.stateHash(f));
        }

        // ---- predictive pass: p1 arrives D frames late ----
        snap0.restore();
        // ring backing store on the arena (5 x 2MB would smash the stack)
        const ring_mem = try arena.alloc(u8, (D + 1) * c.MEM_SIZE);
        var ring: [D + 1]tt.Snapshot = undefined;
        for (0..D + 1) |k| ring[k] = .{ .frame = 0, .mem = ring_mem[k * c.MEM_SIZE ..][0..c.MEM_SIZE] };
        var rollbacks: u32 = 0;
        var resim_frames: u64 = 0;
        var resim_ns: u64 = 0;
        var mismatches: u32 = 0;
        var pf: u32 = 0;
        while (pf < N) : (pf += 1) {
            // capture pre-frame state (ring of D+1 covers the max rollback)
            @memcpy(ring[pf % (D + 1)].mem, c.mem[0..]);
            ring[pf % (D + 1)].frame = pf;

            const pred_p1: u32 = if (pf == 0) 0 else p1input(pf - 1); // hold-last
            if (pf >= D) {
                // the remote input for frame pf-D has now arrived
                const late_p1 = p1input(pf - D);
                const pred_late: u32 = if (pf - D == 0) 0 else p1input(pf - D - 1);
                if (late_p1 != pred_late) {
                    // ROLLBACK: restore frame pf-D, resim with the corrected
                    // remote history, verify every resimmed hash vs golden.
                    // With isolated errors the base state is golden, so the
                    // resim MUST land exactly on the golden timeline.
                    const rb_from = pf - D;
                    ring[rb_from % (D + 1)].restore();
                    const t0 = engine.nowNs();
                    var k: u32 = rb_from;
                    while (k < pf) : (k += 1) {
                        // refresh the ring with the corrected timeline
                        @memcpy(ring[k % (D + 1)].mem, c.mem[0..]);
                        ring[k % (D + 1)].frame = k;
                        c.wr32(c.ADDR_INPUT, autopilot(k) | p1input(k));
                        engine.frame();
                        _ = audio.pull(&sink);
                        resim_frames += 1;
                        if (tt.stateHash(k) != golden.items[k]) mismatches += 1;
                    }
                    resim_ns +%= engine.nowNs() -| t0;
                    rollbacks += 1;
                }
            }
            // normal frame: p0 immediate, p1 predicted (hold-last).
            // No golden hash check here: during the D-frame prediction
            // window the timeline intentionally deviates until the rollback
            // lands (that is the entire point of prediction).
            c.wr32(c.ADDR_INPUT, autopilot(pf) | pred_p1);
            engine.frame();
            _ = audio.pull(&sink);
        }
        const final_hash = tt.stateHash(N - 1);
        std.debug.print("  rollbacks: {d} | frames resimulated: {d} | resim cost {d}us total ({d}us/rollback avg)\n", .{
            rollbacks,                     resim_frames, resim_ns / 1000,
            if (rollbacks > 0) resim_ns / 1000 / rollbacks else 0,
        });
        std.debug.print("  post-rollback convergence: {s} | final state == golden: {s}\n", .{
            if (mismatches == 0) "MATCH" else "DIVERGED",
            if (final_hash == golden.items[N - 1]) "YES" else "NO",
        });
        if (mismatches == 0 and final_hash == golden.items[N - 1]) {
            std.debug.print("  VERDICT: PASS — rollback converges bit-exactly; lockstep-safe\n", .{});
        } else {
            std.debug.print("  VERDICT: FAIL\n", .{});
            std.process.exit(1);
            return;
        }
        return;
    }

    var frame: u32 = start_frame;
    while (frame < frames) : (frame += 1) {
        // deterministic live coding: swap the bytecode mid-run; entity +
        // page-0 state survive when the schema hash matches
        if (frame == patch_at) {
            const fresh = dir.readFile(io, patch_path.?, &patch_buf) catch |e| {
                std.debug.print("patch: cannot read {s}: {}\n", .{ patch_path.?, e });
                std.process.exit(1);
                return;
            };
            const preserved = engine.hotSwap(fresh);
            std.debug.print("patch: hot-swap @ frame {d}: state_preserved={} instrs={d}\n", .{ frame, preserved, c.instr_count });
        }
        // full save-state write
        if (frame == snap_at) {
            var snap = try tt.Snapshot.capture(arena, frame);
            tt.setSchemaSig(engine.computeSchemaSig());
            snap.scalars = tt.readScalars();
            const image = try tt.serialize(arena, snap);
            try dir.writeFile(io, .{ .sub_path = snap_path.?, .data = image });
            std.debug.print("save-state: wrote {s} ({d}KB @ frame {d})\n", .{ snap_path.?, image.len / 1024, frame });
        }
        // snapshot ring capture (pre-frame state)
        if (rewind_n > 0 and frame % snap_every == 0) {
            snaps[frame / snap_every] = try tt.Snapshot.capture(arena, frame);
        }

        c.wr32(c.ADDR_INPUT, if (noauto) 0 else autopilot(frame));
        engine.frame();

        // pull exactly one frame of audio
        _ = audio.pull(&pcm);
        for (pcm) |s| {
            var b: [2]u8 = undefined;
            std.mem.writeInt(i16, &b, s, .little);
            try wav.appendSlice(arena, &b);
        }

        if (want_hashes) {
            try main_hashes.append(arena, tt.stateHash(frame));
            try main_inputs.append(arena, c.rd32(c.ADDR_INPUT));
        }

        // periodic gameplay trace (score/lives/wave + entity census); adapts
        // to the game's schema instead of assuming the shooter's globals
        if (frame % 150 == 149) {
            if (n_globals >= 5) {
                // shooter-shaped schema (score/hi/wave/lives/over/...):
                // labeled trace. Fewer globals -> generic census below.
                std.debug.print("  f{d:0>4}: score={d} lives={d} wave={d} enemies={d} bullets={d}\n", .{
                    frame + 1, ent.globalLoad(tfScore),
                    ent.globalLoad(tfScore + 3),
                    ent.globalLoad(tfScore + 2),
                    ent.count(2), ent.count(1),
                });
            } else {
                var tbuf: [128]u8 = undefined;
                var ts: usize = 0;
                var t: u32 = 0;
                while (t < c.n_entities and ts + 16 < tbuf.len) : (t += 1) {
                    const w = std.fmt.bufPrint(tbuf[ts..], "t{d}={d} ", .{ t, ent.count(t) }) catch break;
                    ts += w.len;
                }
                if (n_globals >= 1) {
                    std.debug.print("  f{d:0>4}: score={d} {s}\n", .{ frame + 1, ent.globalLoad(tfScore), tbuf[0..ts] });
                } else {
                    std.debug.print("  f{d:0>4}: {s}\n", .{ frame + 1, tbuf[0..ts] });
                }
            }
        }

        // chaos-metric sampling (shooter schema only — other games skip)
        if (metricsOk) {
            var idx = ent.nextLive(2, 0);
            while (idx != 0xFFFFFFFF) {
                const h = idx | (@as(u32, ent.genOf(2, idx)) << 20);
                const beh = ent.fieldLoad(tfEnemyBeh, h);
                if (beh == 1) { // weaver: vy weave amplitude (raw Q24.8)
                    const vy = signed17(ent.fieldLoad(tfEnemyVy, h));
                    if (vy < weaver_vy_min) weaver_vy_min = vy;
                    if (vy > weaver_vy_max) weaver_vy_max = vy;
                } else if (beh == 2) { // hunter: per-axis speed ceiling
                    const vx = signed17(ent.fieldLoad(tfEnemyVx, h));
                    const vy = signed17(ent.fieldLoad(tfEnemyVy, h));
                    const sp = @max(if (vx < 0) -vx else vx, if (vy < 0) -vy else vy);
                    if (sp > hunter_axis_max) hunter_axis_max = sp;
                }
                idx = ent.nextLive(2, idx + 1);
            }
            var bidx = ent.nextLive(1, 0);
            var neb: u32 = 0;
            while (bidx != 0xFFFFFFFF) {
                const bh = bidx | (@as(u32, ent.genOf(1, bidx)) << 20);
                if (ent.fieldLoad(tfBulletKind, bh) == 1) neb += 1;
                bidx = ent.nextLive(1, bidx + 1);
            }
            if (neb > max_ebullets) max_ebullets = neb;
            if (ent.count(1) > max_bullets) max_bullets = ent.count(1);
            if (ent.count(2) > max_enemies) max_enemies = ent.count(2);
        }

        // snapshot frames of interest
        if (frame == 0 or frame == 90 or frame == 300 or frame == frames - 1) {
            raster.rasterize();
            var name_buf: [64]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "frame_{d:0>4}.png", .{frame});
            try raster.writePng(arena, name, io);
            std.debug.print("  frame {d}: dumped {s} (stream={d} player={d} enemy={d} bullet={d} score={d})\n", .{
                frame, name, c.rd32(c.ADDR_STREAM), ent.count(0), ent.count(2), ent.count(1),
                ent.globalLoad(tfScore),
            });
        }
    }

    // WAV out
    try dir.writeFile(io, .{ .sub_path = "game.wav", .data = wav.items });
    std.debug.print("wrote game.wav ({d} samples) + {d} PNG frames\n", .{ n_sim * SAMPLES_PER_FRAME, 4 });

    // v7: persist SRAM if the game saved anything (and a path was given)
    if (sram_path) |sp| {
        try dir.writeFile(io, .{ .sub_path = sp, .data = c.mem[c.ADDR_SRAM .. c.ADDR_SRAM + c.SRAM_BYTES] });
        std.debug.print("sram: wrote {d}B to {s}\n", .{ c.SRAM_BYTES, sp });
    }

    // the anti-chaos report: concurrency ceilings + threat kinematics
    if (metricsOk) {
        std.debug.print("chaos metrics: max_enemies={d} max_bullets={d} max_enemy_bullets={d}\n", .{ max_enemies, max_bullets, max_ebullets });
        std.debug.print("  weaver vy weave: [{d}..{d}] raw (expect ~[-512..512] = ±2px/frame)\n", .{ weaver_vy_min, weaver_vy_max });
        std.debug.print("  hunter axis speed max: {d} raw (expect <= 384 = 1.5px/frame; player 4px)\n", .{hunter_axis_max});
    }

    // budget diagnostics (v6): a production engine tells you when it hit a
    // ceiling instead of letting content silently vanish
    if (c.stream_drop_total > 0) {
        std.debug.print("DIAG: {d} vertices dropped (vertex budget {d}/frame exceeded — raise max_verts)\n", .{ c.stream_drop_total, c.MAX_VERTS });
    }
    if (c.spawn_denied_total > 0) {
        std.debug.print("DIAG: {d} spawns denied (pool/per-type cap {d} full — raise max_ent)\n", .{ c.spawn_denied_total, c.MAX_ENT });
    }
    if (c.stream_drop_total == 0 and c.spawn_denied_total == 0) {
        std.debug.print("budgets clean: 0 verts dropped, 0 spawns denied\n", .{});
    }


    // ---------------- movie record ------------------------------------------
    if (record_path) |rp| {
        var mov: std.ArrayList(u8) = .empty;
        defer mov.deinit(arena);
        try mov.appendSlice(arena, MOVIE_MAGIC);
        var b4: [4]u8 = undefined;
        std.mem.writeInt(u32, &b4, @intCast(main_inputs.items.len), .little);
        try mov.appendSlice(arena, &b4);
        for (main_inputs.items, main_hashes.items) |inp, h| {
            var b: [12]u8 = undefined;
            std.mem.writeInt(u32, b[0..4], inp, .little);
            std.mem.writeInt(u64, b[4..12], h, .little);
            try mov.appendSlice(arena, &b);
        }
        try dir.writeFile(io, .{ .sub_path = rp, .data = mov.items });
        std.debug.print("movie: wrote {s} ({d} frames, {d}B — input log + state-hash chain)\n", .{ rp, main_inputs.items.len, mov.items.len });
    }

    // ---------------- deterministic rewind + fork ---------------------------
    if (rewind_n > 0 and frames >= 16 and snaps.len > 0) {
        const K = snap_every;
        const T = (frames / 2 / K) * K; // rewind point with a ring snapshot
        if (T >= K and T / K < snaps.len) {
            std.debug.print("\n== time travel: deterministic rewind ==\n", .{});
            const si = T / K;
            std.debug.print("  rewinding {d} frames: frame {d} -> {d} (restore snapshot {d})\n", .{ frames - T, frames, T, si });
            var sink: [SAMPLES_PER_FRAME]i16 = undefined;
            snaps[si].restore();
            var match: u32 = 0;
            var first_bad: u32 = 0xFFFFFFFF;
            var f2: u32 = T;
            while (f2 < frames) : (f2 += 1) {
                c.wr32(c.ADDR_INPUT, autopilot(f2));
                engine.frame();
                _ = audio.pull(&sink);
                if (tt.stateHash(f2) == main_hashes.items[f2]) match += 1 else if (first_bad == 0xFFFFFFFF) first_bad = f2;
            }
            if (first_bad == 0xFFFFFFFF) {
                std.debug.print("  re-simulation: {d}/{d} frame hashes MATCH, final state == main timeline (bit-exact restore)\n", .{ match, frames - T });
            } else {
                std.debug.print("  re-simulation DIVERGED @ frame {d} ({d}/{d} match) — restore is not bit-exact!\n", .{ first_bad, match, frames - T });
            }
            // alternate timeline: same restore point, perturbed input
            snaps[si].restore();
            var diverged_at: u32 = 0xFFFFFFFF;
            f2 = T;
            while (f2 < frames) : (f2 += 1) {
                c.wr32(c.ADDR_INPUT, autopilot(f2) ^ (1 << 3)); // fire bit flipped
                engine.frame();
                _ = audio.pull(&sink);
                if (diverged_at == 0xFFFFFFFF and tt.stateHash(f2) != main_hashes.items[f2]) diverged_at = f2;
            }
            if (diverged_at != 0xFFFFFFFF) {
                std.debug.print("  fork: fire bit inverted @ frame {d} -> timeline diverged at frame {d} (zero re-init, zero reload)\n", .{ T, diverged_at });
            } else {
                std.debug.print("  fork: perturbation produced an identical timeline (game ignored the bit here)\n", .{});
            }
        }
    }

    // ---------------- per-system profiler ------------------------------------
    if (profile) {
        const Order = struct { ns: u64, calls: u64, idx: u32 };
        var rows: [c.MAX_FNS]Order = undefined;
        var n_rows: usize = 0;
        var total_ns: u64 = engine.prof_integ_ns;
        i = 0;
        while (i < c.n_fns) : (i += 1) {
            rows[n_rows] = .{ .ns = engine.prof_fn_ns[i], .calls = engine.prof_fn_calls[i], .idx = @intCast(i) };
            total_ns += engine.prof_fn_ns[i];
            n_rows += 1;
        }
        // insertion sort by ns desc (n <= 128)
        var a: usize = 1;
        while (a < n_rows) : (a += 1) {
            const key = rows[a];
            var b: usize = a;
            while (b > 0 and rows[b - 1].ns < key.ns) : (b -= 1) rows[b] = rows[b - 1];
            rows[b] = key;
        }
        std.debug.print("\n== profile (per-system wall clock, {d} frames) ==\n", .{frames -| start_frame});
        var lbuf: [48]u8 = undefined;
        const show = @min(n_rows, 14);
        i = 0;
        while (i < show) : (i += 1) {
            const r = rows[i];
            if (r.ns == 0) break;
            const lbl = fnLabel(&lbuf, r.idx);
            std.debug.print("  {s: <18} {d: >8} calls  {d: >9.2} ms  {d: >5.1}%\n", .{
                lbl, r.calls, @as(f64, @floatFromInt(r.ns)) / 1e6,
                @as(f64, @floatFromInt(r.ns)) / @as(f64, @floatFromInt(total_ns)) * 100.0,
            });
        }
        std.debug.print("  {s: <18} {d: >8} calls  {d: >9.2} ms  {d: >5.1}%\n", .{
            "integration", engine.prof_integ_calls, @as(f64, @floatFromInt(engine.prof_integ_ns)) / 1e6,
            @as(f64, @floatFromInt(engine.prof_integ_ns)) / @as(f64, @floatFromInt(total_ns)) * 100.0,
        });
        std.debug.print("  scheduled total: {d:.2} ms ({d:.3} ms/frame)\n", .{
            @as(f64, @floatFromInt(total_ns)) / 1e6,
            @as(f64, @floatFromInt(total_ns)) / 1e6 / @as(f64, @floatFromInt(frames -| start_frame)),
        });
    }

    if (watch) {
        std.debug.print("watch mode: polling {s} for changes (Ctrl-C to stop)\n", .{in_path});
        var last_sig: u64 = 0;
        // stat-based polling
        var stat_buf: [128]u8 = undefined;
        _ = &stat_buf;
        var iter: u32 = 0;
        while (iter < 100000) : (iter += 1) {
            io.sleep(Io.Duration.fromMilliseconds(50), .boot) catch {};
            const fresh = dir.readFile(io, in_path, &blob_buf) catch continue;
            // crude change detect: size + fnv hash
            var h: u64 = 1469598103934665603;
            for (fresh) |b| {
                h ^= b;
                h *%= 1099511628211;
            }
            if (h != last_sig and iter > 0) {
                if (last_sig != 0) {
                    std.debug.print("change detected ({d}B) — hot-swapping\n", .{fresh.len});
                    const preserved = engine.hotSwap(fresh);
                    std.debug.print("  hot-swap ok (state preserved: {}) — {d} instrs\n", .{ preserved, c.instr_count });
                }
                last_sig = h;
            }
        }
    }
}
