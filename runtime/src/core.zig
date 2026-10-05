//! LILA runtime core: config-derived flat memory map, Q24.8 fixed-point,
//! comptime trig LUTs (CTFE), and shared constants. Pure data + functions —
//! no std.Io, no allocation — identical code paths on native and
//! wasm32-freestanding.

const std = @import("std");
const cfg = @import("engine_config.zig");

// ---------------- memory map (comptime-derived, per docs/SPEC.md) ----------------
// The map is GENERATED from the build profile (engine_config.zig) at comptime:
// regions grow with their pools, anchors are preserved at the default profile,
// and the total size is asserted to fit. Default profile = byte-identical to
// the fixed v1 map (see test "default map matches v1 layout" in test_map.zig).

fn alignUp(v: u32, a: u32) u32 {
    return (v + a - 1) & ~(a - 1);
}

pub const MEM_SIZE: u32 = cfg.MEM_MB * 1024 * 1024;

pub const ADDR_INPUT: u32 = 0x0000; // u32 bitmask, written by host
pub const ADDR_FRAME: u32 = 0x0004; // u32 frame counter
pub const ADDR_RAND: u32 = 0x0008; // u32 LCG state
pub const ADDR_SCENE: u32 = 0x000C; // u32 scene domain pos + 1 (0 = none)
pub const ADDR_SHAKE: u32 = 0x0050; // u32 screen-shake amplitude (px), decays per frame

// v7: camera registers (Q24.8 world-space viewport CENTER, clamped to the
// world by OP_CAMERA). Render subtracts them from world-layer draws; the
// starfield parallaxes against them. cam = 0 keeps v1 screen-space behavior
// for games that never call camera().
pub const ADDR_CAM_X: u32 = 0x0054;
pub const ADDR_CAM_Y: u32 = 0x0058;

// ---------------- v11: 3D camera + projection stash (page 0) ----------------
// All inside 0x00..0xFF so hot-swap page-0 backup captures them (state
// continuity) and the rewind/rollback 2MB memcpy covers them automatically.
pub const ADDR_CAM3_X: u32 = 0x005C;     // Q24.8 world position
pub const ADDR_CAM3_Y: u32 = 0x0060;
pub const ADDR_CAM3_Z: u32 = 0x0064;
pub const ADDR_CAM3_YAW: u32 = 0x0068;   // ang units (4096 = 360 deg)
pub const ADDR_CAM3_PITCH: u32 = 0x006C; // ang units
pub const ADDR_PROJ_X: u32 = 0x0070;     // last proj3 screen x (Q24.8)
pub const ADDR_PROJ_Y: u32 = 0x0074;     // last proj3 screen y (Q24.8)
pub const ADDR_PROJ_OK: u32 = 0x0078;    // last proj3 ok flag (0/1: in front)

// 3D camera convention (matches the 2D screen convention's handedness):
//   +X right, +Y forward (yaw 0 looks along +Y), +Z up.
// yaw rotates around Z (clockwise seen from above), pitch tilts the forward
// axis up (positive pitch looks up). fov is fixed at 90 degrees:
// focal = half screen width (Q24.8), so scale = focal / view_forward.

// v13: the globals block moved from 0x10 to 0x80. It always ended at 0x50,
// but nothing stopped a game from DECLARING more than 512 bits — the spill
// silently overwrote the engine's own registers (shake 0x50, 2D cam 0x54,
// cam3 0x5C..0x6C, proj stash 0x70..0x7C). The block now spans the reserved
// register-free gap 0x80..0x100 (META base): 128 bytes = 1024 bits, still
// inside PAGE0_BYTES (hot-swap/rewind capture it with the standard memcpy).
// Bytecode field offsets are relative to this base, so old .libyte files
// load unchanged. The checker now rejects overflow loudly (see checker.rs).
pub const ADDR_GLOBALS: u32 = 0x0080; // bit-packed globals block
pub const GLOBALS_BYTES: u32 = 128;

// ---------------- flat memory (the engine buffer; wasm exports it) ----------------

pub var mem: [MEM_SIZE]u8 align(16) = [_]u8{0} ** MEM_SIZE;

// entity meta: per type: gens u16[MAX_ENT] + existence bitmap u64[BM_WORDS]
// meta stride = MAX_ENT*2 + BM_WORDS*8
// v7: MAX_TYPES is a build-profile knob (default 16 = v1 layout).
pub const MAX_TYPES: u32 = cfg.MAX_TYPES;
pub const MAX_ENT: u32 = cfg.MAX_ENT;
pub const BM_WORDS: u32 = (MAX_ENT + 63) / 64; // u64 words in the existence bitmap

pub const ADDR_META: u32 = 0x0100;
pub const META_STRIDE: u32 = MAX_ENT * 2 + BM_WORDS * 8; // gens + bitmap

// dense rows: MAX_TYPES * MAX_ENT slots * 32B (max row 32B)
pub const ADDR_DENSE: u32 = alignUp(ADDR_META + MAX_TYPES * META_STRIDE, 0x8000);
pub const DENSE_TYPE_STRIDE: u32 = MAX_ENT * 32; // max row 32B

// ---- cache-line alignment contract (v8 subsystem 1, memory topology) ----
// Every entity row plane begins on a 64-byte L1 line boundary:
//   - ADDR_DENSE is aligned to 32KB (well past one cache line),
//   - DENSE_TYPE_STRIDE = MAX_ENT*32 = 16384 is a multiple of 64, so every
//     per-type dense plane starts line-aligned — a scan loop over one type
//     never shares a cache line with a foreign type's rows,
//   - cold planes ride the same multiples.
// opt::soa re-proves the per-type row geometry in the Zig manifest at engine
// build time; this block proves the FLAT MEMORY MAP can never drift from the
// contract. If any constant above changes, the engine refuses to compile.
comptime {
    if (ADDR_DENSE % 64 != 0) @compileError("dense plane base not 64B cache-line aligned");
    if (DENSE_TYPE_STRIDE % 64 != 0) @compileError("dense type stride not 64B aligned");
    if (ADDR_COLD % 64 != 0) @compileError("cold plane base not 64B aligned");
    if (COLD_TYPE_STRIDE % 64 != 0) @compileError("cold type stride not 64B aligned");
}

// cold rows: MAX_TYPES * MAX_ENT * 8 (max cold row 8B)
pub const ADDR_COLD: u32 = ADDR_DENSE + MAX_TYPES * DENSE_TYPE_STRIDE;
pub const COLD_TYPE_STRIDE: u32 = MAX_ENT * 8;

// cold side-table gens: MAX_TYPES * MAX_ENT * 2
pub const ADDR_COLDGEN: u32 = ADDR_COLD + MAX_TYPES * COLD_TYPE_STRIDE;

// decoded instructions: MAX_INSTR * 20B
pub const ADDR_CODE: u32 = alignUp(ADDR_COLDGEN + MAX_TYPES * MAX_ENT * 2, 0x10000);
pub const MAX_INSTR: u32 = cfg.MAX_INSTR;
pub const INSTR_SIZE: u32 = 20; // op u8, pad[3], a,b,c i32, extra_off u32

// spawn tf-list pool: worst case 1 u16 per instruction
pub const ADDR_EXTRA: u32 = ADDR_CODE + MAX_INSTR * INSTR_SIZE;
pub const EXTRA_BYTES: u32 = MAX_INSTR * 2;

// typed-field registry: MAX_TF * 16B
pub const TF_BASE: u32 = ADDR_EXTRA + EXTRA_BYTES;
pub const MAX_TF: u32 = 512;
pub const TF_SIZE: u32 = 16;

// entity schemas: MAX_TYPES * 6B, then integration pairs MAX_TYPES * 84B
pub const ENT_SCHEMA_BASE: u32 = TF_BASE + MAX_TF * TF_SIZE;
pub const INTEG_BASE: u32 = alignUp(ENT_SCHEMA_BASE + MAX_TYPES * 6, 0x200);
pub const INTEG_STRIDE: u32 = 84; // n u8 + pad[3] + 8 pairs × 10B

// fn records: MAX_FNS * 16B
pub const ADDR_FNS: u32 = alignUp(INTEG_BASE + MAX_TYPES * INTEG_STRIDE, 0x1000);
pub const MAX_FNS: u32 = cfg.MAX_FNS;
pub const FN_SIZE: u32 = 16; // kind u8, entity u8, nparams u8, nlocals u8, start u32, len u32, pad u32

// dispatch tables: MAX_TABLES * 256 * 2
pub const ADDR_TABLES: u32 = alignUp(ADDR_FNS + MAX_FNS * FN_SIZE, 0x1000);
pub const MAX_TABLES: u32 = 8;
pub const TABLE_SLOTS: u32 = 256; // key -> fn u16
pub const TABLE_STRIDE: u32 = TABLE_SLOTS * 2;

// scenes: MAX_SCENES * 2, then constant pool MAX_POOL * 4
pub const ADDR_SCENES: u32 = ADDR_TABLES + MAX_TABLES * TABLE_STRIDE;
pub const MAX_SCENES: u32 = 32;

pub const ADDR_POOL: u32 = alignUp(ADDR_SCENES + MAX_SCENES * 2, 0x100);
pub const MAX_POOL: u32 = cfg.MAX_POOL;

// sprites: MAX_SPRITES * 128 verts * 14B
pub const ADDR_SPRITES: u32 = alignUp(ADDR_POOL + MAX_POOL * 4, 0x4000);
pub const MAX_SPRITES: u32 = cfg.MAX_SPRITES;
pub const MAX_SPR_VERTS: u32 = 128; // per sprite
pub const SPRITE_STRIDE: u32 = MAX_SPR_VERTS * 14;

// music: MAX_TRACKS * 1024, then sfx MAX_SFX * 8
pub const ADDR_MUSIC: u32 = ADDR_SPRITES + MAX_SPRITES * SPRITE_STRIDE;
pub const MAX_TRACKS: u32 = cfg.MAX_TRACKS;
pub const MUSIC_STRIDE: u32 = 1024;

pub const ADDR_SFX: u32 = ADDR_MUSIC + MAX_TRACKS * MUSIC_STRIDE;
pub const MAX_SFX: u32 = cfg.MAX_SFX;

// anims: MAX_ANIMS * (4 + 32*20)
pub const ADDR_ANIM: u32 = alignUp(ADDR_SFX + MAX_SFX * 8, 0x400);
pub const MAX_ANIMS: u32 = cfg.MAX_ANIMS;
pub const MAX_SEGS: u32 = 32;
pub const ANIM_STRIDE: u32 = 4 + MAX_SEGS * 20;

// texts: MAX_TEXTS * 64B
pub const ADDR_TEXTS: u32 = alignUp(ADDR_ANIM + MAX_ANIMS * ANIM_STRIDE, 0x1000);
pub const MAX_TEXTS: u32 = cfg.MAX_TEXTS;
pub const TEXT_STRIDE: u32 = 64;

// vertex stream: u32 count + MAX_VERTS * 14B (legacy anchor: 1 MB)
pub const ADDR_STREAM: u32 = @max(0x100000, alignUp(ADDR_TEXTS + MAX_TEXTS * TEXT_STRIDE, 0x10000));
pub const MAX_VERTS: u32 = cfg.MAX_VERTS;

// audio subsystem state (legacy anchor: 1.5 MB)
pub const ADDR_AUDIO: u32 = @max(0x180000, alignUp(ADDR_STREAM + 4 + MAX_VERTS * 14, 0x10000));
pub const AUDIO_BYTES: u32 = 0x8000;

// v7: battery-backed SRAM — 64 raw 32-bit save slots. Survives engine
// re-init (init skips this range), so hi-scores/progress persist across
// in-game restarts; hosts persist it to localStorage / files.
pub const SRAM_BYTES: u32 = 256;
pub const SRAM_SLOTS: u32 = SRAM_BYTES / 4;
pub const ADDR_SRAM: u32 = alignUp(ADDR_AUDIO + AUDIO_BYTES, 0x100);

// v8 OPTIMIZER tail: FSM bit-plane transition tables (subsystem 5) + fitted
// sfx synthesis programs (subsystem 4). Written ONLY by the v3 loader for
// v3 .libyte files; every v1/v6 anchor above is untouched. Both regions are
// zero on init — kind byte 0 = "no fitted program".
pub const ADDR_FSM: u32 = alignUp(ADDR_SRAM + SRAM_BYTES, 0x100);
pub const MAX_FSM: u32 = 16;
pub const FSM_MAX_STATES: u32 = 16;
// header: atom u16, nstates u8, span u8 — then T[state*span + events] bytes
pub const FSM_STRIDE: u32 = 4 + FSM_MAX_STATES * FSM_MAX_STATES;

// fitted sfx programs, parallel to the ADDR_SFX base records (same sound-
// domain position index). One 28B record per program, all zero when absent:
//   fb+0  kind u8 (0 none, 4 = FM, 5 = additive)
//   fb+4  f0 u16 (Hz)
//   FM:   fb+8 sweep i16 (Hz/s), fb+10 ratio_q4, fb+11 index_q4,
//         fb+12 decay (ticks), fb+13 vol (0..15), fb+14 noise (0..16)
//   ADD:  fb+12 decay, fb+13 vol, fb+15 npart (1..4),
//         fb+16+i*4 (ratio_q6 u8, amp u8, phase u8, pad u8)
pub const ADDR_SFX_FIT: u32 = ADDR_FSM + MAX_FSM * FSM_STRIDE;
pub const SFX_FIT_STRIDE: u32 = 28;

// v10 SUBSYSTEM 10: compile-proven parallel systems. The .libyte ships the
// DISJOINT GROUPS (UpdateEnt fn indices the compiler proved side-effect-free
// and data-disjoint); the engine may run each group's members on concurrent
// workers — bit-identical to any sequential order by the interchange law.
pub const MAX_PAR_GROUPS: u32 = 8;
pub const PAR_MAX_MEMBERS: u32 = 8;
pub var n_par_groups: u32 = 0;
pub var par_groups: [MAX_PAR_GROUPS * PAR_MAX_MEMBERS]u16 = [_]u16{0} ** (MAX_PAR_GROUPS * PAR_MAX_MEMBERS);
pub var par_group_len: [MAX_PAR_GROUPS]u8 = [_]u8{0} ** MAX_PAR_GROUPS;

pub inline fn parMember(g: u32, m: u32) u16 {
    return par_groups[g * PAR_MAX_MEMBERS + m];
}

pub const MAP_END: u32 = ADDR_SFX_FIT + cfg.MAX_SFX * SFX_FIT_STRIDE;
comptime {
    if (MEM_SIZE < MAP_END)
        @compileError("LILA engine profile too small: memory map needs " ++
            "at least " ++ std.fmt.comptimePrint("{d}", .{(MAP_END + 1024 * 1024 - 1) / (1024 * 1024)}) ++
            " MB (profile '" ++ cfg.PROFILE_NAME ++ "' declares " ++
            std.fmt.comptimePrint("{d}", .{cfg.MEM_MB}) ++ " MB)");
}

// ---------------- v11: ARRAYS ----------------
// Fixed-capacity arrays live in their own planes AFTER the v8 map tail, so
// every v1..v10 anchor is untouched and legacy memory hashes are stable.
//   global arrays: cap*4 bytes each, packed in id order
//   entity arrays: max_live(type) rows of cap*4 bytes, per array
// Cells are raw 32-bit words (the VM value model): one load/store path,
// zero bit-twiddling on the hot path. Width sub-word packing is a future
// optimizer concern, not a runtime one.
pub const MAX_ARRS: u32 = 64;
pub const ADDR_GARR: u32 = alignUp(MAP_END, 0x1000);
pub const GARR_BYTES: u32 = 0x20000; // 128 KB global array plane
pub const ADDR_EARR: u32 = ADDR_GARR + GARR_BYTES;
pub const EARR_BYTES: u32 = 0x40000; // 256 KB entity array plane
comptime {
    if (ADDR_EARR + EARR_BYTES > MEM_SIZE)
        @compileError("arrays planes exceed the memory map: raise mem_mb or shrink garr/earr planes");
}

// schema written by the loader from the .libyte arrays table (id order)
pub var n_arrs: u32 = 0;
pub var arr_ent: [MAX_ARRS]u8 = [_]u8{0xFF} ** MAX_ARRS; // 0xFF = global
pub var arr_cap: [MAX_ARRS]u16 = [_]u16{0} ** MAX_ARRS;
pub var arr_base: [MAX_ARRS]u32 = [_]u32{0} ** MAX_ARRS; // element 0 (entity: slot 0 row)

/// Address of element `i` of global array `id` (index pre-masked by caller).
pub inline fn garrElem(id: u32, i: u32) u32 {
    return arr_base[id] + i * 4;
}

/// Base address of entity-array `id` for slot `idx` (the entity ROW index).
pub inline fn earrRow(id: u32, idx: u32) u32 {
    return arr_base[id] + idx * @as(u32, arr_cap[id]) * 4;
}

/// Bounds safety net: power-of-two capacities mask (one AND), everything
/// else modulo-wraps. Negative (wrapped-huge) indices wrap to the top —
/// deterministic, allocation-free, trap-free, exactly like SRAM slots.
pub inline fn arrIndex(id: u32, idx: i32) u32 {
    const cap: u32 = arr_cap[id];
    const iu: u32 = @bitCast(idx);
    if (cap == 0) return 0;
    if ((cap & (cap - 1)) == 0) return iu & (cap - 1);
    return iu % cap;
}

// ---------------- v11: 3D projection (fixed-point, pure) ----------------

pub const PROJ_NEAR: i32 = 64; // Q24.8: 0.25 world units — behind => !ok

pub const ProjResult = struct { sx: i32, sy: i32, scale: i32, ok: bool };

/// Perspective-project a world point (Q24.8 xyz) through the cam3 registers.
/// Pure integer math + the shared ang LUT: bit-identical on every runtime,
/// and stashable in page 0 so rewind/rollback capture it for free.
///   view: un-rotate yaw, un-rotate pitch -> (right, forward, up)
///   screen: sx = cx + right*focal/fwd; sy = cy - up*focal/fwd
///   scale = focal/fwd (feed it to draw() scale for billboard sprites)
pub fn proj3(px: i32, py: i32, pz: i32) ProjResult {
    const cx: i32 = @bitCast(rd32(ADDR_CAM3_X));
    const cy: i32 = @bitCast(rd32(ADDR_CAM3_Y));
    const cz: i32 = @bitCast(rd32(ADDR_CAM3_Z));
    const yaw: i32 = @bitCast(rd32(ADDR_CAM3_YAW));
    const pitch: i32 = @bitCast(rd32(ADDR_CAM3_PITCH));
    const dx = px -% cx;
    const dy = py -% cy;
    const dz = pz -% cz;
    const syaw = sinA(yaw);
    const cyaw = cosA(yaw);
    // un-yaw (R(-yaw)): right = dx*cos - dy*sin; fwd0 = dx*sin + dy*cos
    const right = mulF(dx, cyaw) -% mulF(dy, syaw);
    const fwd0 = mulF(dx, syaw) +% mulF(dy, cyaw);
    // un-pitch: fwd = fwd0*cos + dz*sin; up = dz*cos - fwd0*sin
    const spit = sinA(pitch);
    const cpit = cosA(pitch);
    const fwd = mulF(fwd0, cpit) +% mulF(dz, spit);
    const up = mulF(dz, cpit) -% mulF(fwd0, spit);
    if (fwd < PROJ_NEAR) {
        return .{ .sx = 0, .sy = 0, .scale = 0, .ok = false };
    }
    const focal: i32 = @intCast((game_w / 2) << 8); // fov 90 degrees
    const half_w: i32 = @intCast((game_w / 2) << 8);
    const half_h: i32 = @intCast((game_h / 2) << 8);
    const sx = half_w +% divF(mulF(right, focal), fwd);
    const sy = half_h -% divF(mulF(up, focal), fwd);
    const scale = divF(focal, fwd);
    return .{ .sx = sx, .sy = sy, .scale = scale, .ok = true };
}

// ---------------- flat memory access ----------------

pub inline fn rd32(addr: u32) u32 {
    return std.mem.readInt(u32, mem[addr..][0..4], .little);
}
pub inline fn wr32(addr: u32, v: u32) void {
    std.mem.writeInt(u32, mem[addr..][0..4], v, .little);
}
pub inline fn rd16(addr: u32) u16 {
    return std.mem.readInt(u16, mem[addr..][0..2], .little);
}
pub inline fn wr16(addr: u32, v: u16) void {
    std.mem.writeInt(u16, mem[addr..][0..2], v, .little);
}
pub inline fn rd8(addr: u32) u8 {
    return mem[addr];
}
pub inline fn wr8(addr: u32, v: u8) void {
    mem[addr] = v;
}

// ---------------- typed-field registry access ----------------
// tf records (at TF_BASE, derived from the profile): ent u8, pad u8,
// off_bits u16, bits u8, vt u8, cold u8, pad[2], default i32 — MAX_TF x TF_SIZE

pub var n_tfs: u32 = 0;
pub var n_entities: u32 = 0;
pub var n_fns: u32 = 0;
pub var n_tables: u32 = 0;
pub var n_scenes: u32 = 0;
pub var n_sprites: u32 = 0;
pub var n_tracks: u32 = 0;
pub var n_sfx: u32 = 0;
pub var n_anims: u32 = 0;
pub var n_texts: u32 = 0;
pub var n_keys: u32 = 0;
pub var game_w: u32 = 512;
pub var game_h: u32 = 512;
pub var game_wrap: bool = false;
// v7: simulation bounds (world > screen = scrolling world). Loader sets
// these from the v2 header; default = screen (v1 behavior).
pub var game_world_w: u32 = 512;
pub var game_world_h: u32 = 512;
pub var pool_len: u32 = 0;
pub var instr_count: u32 = 0;
pub var n_fsms: u32 = 0; // v8: bit-plane transition tables loaded from a v3 file

pub inline fn tfEnt(i: u32) u8 {
    return rd8(TF_BASE + i * TF_SIZE);
}
pub inline fn tfOff(i: u32) u16 {
    return rd16(TF_BASE + i * TF_SIZE + 2);
}
pub inline fn tfBits(i: u32) u8 {
    return rd8(TF_BASE + i * TF_SIZE + 4);
}
pub inline fn tfVt(i: u32) u8 {
    return rd8(TF_BASE + i * TF_SIZE + 5);
}
pub inline fn tfCold(i: u32) u8 {
    return rd8(TF_BASE + i * TF_SIZE + 6);
}

// ---------------- entity schema (per type) ----------------
// per .libyte entity: row_bytes u16, cold_bytes u16, max_live u16 at
// ENT_SCHEMA_BASE (profile-derived); integration pairs at INTEG_BASE.

pub inline fn integCount(t: u32) u8 {
    return rd8(INTEG_BASE + t * INTEG_STRIDE);
}
pub inline fn integPos(t: u32, k: u32) u16 {
    return rd16(INTEG_BASE + t * INTEG_STRIDE + 4 + k * 10);
}
pub inline fn integVel(t: u32, k: u32) u16 {
    return rd16(INTEG_BASE + t * INTEG_STRIDE + 4 + k * 10 + 2);
}
pub inline fn integMask(t: u32, k: u32) i32 {
    return @bitCast(rd32(INTEG_BASE + t * INTEG_STRIDE + 4 + k * 10 + 4));
}

pub inline fn rowBytes(t: u32) u32 {
    return rd16(ENT_SCHEMA_BASE + t * 6);
}
pub inline fn coldBytes(t: u32) u32 {
    return rd16(ENT_SCHEMA_BASE + t * 6 + 2);
}
pub inline fn maxLive(t: u32) u32 {
    const v = rd16(ENT_SCHEMA_BASE + t * 6 + 4);
    return @min(v, MAX_ENT);
}

// decoded instruction access
pub inline fn iOp(i: u32) u8 {
    return rd8(ADDR_CODE + i * INSTR_SIZE);
}
pub inline fn iA(i: u32) i32 {
    return @bitCast(rd32(ADDR_CODE + i * INSTR_SIZE + 4));
}
pub inline fn iB(i: u32) i32 {
    return @bitCast(rd32(ADDR_CODE + i * INSTR_SIZE + 8));
}
pub inline fn iC(i: u32) i32 {
    return @bitCast(rd32(ADDR_CODE + i * INSTR_SIZE + 12));
}
pub inline fn iExtra(i: u32) u32 {
    return rd32(ADDR_CODE + i * INSTR_SIZE + 16);
}

// fn record access
pub inline fn fnKind(i: u32) u8 {
    return rd8(ADDR_FNS + i * FN_SIZE);
}
pub inline fn fnEntity(i: u32) u8 {
    return rd8(ADDR_FNS + i * FN_SIZE + 1);
}
pub inline fn fnParams(i: u32) u8 {
    return rd8(ADDR_FNS + i * FN_SIZE + 2);
}
pub inline fn fnLocals(i: u32) u8 {
    return rd8(ADDR_FNS + i * FN_SIZE + 3);
}
pub inline fn fnStart(i: u32) u32 {
    return rd32(ADDR_FNS + i * FN_SIZE + 4);
}
pub inline fn fnLen(i: u32) u32 {
    return rd32(ADDR_FNS + i * FN_SIZE + 8);
}

// ---------------- Q24.8 fixed point ----------------

pub inline fn mulF(a: i32, b: i32) i32 {
    const r: i64 = @as(i64, a) * @as(i64, b);
    return @as(i32, @truncate(r >> 8));
}
pub inline fn divF(a: i32, b: i32) i32 {
    if (b == 0) return 0;
    const r: i64 = (@as(i64, a) << 8);
    return @as(i32, @truncate(@divTrunc(r, @as(i64, b))));
}

// ---------------- comptime CTFE trig LUTs ----------------
// 4096-entry sine table in Q15 (i16), 360 degrees = 4096 ang units.

pub const SIN_LUT: [4096]i16 = blk: {
    @setEvalBranchQuota(400000);
    var t: [4096]i16 = undefined;
    for (0..4096) |i| {
        const deg: f64 = @as(f64, @floatFromInt(i)) * (360.0 / 4096.0);
        const s: f64 = @sin(deg * std.math.pi / 180.0);
        const v: i32 = @intFromFloat(s * 32767.0);
        t[i] = @intCast(v);
    }
    break :blk t;
};

pub inline fn sinA(ang: i32) i32 {
    // bitCast (not intCast): negative angles wrap two's-complement-style —
    // sin(-1 ang) == sin(4095 ang), exactly. For 0 <= ang < 2^31 the bit
    // pattern is identical, so v1..v10 outputs are bit-unchanged.
    const idx: u32 = @as(u32, @bitCast(ang)) & 0xFFF;
    return @as(i32, SIN_LUT[idx]) >> 7; // Q15 -> Q24.8
}
pub inline fn cosA(ang: i32) i32 {
    const idx: u32 = (@as(u32, @bitCast(ang)) +% 1024) & 0xFFF;
    return @as(i32, SIN_LUT[idx]) >> 7;
}

// ---------------- deterministic LCG ----------------

pub inline fn randNext() u32 {
    const s = rd32(ADDR_RAND);
    const n = s *% 1664525 +% 1013904223;
    wr32(ADDR_RAND, n);
    return n;
}
pub inline fn randMax(maxv: i32) i32 {
    if (maxv <= 0) return 0;
    const r = randNext() >> 8; // 24-bit
    return @intCast(r % @as(u32, @intCast(maxv)));
}

// ---------------- error codes ----------------

pub const ERR_OK: i32 = 0;
pub const ERR_MAGIC: i32 = -1;
pub const ERR_VERSION: i32 = -2;
pub const ERR_TOO_BIG: i32 = -3; // blob larger than the staging buffer
pub const ERR_TRUNCATED: i32 = -4;
pub const ERR_HUFF: i32 = -5;
pub const ERR_BAD_HEADER: i32 = -6;

// limit errors (v6): a production engine fails LOUDLY with a precise code
// instead of silently clamping a game that outgrew its build profile.
pub const ERR_LIMIT_ENTITIES: i32 = -7; // game needs more entity slots than MAX_ENT
pub const ERR_LIMIT_FNS: i32 = -8;
pub const ERR_LIMIT_TABLES: i32 = -9;
pub const ERR_LIMIT_SCENES: i32 = -10;
pub const ERR_LIMIT_TEXTS: i32 = -11;
pub const ERR_LIMIT_SPRITES: i32 = -12;
pub const ERR_LIMIT_SPR_VERTS: i32 = -13;
pub const ERR_LIMIT_MUSIC: i32 = -14;
pub const ERR_LIMIT_SFX: i32 = -15;
pub const ERR_LIMIT_ANIMS: i32 = -16;
pub const ERR_LIMIT_TF: i32 = -17;
pub const ERR_LIMIT_POOL: i32 = -18;
pub const ERR_LIMIT_INSTR: i32 = -19;
pub const ERR_LIMIT_EXTRA: i32 = -20;
pub const ERR_LIMIT_TYPES: i32 = -21; // more than 16 entity types (structural)
pub const ERR_LIMIT_FSM: i32 = -22; // more FSM tables than this build carries (v8)
pub const ERR_LIMIT_ARRAYS: i32 = -23; // arrays exceed this build's planes (v11)

// ---------------- engine diagnostics (v6) ----------------
// Runtime counters surfaced to hosts via wasm exports — the same class of
// telemetry a native engine logs. Reset only by engine re-init.
pub var last_error: i32 = 0; // loader rc of the most recent load attempt
pub var stream_drop_total: u32 = 0; // vertices dropped: vertex budget exceeded
pub var spawn_denied_total: u32 = 0; // spawns dropped: entity pool / cap full

// v7: set by OP_SAVE, read-and-cleared by the host via lila_sram_dirty()
// (hosts persist the SRAM block only when it actually changed).
pub var sram_dirty: u32 = 0;

// ---------------- v7 camera clamping ----------------

/// Convert a requested viewport CENTER (Q24.8 world coords, as written by
/// camera(x, y)) into the viewport TOP-LEFT register value the renderer
/// offsets by. world == screen pins the camera at (0, 0) — bit-exact v1
/// screen-space behavior even if the game calls camera(). cam_x/cam_y
/// expressions expose this top-left (spawn-relative-to-view math).
pub fn clampCam(cx: i32, cy: i32) struct { x: i32, y: i32 } {
    const sw: i32 = @intCast(game_w);
    const sh: i32 = @intCast(game_h);
    const ww: i32 = @intCast(game_world_w);
    const wh: i32 = @intCast(game_world_h);
    // requested top-left = center - half viewport
    var lx = cx -% (sw << 7);
    var ly = cy -% (sh << 7);
    if (ww <= sw) {
        lx = 0; // world == screen (or smaller): fixed viewport, offset 0
    } else {
        lx = @min(@max(lx, 0), (ww << 8) -% (sw << 8));
    }
    if (wh <= sh) {
        ly = 0;
    } else {
        ly = @min(@max(ly, 0), (wh << 8) -% (sh << 8));
    }
    return .{ .x = lx, .y = ly };
}

/// Camera top-left in integer screen pixels (render offset).
pub inline fn camPx() struct { x: i32, y: i32 } {
    return .{ .x = @as(i32, @bitCast(rd32(ADDR_CAM_X))) >> 8, .y = @as(i32, @bitCast(rd32(ADDR_CAM_Y))) >> 8 };
}

// ---------------- v7 integer math: CORDIC atan2 + isqrt ----------------

/// atan(2^-i) in ang units (4096 = 360 degrees), comptime-evaluated.
const ATAN_TAB: [15]i32 = blk: {
    @setEvalBranchQuota(10000);
    var t: [15]i32 = undefined;
    for (0..15) |i| {
        const xi: i64 = @as(i64, 1) << @intCast(i);
        const v: f64 = std.math.atan(1.0 / @as(f64, @floatFromInt(xi)));
        // ang units: 4096 per 360 degrees
        t[i] = @intFromFloat(v * 4096.0 / (std.math.pi * 2.0));
    }
    break :blk t;
};

/// Screen-convention atan2: returns the ang (0 = up, clockwise, 4096 = 360
/// degrees) whose sin/cos match the direction vector (dx, dy) under the
/// engine's screen convention (vx = sin(rot), vy = -cos(rot)). The angle of
/// the math vector (x = -dy, y = dx), computed by pure integer CORDIC
/// vectoring — deterministic, no libm. Inputs are pre-scaled to ~2^24 so all
/// 15 micro-rotations are effective (sub-ang-unit accuracy for any world).
pub fn atan2ang(dy: i32, dx: i32) i32 {
    if (dx == 0 and dy == 0) return 0;
    // math vector: cos-side = -dy, sin-side = dx
    var x: i32 = -%dy;
    var y: i32 = dx;
    // scale up so the larger component sits near bit 24 (shifts in the
    // CORDIC loop then never truncate to zero before iteration 14)
    const m: u32 = @max(@abs(x), @abs(y));
    const shift: u5 = @intCast(@max(0, 24 - @as(i32, @intCast(32 - @clz(m)))));
    x <<= shift;
    y <<= shift;
    var ang: i32 = 0;
    if (x < 0) {
        x = -%x;
        y = -%y;
        ang = 2048; // rotated by 180 degrees
    }
    var i: u5 = 0;
    while (i < 15) : (i += 1) {
        const xs = x >> i;
        const ys = y >> i;
        if (y > 0) {
            x = x +% ys;
            y = y -% xs;
            ang +%= ATAN_TAB[i];
        } else if (y < 0) {
            x = x -% ys;
            y = y +% xs;
            ang -%= ATAN_TAB[i];
        }
        // y == 0: residual angle is exactly zero (x >= 0 half-plane) — no
        // rotation needed; further iterations would only oscillate.
    }
    return ang & 0xFFF;
}

/// Integer square root of a u64 — the internal for dist(). Classic exact
/// digit-by-digit method (no floating point, no convergence tuning).
pub fn isqrt64(v_in: u64) u32 {
    var v = v_in;
    var res: u64 = 0;
    var bit: u64 = @as(u64, 1) << 62; // highest power of four <= 2^63
    while (bit > v) bit >>= 2;
    while (bit != 0) {
        if (v >= res + bit) {
            v -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    return @intCast(res);
}

/// Q24.8 euclidean distance, computed in i64 (cannot overflow for any
/// world up to 32768 px — the reason this is an engine builtin).
pub fn distF(x1: i32, y1: i32, x2: i32, y2: i32) i32 {
    const dx: i64 = @as(i64, x2) - @as(i64, x1);
    const dy: i64 = @as(i64, y2) - @as(i64, y1);
    const d = isqrt64(@abs(dx) * @abs(dx) + @abs(dy) * @abs(dy));
    return @bitCast(d);
}

// ---------------- opcodes ----------------

pub const OP_NOP: u8 = 0x00;
pub const OP_PUSH_S8: u8 = 0x01;
pub const OP_PUSH_POOL: u8 = 0x02;
pub const OP_LD_LOCAL: u8 = 0x03;
pub const OP_ST_LOCAL: u8 = 0x04;
pub const OP_LD_GLBL: u8 = 0x05;
pub const OP_ST_GLBL: u8 = 0x06;
pub const OP_LD_ENT: u8 = 0x07;
pub const OP_ST_ENT: u8 = 0x08;
pub const OP_ADD_F: u8 = 0x09;
pub const OP_SUB_F: u8 = 0x0A;
pub const OP_MUL_F: u8 = 0x0B;
pub const OP_DIV_F: u8 = 0x0C;
pub const OP_ADD_I: u8 = 0x0D;
pub const OP_SUB_I: u8 = 0x0E;
pub const OP_MUL_I: u8 = 0x0F;
pub const OP_DIV_I: u8 = 0x10;
pub const OP_MOD_I: u8 = 0x11;
pub const OP_AND_I: u8 = 0x12;
pub const OP_OR_I: u8 = 0x13;
pub const OP_XOR_I: u8 = 0x14;
pub const OP_SHL: u8 = 0x15;
pub const OP_SHR: u8 = 0x16;
pub const OP_AND_B: u8 = 0x17;
pub const OP_OR_B: u8 = 0x18;
pub const OP_NOT_B: u8 = 0x19;
pub const OP_NEG: u8 = 0x1A;
pub const OP_EQ_F: u8 = 0x1B;
pub const OP_NE_F: u8 = 0x1C;
pub const OP_LT_F: u8 = 0x1D;
pub const OP_GT_F: u8 = 0x1E;
pub const OP_LE_F: u8 = 0x1F;
pub const OP_GE_F: u8 = 0x20;
pub const OP_EQ_I: u8 = 0x21;
pub const OP_NE_I: u8 = 0x22;
pub const OP_LT_I: u8 = 0x23;
pub const OP_GT_I: u8 = 0x24;
pub const OP_LE_I: u8 = 0x25;
pub const OP_GE_I: u8 = 0x26;
pub const OP_SIN: u8 = 0x27;
pub const OP_COS: u8 = 0x28;
pub const OP_RAND_MAX: u8 = 0x29;
pub const OP_ANIM: u8 = 0x2A;
pub const OP_KEY: u8 = 0x2B;
pub const OP_JMP: u8 = 0x2C;
pub const OP_JZ: u8 = 0x2D;
pub const OP_JNZ: u8 = 0x2E;
pub const OP_FOR_BGN: u8 = 0x2F;
pub const OP_FOR_ADV: u8 = 0x30;
pub const OP_SPAWN: u8 = 0x31;
pub const OP_KILL: u8 = 0x32;
pub const OP_SFX: u8 = 0x33;
pub const OP_MUSIC: u8 = 0x34;
pub const OP_STOP_MUSIC: u8 = 0x35;
pub const OP_GOTO: u8 = 0x36;
pub const OP_DRAW: u8 = 0x37;
pub const OP_DRAW_TEXT: u8 = 0x38;
pub const OP_DRAW_NUM: u8 = 0x39;
pub const OP_CALL_TBL: u8 = 0x3A;
pub const OP_COUNT: u8 = 0x3B;
pub const OP_RET: u8 = 0x3C;
pub const OP_CALL_FN: u8 = 0x3D;
pub const OP_I2F: u8 = 0x3E;
pub const OP_F2I: u8 = 0x3F;
pub const OP_SHAKE: u8 = 0x40;

// ---- v7 opcodes: camera / persistence / aim math / music volume ----
pub const OP_CAMERA: u8 = 0x41;    // pops y, x (fixed) -> clamped camera center
pub const OP_MUSIC_VOL: u8 = 0x42; // u8 master music volume 0..16
pub const OP_SAVE: u8 = 0x43;      // pops v, ix -> SRAM slot (ix & 63)
pub const OP_SAVED: u8 = 0x44;     // pops ix -> pushes slot value (raw i32)
pub const OP_DIST: u8 = 0x45;     // pops y2 x2 y1 x1 -> pushes Q24.8 distance
pub const OP_ATAN2: u8 = 0x46;    // pops dx, dy -> pushes screen-convention ang
pub const OP_CAM_X: u8 = 0x47;    // pushes camera center x (fixed)
pub const OP_CAM_Y: u8 = 0x48;    // pushes camera center y (fixed)

// ---- v8 opcodes: optimizer-baked physics / AI (the compiler emits these;
// user code never writes them directly) ----
pub const OP_SWEPT: u8 = 0x49;         // pops hh hw by bx avy avx ay ax -> bool swept Minkowski interval
pub const OP_COLLIDE_MASK: u8 = 0x4A;  // pops cy cx -> u32 live-slot bitmask (Galois 32 lanes); a=type, b/c=pool hw/hh, extra=[x_tf,y_tf]
pub const OP_FOR_MASK_BGN: u8 = 0x4B;  // pops mask -> iterate set slots (a=local slot, b=type, c=exit rel)
pub const OP_FOR_MASK_ADV: u8 = 0x4C;  // advance ctz scan (a=body rel)
pub const OP_FSM_NEXT: u8 = 0x4D;      // pops s, em -> next = T[s*span + em] (bit-plane transition; a=table)
pub const OP_SEL: u8 = 0x4E;           // pops else, then, cond -> branchless conditional move

// ---- v11 opcodes: arrays + 3D ----
pub const OP_LD_GARR: u8 = 0x4F;       // a=arr id; pops idx -> pushes elem (global)
pub const OP_ST_GARR: u8 = 0x50;       // a=arr id; pops val, idx -> store (mask|mod bounds)
pub const OP_LD_EARR: u8 = 0x51;       // a=slot, b=arr id; pops idx -> pushes elem
pub const OP_ST_EARR: u8 = 0x52;       // a=slot, b=arr id; pops val, idx -> store
pub const OP_DUP: u8 = 0x53;           // duplicate top of stack
pub const OP_CAM3: u8 = 0x54;          // pops pitch, yaw, z, y, x -> cam3 registers
pub const OP_PROJ3: u8 = 0x55;         // pops z, y, x -> scale (+ stash x/y/ok)
pub const OP_PROJ_X: u8 = 0x56;        // pushes stashed screen x
pub const OP_PROJ_Y: u8 = 0x57;        // pushes stashed screen y
pub const OP_PROJ_OK: u8 = 0x58;       // pushes stashed ok flag
pub const OP_DRAW3D: u8 = 0x59;        // a=arr id; pops rgba, yaw, z, y, x, n -> mesh pass

// ---- v12 opcodes: articulated 3D (mat4/quat over arrays + indexed meshes) ----
pub const OP_QUAT_AA: u8 = 0x5A;       // pops ang, az, ay, ax, off, q_aid -> q[off..+3] = axis-angle quat
pub const OP_Q_MUL: u8 = 0x5B;         // pops boff,b_aid,aoff,a_aid,doff,d_aid -> d = a*b (Hamilton)
pub const OP_M4_QT: u8 = 0x5C;         // pops tz,ty,tx,q_off,q_aid,m_off,m_aid -> M = T(t)*R(q)
pub const OP_M4_MUL: u8 = 0x5D;        // pops boff,b_aid,aoff,a_aid,doff,d_aid -> D = A*B (alias-safe)
pub const OP_SKIN3: u8 = 0x5E;         // pops z,y,x,m_off,m_aid,v_off,v_aid -> verts[voff..+2] = M*(x,y,z,1)
pub const OP_DRAW3DI: u8 = 0x5F;       // a=vert arr, b=idx arr; pops rgba, yaw, z, y, x, ni, nv -> indexed mesh pass

// v12 PROFILE KNOB (runtime mirror of engine_config.ZBUFFER3D): false =
// painter's algorithm (v11 default); true = per-pixel Z-buffer. Engine
// config, NOT game state: it lives outside c.mem, so rewind/rollback/
// snapshots/hot-swap are untouched and a given binary stays deterministic.
pub var zbuf3d: bool = cfg.ZBUFFER3D;

// v15 PROFILE KNOB (runtime mirror of engine_config.GAMMA_BLEND): false =
// blend directly in 8-bit sRGB (legacy, pixel-identical to v14); true =
// gamma-correct blending — RGB is linearized before mixing and re-encoded
// after (comptime sRGB LUTs in raster.zig; alpha stays straight coverage).
// Pixel-output-only: raster writes go to fb, never c.mem, so the state-hash
// chain is identical either way and determinism guarantees are unchanged.
pub var gamma_blend: bool = cfg.GAMMA_BLEND;

pub const FN_INIT: u8 = 0;
pub const FN_UPDATE0: u8 = 1;
pub const FN_UPDATEENT: u8 = 2;
pub const FN_DRAW0: u8 = 3;
pub const FN_DRAWENT: u8 = 4;
pub const FN_HELPER: u8 = 5;
pub const FN_SCENE: u8 = 6;
pub const FN_TABLE: u8 = 7;
