//! LILA entity system (manual section 2):
//! - u32 handle = (12-bit generation << 20) | 20-bit index
//! - dense SoA rows: entity index = row coordinate
//! - 1-bit existence bitmaps in u64 blocks, ctz scan iteration
//! - bit-packed fields (arbitrary bit boundaries, RMW via 8-byte windows)
//! - cold fields in separate slot-indexed arrays with side-table gens

const std = @import("std");
const c = @import("core.zig");

pub inline fn genBase(t: u32) u32 {
    return c.ADDR_META + t * c.META_STRIDE;
}
pub inline fn bitmapBase(t: u32) u32 {
    return genBase(t) + c.MAX_ENT * 2;
}
pub inline fn denseBase(t: u32) u32 {
    return c.ADDR_DENSE + t * c.DENSE_TYPE_STRIDE;
}
pub inline fn coldBase(t: u32) u32 {
    return c.ADDR_COLD + t * c.COLD_TYPE_STRIDE;
}
pub inline fn coldGenBase(t: u32) u32 {
    return c.ADDR_COLDGEN + t * c.MAX_ENT * 2;
}

pub fn resetAll() void {
    const total = c.MAX_TYPES * c.META_STRIDE + @as(u32, c.MAX_TYPES) * c.DENSE_TYPE_STRIDE +
        c.MAX_TYPES * c.COLD_TYPE_STRIDE + c.MAX_TYPES * c.MAX_ENT * 2;
    @memset(c.mem[c.ADDR_META .. c.ADDR_META + total], 0);
}

pub fn genOf(t: u32, idx: u32) u16 {
    return c.rd16(genBase(t) + idx * 2);
}

pub fn isAlive(t: u32, idx: u32) bool {
    const w = idx >> 6;
    const bit = idx & 63;
    const word = if (bit < 32) c.rd32(bitmapBase(t) + w * 8) else c.rd32(bitmapBase(t) + w * 8 + 4);
    return (word >> @intCast(bit & 31)) & 1 != 0;
}

/// Spawn: find a free slot via complement-bitmap ctz scan; returns handle or
/// 0xFFFFFFFF when the type is full. A denied spawn is COUNTED
/// (spawn_denied_total) — the host surfaces it, games no longer fail invisibly.
pub fn spawn(t: u32) u32 {
    if (t >= c.n_entities) return 0xFFFFFFFF;
    const bm = bitmapBase(t);
    var w: u32 = 0;
    while (w < c.BM_WORDS) : (w += 1) {
        const live = c.rd32(bm + w * 8) | (@as(u64, c.rd32(bm + w * 8 + 4)) << 32);
        const free = ~live;
        if (free != 0) {
            const bit: u32 = @ctz(free);
            const idx = w * 64 + bit;
            if (idx >= c.maxLive(t)) continue; // respect per-type cap
            const g = genOf(t, idx);
            // set bit
            if (bit < 32) {
                const v = c.rd32(bm + w * 8);
                c.wr32(bm + w * 8, v | (@as(u32, 1) << @intCast(bit)));
            } else {
                const v = c.rd32(bm + w * 8 + 4);
                c.wr32(bm + w * 8 + 4, v | (@as(u32, 1) << @intCast(bit - 32)));
            }
            // zero the dense row
            const rb = c.rowBytes(t);
            if (rb > 0) @memset(c.mem[denseBase(t) + idx * rb .. denseBase(t) + idx * rb + rb], 0);
            // zero cold row
            const cb = c.coldBytes(t);
            if (cb > 0) @memset(c.mem[coldBase(t) + idx * cb .. coldBase(t) + idx * cb + cb], 0);
            // v11: zero this type's ARRAY rows for the reused slot — a
            // respawned entity must never observe a previous occupant's
            // trail/inventory (same contract as the dense-row zeroing above).
            var ka: u32 = 0;
            while (ka < c.n_arrs) : (ka += 1) {
                if (c.arr_ent[ka] == t) {
                    const row = c.earrRow(ka, idx);
                    const row_bytes = @as(u32, c.arr_cap[ka]) * 4;
                    @memset(c.mem[row .. row + row_bytes], 0);
                }
            }
            // cold side-table gen mirrors the slot gen
            c.wr16(coldGenBase(t) + idx * 2, g);
            return (@as(u32, g) << 20) | idx;
        }
    }
    c.spawn_denied_total +%= 1;
    return 0xFFFFFFFF;
}

/// Kill: clear existence bit + increment slot generation (12-bit wrap).
/// Stale handles referencing the old generation fail their gen check for free.
pub fn kill(t: u32, idx: u32) void {
    if (t >= c.MAX_TYPES) return; // v14: crafted tf/table operands — never a trap
    if (idx >= c.MAX_ENT) return;
    const w = idx >> 6;
    const bit = idx & 63;
    const addr = bitmapBase(t) + w * 8 + (if (bit < 32) @as(u32, 0) else @as(u32, 4));
    const sh: u5 = @intCast(bit & 31);
    const v = c.rd32(addr);
    if ((v >> sh) & 1 == 0) return; // already dead
    c.wr32(addr, v & ~(@as(u32, 1) << sh));
    const g = (genOf(t, idx) +% 1) & 0xFFF;
    c.wr16(genBase(t) + idx * 2, g);
    c.wr16(coldGenBase(t) + idx * 2, g);
}

pub fn count(t: u32) u32 {
    if (t >= c.MAX_TYPES) return 0; // v14: soft-land out-of-range types
    var n: u32 = 0;
    const bm = bitmapBase(t);
    var w: u32 = 0;
    while (w < c.BM_WORDS) : (w += 1) {
        const lo = c.rd32(bm + w * 8);
        const hi = c.rd32(bm + w * 8 + 4);
        // widen BEFORE adding: @popCount(u32) is u6, and 32+32=64 overflows
        // u6 — latent until a fully-populated bitmap word exists (>512 live
        // entities), which the v6 capacity knob just made reachable
        n += @as(u32, @popCount(lo)) + @as(u32, @popCount(hi));
    }
    return n;
}

/// Iterator: find next live index >= `from`; returns 0xFFFFFFFF at end.
pub fn nextLive(t: u32, from: u32) u32 {
    if (t >= c.MAX_TYPES) return 0xFFFFFFFF; // v14: soft-land out-of-range types
    const bm = bitmapBase(t);
    var w: u32 = from >> 6;
    if (w >= c.BM_WORDS) return 0xFFFFFFFF;
    while (w < c.BM_WORDS) : (w += 1) {
        const lo = c.rd32(bm + w * 8);
        const hi = c.rd32(bm + w * 8 + 4);
        var mask_lo: u32 = 0;
        var mask_hi: u64 = 0;
        if (w == from >> 6) {
            const bit = from & 63;
            if (bit < 32) {
                mask_lo = ~((@as(u32, 1) << @intCast(bit)) -% 1);
                mask_hi = 0xFFFFFFFF;
            } else {
                mask_hi = ~((@as(u64, 1) << @intCast(bit)) -% 1);
            }
        } else {
            mask_lo = 0xFFFFFFFF;
            mask_hi = 0xFFFFFFFF;
        }
        const livelo = lo & mask_lo;
        if (livelo != 0) {
            return w * 64 + @ctz(livelo);
        }
        const livehi: u32 = @truncate((@as(u64, hi) & mask_hi));
        if (livehi != 0) {
            return w * 64 + 32 + @ctz(livehi);
        }
    }
    return 0xFFFFFFFF;
}

// ---------------- bit-packed field access ----------------
// Read/extract a field of `bits` at `off_bits` bit offset from a row base.
// Uses an unaligned 8-byte window + shift + mask. Sign-extends for vt=1 with
// top bit set (compiler marks signed via bits high-bit convention: sN fields
// get bits | 0x80 in the tf registry — see loader).

inline fn extractBits(base: u32, off_bits: u32, bits: u8) u64 {
    const byte = base + (off_bits >> 3);
    const window = std.mem.readInt(u64, c.mem[byte..][0..8], .little);
    const sh: u6 = @intCast(off_bits & 7);
    const mask: u64 = if (bits >= 64) 0xFFFFFFFFFFFFFFFF else (@as(u64, 1) << @intCast(bits)) - 1;
    return (window >> sh) & mask;
}

inline fn depositBits(base: u32, off_bits: u32, bits: u8, val: u64) void {
    const byte = base + (off_bits >> 3);
    var window = std.mem.readInt(u64, c.mem[byte..][0..8], .little);
    const sh: u6 = @intCast(off_bits & 7);
    const mask: u64 = if (bits >= 64) 0xFFFFFFFFFFFFFFFF else (@as(u64, 1) << @intCast(bits)) - 1;
    window &= ~(mask << sh);
    window |= (val & mask) << sh;
    std.mem.writeInt(u64, c.mem[byte..][0..8], window, .little);
}

pub fn fieldLoad(tf: u32, handle: u32) i32 {
    if (tf >= c.n_tfs) return 0; // v14: VM masks tf to u16 — soft-land reads
    const t: u32 = c.tfEnt(tf);
    const idx: u32 = handle & 0xFFFFF;
    const gen: u32 = (handle >> 20) & 0xFFF;
    if (idx >= c.MAX_ENT) return 0;
    // generation check: stale reference reads as 0 (never crashes)
    if (genOf(t, idx) != gen) return 0;
    if (!isAlive(t, idx)) return 0;

    const bits = c.tfBits(tf) & 0x7F;
    const signed = (c.tfBits(tf) & 0x80) != 0;
    const off: u32 = c.tfOff(tf);
    const base = if (c.tfCold(tf) != 0)
        coldBase(t) + idx * c.coldBytes(t)
    else
        denseBase(t) + idx * c.rowBytes(t);

    const raw = extractBits(base, off, bits);
    if (signed) {
        // sign-extend from `bits`
        const sh: u6 = @intCast(64 - bits);
        return @as(i32, @truncate((@as(i64, @bitCast(raw)) << sh) >> sh));
    }
    return @bitCast(@as(u32, @truncate(raw)));
}

pub fn fieldStore(tf: u32, handle: u32, val: i32) void {
    if (tf >= c.n_tfs) return; // v14: VM masks tf to u16 — soft-land writes
    const t: u32 = c.tfEnt(tf);
    const idx: u32 = handle & 0xFFFFF;
    const gen: u32 = (handle >> 20) & 0xFFF;
    if (idx >= c.MAX_ENT) return;
    if (genOf(t, idx) != gen) return;
    if (!isAlive(t, idx)) return;

    const bits = c.tfBits(tf) & 0x7F;
    const off: u32 = c.tfOff(tf);
    const base = if (c.tfCold(tf) != 0)
        coldBase(t) + idx * c.coldBytes(t)
    else
        denseBase(t) + idx * c.rowBytes(t);
    depositBits(base, off, bits, @as(u64, @bitCast(@as(i64, val))) & 0xFFFFFFFF);
}

pub fn fieldDefault(tf: u32) i32 {
    return @bitCast(c.rd32(c.TF_BASE + tf * c.TF_SIZE + 8));
}

/// Initialize ALL fields of a freshly spawned entity to schema defaults.
pub fn applyDefaults(t: u32, handle: u32) void {
    var tf: u32 = 0;
    while (tf < c.n_tfs) : (tf += 1) {
        if (c.tfEnt(tf) == t) {
            const d = fieldDefault(tf);
            fieldStore(tf, handle, d);
        }
    }
}

pub fn globalLoad(tf: u32) i32 {
    if (tf >= c.n_tfs) return 0; // v14: soft-land out-of-registry reads
    const bits = c.tfBits(tf) & 0x7F;
    const signed = (c.tfBits(tf) & 0x80) != 0;
    const raw = extractBits(c.ADDR_GLOBALS, c.tfOff(tf), bits);
    if (signed) {
        const sh: u6 = @intCast(64 - bits);
        return @as(i32, @truncate((@as(i64, @bitCast(raw)) << sh) >> sh));
    }
    return @bitCast(@as(u32, @truncate(raw)));
}

pub fn globalStore(tf: u32, val: i32) void {
    if (tf >= c.n_tfs) return; // v14: soft-land out-of-registry writes
    const bits = c.tfBits(tf) & 0x7F;
    depositBits(c.ADDR_GLOBALS, c.tfOff(tf), bits, @as(u64, @bitCast(@as(i64, val))) & 0xFFFFFFFF);
}

pub fn globalsReset() void {
    @memset(c.mem[c.ADDR_GLOBALS .. c.ADDR_GLOBALS + c.GLOBALS_BYTES], 0);
    var tf: u32 = 0;
    while (tf < c.n_tfs) : (tf += 1) {
        if (c.tfEnt(tf) == 0xFF) {
            globalStore(tf, fieldDefault(tf));
        }
    }
}
