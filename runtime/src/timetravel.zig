//! Deterministic time travel: snapshot / restore / state-hash primitives.
//!
//! The whole engine state is ONE flat memory image (entities, globals, FSM
//! tables, SRAM, audio voices, ADPCM ring, music phase — everything lives
//! inside c.mem) plus a handful of host-side scalars. Therefore:
//!
//!   snapshot = memcpy(2MB)          restore = memcpy back + vm.resetVm()
//!
//! Bit-exact by construction — there is no hidden state to miss, no
//! serialization to get wrong. This is what "deterministic fantasy console"
//! buys you: rewind (Braid), rollback netcode (GGPO), TAS-grade replay
//! verification and 2MB instant save states are all the SAME 30-line
//! primitive. Unreal needs a replication graph; Unity needs Float order
//! audits; we need @memcpy.

const std = @import("std");
const c = @import("core.zig");
const vm = @import("vm.zig");

pub const SNAP_MAGIC: u32 = 0x3153414C; // "LSA1"
pub const SNAP_VERSION: u32 = 1;

/// Host-side scalars that live OUTSIDE c.mem (core.zig module vars).
pub const Scalars = struct {
    spawn_denied_total: u32,
    stream_drop_total: u32,
    sram_dirty: u32,
};

pub fn readScalars() Scalars {
    return .{
        .spawn_denied_total = c.spawn_denied_total,
        .stream_drop_total = c.stream_drop_total,
        .sram_dirty = c.sram_dirty,
    };
}

pub fn writeScalars(s: Scalars) void {
    c.spawn_denied_total = s.spawn_denied_total;
    c.stream_drop_total = s.stream_drop_total;
    c.sram_dirty = s.sram_dirty;
}

/// One full engine state, captured at a frame boundary (the VM interpreter
/// is quiescent there: no live frames, empty stack, no iterators).
pub const Snapshot = struct {
    frame: u32,
    mem: []u8,
    scalars: Scalars = .{ .spawn_denied_total = 0, .stream_drop_total = 0, .sram_dirty = 0 },

    /// Deep-copy the entire engine state. alloc must outlive the snapshot.
    pub fn capture(alloc: std.mem.Allocator, frame: u32) !Snapshot {
        const buf = try alloc.alloc(u8, c.MEM_SIZE);
        @memcpy(buf, c.mem[0..]);
        return .{ .frame = frame, .mem = buf };
    }

    /// Restore: memcpy back + reset the quiescent interpreter. Every byte
    /// of game state (including audio phase + RNG) returns to the captured
    /// instant; the next engine.frame() continues as if nothing happened.
    pub fn restore(self: Snapshot) void {
        @memcpy(c.mem[0..], self.mem);
        writeScalars(self.scalars);
        vm.resetVm();
    }
};

/// Wyhash over the full state image + scalars + frame counter. Pure integer
/// math — identical on every platform, every run, forever.
pub fn stateHash(frame: u32) u64 {
    var h = std.hash.Wyhash.init(0x6C696C61); // "lila"
    h.update(std.mem.asBytes(&frame));
    h.update(c.mem[0..]);
    const s = readScalars();
    h.update(std.mem.asBytes(&s));
    return h.final();
}

/// Serialize a snapshot to an LSA1 save-state file image.
/// Layout (header = 44B): magic u32 | version u32 | frame u32 |
///   mem_size u32 | schema_sig u64 | spawn_denied u32 | stream_drop u32 |
///   sram_dirty u32 | pad u32 | mem bytes
pub fn serialize(alloc: std.mem.Allocator, snap: Snapshot) ![]u8 {
    const out = try alloc.alloc(u8, 44 + c.MEM_SIZE);
    @memset(out[0..44], 0);
    std.mem.writeInt(u32, out[0..4], SNAP_MAGIC, .little);
    std.mem.writeInt(u32, out[4..8], SNAP_VERSION, .little);
    std.mem.writeInt(u32, out[8..12], snap.frame, .little);
    std.mem.writeInt(u32, out[12..16], c.MEM_SIZE, .little);
    std.mem.writeInt(u64, out[16..24], snapSchemaSig(), .little);
    const s = snap.scalars;
    std.mem.writeInt(u32, out[24..28], s.spawn_denied_total, .little);
    std.mem.writeInt(u32, out[28..32], s.stream_drop_total, .little);
    std.mem.writeInt(u32, out[32..36], s.sram_dirty, .little);
    @memcpy(out[44..], snap.mem[0..]);
    return out;
}

/// Schema signature captured alongside (saved by the runner via setSchemaSig).
var saved_schema_sig: u64 = 0;
pub fn setSchemaSig(sig: u64) void {
    saved_schema_sig = sig;
}
fn snapSchemaSig() u64 {
    return saved_schema_sig;
}

pub const DeserError = error{ BadMagic, BadVersion, BadSize };

/// Parse an LSA1 image into frame + scalars and copy mem back into c.mem.
pub fn deserializeInto(image: []const u8) !u32 {
    if (image.len < 44) return DeserError.BadMagic;
    if (std.mem.readInt(u32, image[0..4], .little) != SNAP_MAGIC) return DeserError.BadMagic;
    if (std.mem.readInt(u32, image[4..8], .little) != SNAP_VERSION) return DeserError.BadVersion;
    const frame = std.mem.readInt(u32, image[8..12], .little);
    const mem_size = std.mem.readInt(u32, image[12..16], .little);
    if (mem_size != c.MEM_SIZE or image.len < 44 + mem_size) return DeserError.BadSize;
    const s = Scalars{
        .spawn_denied_total = std.mem.readInt(u32, image[24..28], .little),
        .stream_drop_total = std.mem.readInt(u32, image[28..32], .little),
        .sram_dirty = std.mem.readInt(u32, image[32..36], .little),
    };
    @memcpy(c.mem[0..], image[44 .. 44 + mem_size]);
    writeScalars(s);
    vm.resetVm();
    return frame;
}
