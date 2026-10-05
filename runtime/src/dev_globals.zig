//! Dev harness: load a .libyte, run frames, dump the globals block.
//! (a manual driver — the automated suites live in test_*.zig)
// Globals probe: inspect raw bits + globalLoad per tf after init and frames.
const std = @import("std");
const c = @import("core.zig");
const engine = @import("engine.zig");
const ent = @import("entity.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    var dir = try std.Io.Dir.cwd().openDir(io, ".", .{});
    defer dir.close(io);
    var buf: [256 * 1024]u8 = undefined;
    if (args.len < 2) {
        std.debug.print("usage: {s} <game.libyte>\n", .{args[0]});
        return;
    }
    const blob = try dir.readFile(io, args[1], &buf);
    const loader = @import("loader.zig");
    const rc = loader.load(blob);
    if (rc != 0) { std.debug.print("load rc={d}\n", .{rc}); return; }
    std.debug.print("after LOAD (defaults applied, no init):\n", .{});
    std.debug.print("  over(globalLoad 38)={d} palive(39)={d}\n", .{ ent.globalLoad(38), ent.globalLoad(39) });
    std.debug.print("  fieldDefault(38)={d} (39)={d}\n", .{ ent.fieldDefault(38), ent.fieldDefault(39) });
    std.debug.print("  raw 0x14..0x18: ", .{});
    var bb: u32 = 0x14;
    while (bb < 0x18) : (bb += 1) std.debug.print("{x:0>2} ", .{c.mem[bb]});
    std.debug.print("\n", .{});
    var fi: u32 = 0;
    while (fi < 8) : (fi += 1) {
        std.debug.print("fn[{d}] kind={d} entity={d} start={d} len={d}\n", .{
            fi, c.fnKind(fi), c.fnEntity(fi), c.fnStart(fi), c.fnLen(fi),
        });
    }
    const vm = @import("vm.zig");
    _ = vm;
    _ = engine.init(blob);

    std.debug.print("after init:\n", .{});
    var tf: u32 = 34;
    while (tf < 43) : (tf += 1) {
        std.debug.print("  tf[{d}] ent={d} off={d} bits={d} val={d}\n", .{
            tf, c.tfEnt(tf), c.tfOff(tf), c.tfBits(tf) & 0x7F, ent.globalLoad(tf),
        });
    }
    std.debug.print("raw globals 0x10..0x20: ", .{});
    var b: u32 = 0x10;
    while (b < 0x20) : (b += 1) std.debug.print("{x:0>2} ", .{c.mem[b]});
    std.debug.print("\n", .{});

    c.wr32(c.ADDR_INPUT, 0);
    engine.frame();
    std.debug.print("after frame 1:\n", .{});
    tf = 34;
    while (tf < 43) : (tf += 1) {
        std.debug.print("  tf[{d}] val={d}\n", .{ tf, ent.globalLoad(tf) });
    }
    std.debug.print("raw globals 0x10..0x20: ", .{});
    b = 0x10;
    while (b < 0x20) : (b += 1) std.debug.print("{x:0>2} ", .{c.mem[b]});
    std.debug.print("\n", .{});
}
