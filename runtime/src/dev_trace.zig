//! Dev harness: load a .libyte, run frames, trace per-frame state.
//! (a manual driver — the automated suites live in test_*.zig)
// Minimal init trace
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

    const rc = engine.init(blob);
    std.debug.print("init rc={d} n_fns={d} instrs={d} tfs={d} entities={d}\n", .{
        rc, c.n_fns, c.instr_count, c.n_tfs, c.n_entities,
    });
    std.debug.print("after init: player={d}\n", .{ent.count(0)});
    std.debug.print("fn0: kind={d} start={d} len={d}\n", .{ c.fnKind(0), c.fnStart(0), c.fnLen(0) });
    var k: u32 = 0;
    while (k < 6) : (k += 1) {
        std.debug.print("zi[{d}] op={x} a={d} b={d} c={d}\n", .{ k, c.iOp(k), c.iA(k), c.iB(k), c.iC(k) });
    }
    // run 5 frames
    var f: u32 = 0;
    while (f < 5) : (f += 1) {
        engine.frame();
        std.debug.print("frame {d}: player={d} stream={d} scene={d}\n", .{ f, ent.count(0), c.rd32(c.ADDR_STREAM), c.rd32(c.ADDR_SCENE) });
    }
}
