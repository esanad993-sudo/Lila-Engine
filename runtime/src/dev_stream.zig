//! Dev harness: load a .libyte, run frames, dump the vertex stream.
//! (a manual driver — the automated suites live in test_*.zig)
// Stream probe: run 1 frame, histogram the vertex stream by rgba.
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
    if (rc != 0) { std.debug.print("init rc={d}\n", .{rc}); return; }

    c.wr32(c.ADDR_INPUT, 0);
    engine.frame();

    const n = c.rd32(c.ADDR_STREAM);
    std.debug.print("stream verts: {d}\n", .{n});

    // rgba histogram
    var counts = std.AutoHashMap(u32, u32).init(std.heap.page_allocator);
    defer counts.deinit();
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const base = c.ADDR_STREAM + 4 + i * 14;
        const rgba = c.rd32(base + 8);
        const gop = try counts.getOrPut(rgba);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }
    var it = counts.iterator();
    std.debug.print("rgba histogram:\n", .{});
    while (it.next()) |e| {
        std.debug.print("  {x}: {d} verts\n", .{ e.key_ptr.*, e.value_ptr.* });
    }

    // first 12 verts
    i = 0;
    while (i < @min(n, 12)) : (i += 1) {
        const base = c.ADDR_STREAM + 4 + i * 14;
        const x: i16 = @bitCast(c.rd16(base));
        const y: i16 = @bitCast(c.rd16(base + 2));
        std.debug.print("v[{d}] x={d} y={d} rgba={x}\n", .{ i, x, y, c.rd32(base + 8) });
    }
}
