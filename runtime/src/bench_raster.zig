//! dev bench: isolate rasterizer + core-engine frame performance.
//! usage: bench_raster <game.libyte> [--gamma]
const std = @import("std");
const c = @import("core.zig");
const engine = @import("engine.zig");
const render = @import("render.zig");
const raster = @import("raster.zig");
const audio = @import("audio.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena: std.mem.Allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) {
        std.debug.print("usage: bench_raster <game.libyte> [--gamma]\n", .{});
        return;
    }
    for (args[1..]) |a| {
        if (std.mem.eql(u8, a, "--gamma")) c.gamma_blend = true;
    }
    var dir = try std.Io.Dir.cwd().openDir(io, ".", .{});
    defer dir.close(io);
    var buf: [256 * 1024]u8 = undefined;
    const blob = dir.readFile(io, args[1], &buf) catch return;
    if (engine.init(blob) != 0) return;

    // warm up: 300 frames of real gameplay (stream fills with stars/sprites/text)
    var f: u32 = 0;
    while (f < 300) : (f += 1) {
        c.wr32(c.ADDR_INPUT, 0);
        engine.frame();
        _ = audio.pull(&[_]i16{0} ** 0);
    }
    const verts = c.rd32(c.ADDR_STREAM);

    // rasterize() N times (includes clear + full stream)
    const N: u32 = 400;
    const t0 = engine.nowNs();
    var i: u32 = 0;
    var sink: u64 = 0;
    while (i < N) : (i += 1) {
        raster.rasterize();
        sink += raster.fb[0] + raster.fb[raster.fb.len - 1];
    }
    const t1 = engine.nowNs();
    const ns = @as(u64, @intCast(t1 - t0));
    std.debug.print("rasterize: {} frames x {} verts -> {d:.3} ms/frame ({d:.1} Mpx/s sink={})\n", .{
        N,                       verts,
        @as(f64, @floatFromInt(ns)) / 1e6 / N, @as(f64, 512 * 512) * @as(f64, @floatFromInt(N)) / (@as(f64, @floatFromInt(ns)) / 1e9),
        sink,
    });

    // engine.frame() benchmark (VM + stream emission, no raster)
    const M: u32 = 3000;
    const t2 = engine.nowNs();
    i = 0;
    while (i < M) : (i += 1) {
        c.wr32(c.ADDR_INPUT, 0);
        engine.frame();
    }
    const t3 = engine.nowNs();
    const ns2 = @as(u64, @intCast(t3 - t2));
    std.debug.print("engine.frame: {} frames -> {d:.1} us/frame\n", .{
        M, @as(f64, @floatFromInt(ns2)) / 1e3 / M,
    });
}
