//! dev probe: run the demo N frames, dump 2D/3D camera + pyramid stream verts.
const std = @import("std");
const c = @import("core.zig");
const engine = @import("engine.zig");
const render = @import("render.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena: std.mem.Allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) {
        std.debug.print("usage: probe_stream <game.libyte> [frames]\n", .{});
        return;
    }
    const path = args[1];
    const frames: u32 = if (args.len > 2) std.fmt.parseInt(u32, args[2], 10) catch 299 else 299;

    var dir = try std.Io.Dir.cwd().openDir(io, ".", .{});
    defer dir.close(io);
    var buf: [256 * 1024]u8 = undefined;
    const blob = dir.readFile(io, path, &buf) catch return;

    const rc = engine.init(blob);
    if (rc != 0) {
        std.debug.print("init failed: {d}\n", .{rc});
        return;
    }
    var f: u32 = 0;
    while (f < frames) : (f += 1) engine.frame();

    const camx: i32 = @bitCast(c.rd32(c.ADDR_CAM_X));
    std.debug.print("2D cam_x (top-left) = {d}.{d:0>3} px\n", .{ camx >> 8, @as(u32, @intCast(@mod(camx, 256))) });
    std.debug.print("3D cam: x={d} y={d} z={d} yaw={d} pitch={d}\n", .{
        @as(i32, @bitCast(c.rd32(c.ADDR_CAM3_X))) >> 8,
        @as(i32, @bitCast(c.rd32(c.ADDR_CAM3_Y))) >> 8,
        @as(i32, @bitCast(c.rd32(c.ADDR_CAM3_Z))) >> 8,
        @as(i32, @bitCast(c.rd32(c.ADDR_CAM3_YAW))),
        @as(i32, @bitCast(c.rd32(c.ADDR_CAM3_PITCH))),
    });
    // stream: stars (60x6 verts) come first; pyramid = verts 360..377
    const n = c.rd32(c.ADDR_STREAM);
    std.debug.print("stream verts: {d}\n", .{n});
    var i: u32 = 360;
    while (i < @min(n, 378)) : (i += 1) {
        const base = c.ADDR_STREAM + 4 + i * 14;
        const x: i32 = @as(i16, @bitCast(c.rd16(base)));
        const y: i32 = @as(i16, @bitCast(c.rd16(base + 2)));
        const tok = c.rd16(base + 12);
        std.debug.print("  v{d}: ({d},{d}) tok={d}\n", .{ i, x, y, tok });
    }
    const ent = @import("entity.zig");
    // dump ALL globals by tf registry (ent==0xFF)
    var tf: u32 = 0;
    while (tf < c.n_tfs) : (tf += 1) {
        if (c.tfEnt(tf) == 0xFF) {
            const v = ent.globalLoad(tf);
            std.debug.print("  global tf{d} [off={d} bits={d}] = {d} (fixed {d}.{d:0>3})\n", .{
                tf, c.tfOff(tf), c.tfBits(tf) & 0x7F, v, v >> 8, @as(u32, @intCast(@mod(v, 256))),
            });
        }
    }
}
