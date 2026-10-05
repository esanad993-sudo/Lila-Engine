// probe3: evaluate the compiled #wave_curve at every 16th step to prove
// the RDP compressor preserved the weave amplitude (the Task-19 bug class).
const std = @import("std");
const Io = std.Io;
const c = @import("core.zig");
const engine = @import("engine.zig");
const vm = @import("vm.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const in_path = if (args.len > 1) args[1] else "game.libyte";

    var dir = try std.Io.Dir.cwd().openDir(io, ".", .{});
    defer dir.close(io);
    var blob_buf: [256 * 1024]u8 = undefined;
    const blob = dir.readFile(io, in_path, &blob_buf) catch |e| {
        std.debug.print("cannot read {s}: {}\n", .{ in_path, e });
        return;
    };
    const rc = engine.init(blob);
    if (rc != 0) {
        std.debug.print("init failed: {d}\n", .{rc});
        return;
    }
    std.debug.print("curve #wave_curve (raw Q24.8; 1024 = 4px):\n", .{});
    var t: i32 = 0;
    var vmin: i32 = 1 << 30;
    var vmax: i32 = -(1 << 30);
    while (t < 256) : (t += 8) {
        const v = vm.evalAnim(0, t);
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        std.debug.print("  t={d:0>3}: {d:5} ({d}.{d:0>3}px)\n", .{ t, v, v >> 8, @as(u32, @intCast(@mod(v, 256))) });
    }
    std.debug.print("range: [{d}..{d}] raw -> amplitude {d} raw = {d}px (expect 0..1024)\n", .{ vmin, vmax, vmax - vmin, (vmax - vmin) >> 8 });
}
