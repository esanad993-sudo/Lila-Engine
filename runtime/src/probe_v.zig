const std = @import("std");
const c = @import("core.zig");
const engine = @import("engine.zig");
const ent = @import("entity.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const path = if (args.len > 1) args[1] else "../build/scroller.libyte";
    var dir = try std.Io.Dir.cwd().openDir(io, ".", .{});
    defer dir.close(io);
    var buf: [256 * 1024]u8 = undefined;
    const blob = dir.readFile(io, path, &buf) catch return;
    if (engine.init(blob) != 0) return;
    engine.rememberSchema();
    // Player x(tf0), vx(tf2), y(tf1), vy(tf3) — first 4 tfs of entity 0
    var f: u32 = 0;
    while (f < 24) : (f += 1) {
        c.wr32(c.ADDR_INPUT, 2); // right
        engine.frame();
        const idx = ent.nextLive(0, 0);
        if (idx != 0xFFFFFFFF) {
            const h = idx | (@as(u32, ent.genOf(0, idx)) << 20);
            std.debug.print("f{d:0>3}: x={d} y={d} vx={d} vy={d} (raw Q24.8)\n", .{
                f + 1, ent.fieldLoad(0, h), ent.fieldLoad(1, h),
                ent.fieldLoad(2, h), ent.fieldLoad(3, h),
            });
        }
    }
}
