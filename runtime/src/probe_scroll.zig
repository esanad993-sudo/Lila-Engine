//! Scroller state probe: runs N frames and dumps player/camera/globals/SRAM.
//! Run: zig run src/probe_scroll.zig -- ../build/scroller.libyte [frames] [mask]
const std = @import("std");
const c = @import("core.zig");
const engine = @import("engine.zig");
const ent = @import("entity.zig");
const loader = @import("loader.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const path = if (args.len > 1) args[1] else "../build/scroller.libyte";
    const frames: u32 = if (args.len > 2) (std.fmt.parseInt(u32, args[2], 10) catch 10) else 10;
    const mask: u32 = if (args.len > 3) (std.fmt.parseInt(u32, args[3], 0) catch 0) else 0;

    var dir = try std.Io.Dir.cwd().openDir(io, ".", .{});
    defer dir.close(io);
    var buf: [256 * 1024]u8 = undefined;
    const blob = dir.readFile(io, path, &buf) catch return;

    const rc = engine.init(blob);
    if (rc != 0) { std.debug.print("init rc={d}\n", .{rc}); return; }
    engine.rememberSchema();

    // find global tfs (first globals-block fields, in order)
    var gtf: [16]u32 = undefined;
    var ng: u32 = 0;
    var t: u32 = 0;
    while (t < c.n_tfs and ng < 16) : (t += 1) {
        if (c.tfEnt(t) == 0xFF) { gtf[ng] = t; ng += 1; }
    }
    // Player fields: x=tf0-ish — find Player (entity 0) x and y tfs
    var tfPx: u32 = 0xFFFFFFFF;
    var tfPy: u32 = 0xFFFFFFFF;
    t = 0;
    while (t < c.n_tfs) : (t += 1) {
        if (c.tfEnt(t) == 0) {
            // count field ordinal via off_bits? just take the first two of entity 0
            if (tfPx == 0xFFFFFFFF) { tfPx = t; continue; }
            if (tfPy == 0xFFFFFFFF) { tfPy = t; break; }
        }
    }

    var f: u32 = 0;
    while (f < frames) : (f += 1) {
        c.wr32(c.ADDR_INPUT, mask);
        engine.frame();
        if (f % 30 == 0 or f == frames - 1) {
            var px: i32 = 0;
            var py: i32 = 0;
            const idx = ent.nextLive(0, 0);
            if (idx != 0xFFFFFFFF) {
                const h = idx | (@as(u32, ent.genOf(0, idx)) << 20);
                px = ent.fieldLoad(tfPx, h);
                py = ent.fieldLoad(tfPy, h);
            }
            std.debug.print("f{d:0>4}: player=({d},{d})px cam=({d},{d})px dist={d} best={d} hull={d} over={} next_x={d} sram0={d} ents: P{d} B{d} T{d} C{d} S{d}\n", .{
                f + 1, px >> 8, py >> 8,
                @as(i32, @bitCast(c.rd32(c.ADDR_CAM_X))) >> 8, @as(i32, @bitCast(c.rd32(c.ADDR_CAM_Y))) >> 8,
                ent.globalLoad(gtf[0]), ent.globalLoad(gtf[1]), ent.globalLoad(gtf[3]),
                ent.globalLoad(gtf[4]) != 0,
                ent.globalLoad(gtf[6]) >> 8,
                @as(i32, @bitCast(c.rd32(c.ADDR_SRAM))),
                ent.count(0), ent.count(1), ent.count(2), ent.count(3), ent.count(4),
            });
        }
    }
    std.debug.print("sram_dirty={} world={d}x{d} screen={d}x{d}\n", .{
        c.sram_dirty, c.game_world_w, c.game_world_h, c.game_w, c.game_h,
    });
}
