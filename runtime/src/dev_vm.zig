//! Dev harness: load a .libyte, run 300 scripted frames, dump VM state.
//! (a manual driver — the automated suites live in test_*.zig)
const std = @import("std");
const c = @import("core.zig");
const engine = @import("engine.zig");
const ent = @import("entity.zig");
const audio = @import("audio.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) {
        std.debug.print("usage: test_vm <game.libyte>\n", .{});
        return;
    }
    var dir = try std.Io.Dir.cwd().openDir(io, ".", .{});
    defer dir.close(io);
    var buf: [256 * 1024]u8 = undefined;
    const blob = try dir.readFile(io, args[1], &buf);

    const rc = engine.init(blob);
    if (rc != 0) {
        std.debug.print("init failed: {d}\n", .{rc});
        return;
    }

    var frame: u32 = 0;
    while (frame < 300) : (frame += 1) {
        var mask: u32 = 0;
        if (frame >= 10 and frame < 60) mask |= 1 << 3; // fire
        if (frame >= 30 and frame < 90) mask |= 1 << 0; // left
        if (frame >= 20 and frame < 100) mask |= 1 << 2; // up
        c.wr32(c.ADDR_INPUT, mask);
        engine.frame();
        if (frame % 60 == 0) {
            std.debug.print("frame {d}: player={d} bullet={d} enemy={d} powerup={d} particle={d} stream={d}\n", .{
                frame, ent.count(0), ent.count(1), ent.count(2), ent.count(3), ent.count(4),
                c.rd32(c.ADDR_STREAM),
            });
        }
    }

    // dump globals (tf ent==0xFF)
    std.debug.print("globals:\n", .{});
    var tf: u32 = 0;
    while (tf < c.n_tfs) : (tf += 1) {
        if (c.tfEnt(tf) == 0xFF) {
            std.debug.print("  tf[{d}] bits={d} signed={} val={d}\n", .{
                tf, c.tfBits(tf) & 0x7F, (c.tfBits(tf) & 0x80) != 0, ent.globalLoad(tf),
            });
        }
    }

    // audio sanity: pull 1 second
    var pcm: [44100]i16 = undefined;
    _ = audio.pull(&pcm);
    var peak: i32 = 0;
    var nonzero: u32 = 0;
    for (pcm) |s| {
        const v: i32 = s;
        if (v > peak) peak = v;
        if (v != 0) nonzero += 1;
    }
    std.debug.print("audio 1s: peak={d} nonzero={d}/{d}\n", .{ peak, nonzero, pcm.len });
    std.debug.print("VM TEST DONE\n", .{});
}
