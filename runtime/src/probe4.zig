// probe4: dump per-frame enemy state (behavior/step/age/vy) to trace exactly
// what enemy_sine produces at runtime. Debug for the v5 chaos metrics.
const std = @import("std");
const Io = std.Io;
const c = @import("core.zig");
const engine = @import("engine.zig");
const ent = @import("entity.zig");
const audio = @import("audio.zig");

fn autopilot(frame: u32) u32 {
    var m: u32 = 0;
    const phase = frame % 600;
    if (phase < 240) m |= 1 << 2;
    if ((frame / 45) % 2 == 0) m |= 1 << 0;
    if ((frame / 67) % 3 == 0) m |= 1 << 1;
    if (phase > 60 and (frame % 9) < 5) m |= 1 << 3;
    return m;
}

fn entTf(t: u32, ordinal: u32) u32 {
    var tf: u32 = 0;
    var n: u32 = 0;
    while (tf < c.n_tfs) : (tf += 1) {
        if (c.tfEnt(tf) == t) {
            if (n == ordinal) return tf;
            n += 1;
        }
    }
    return 0xFFFFFFFF;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const in_path = if (args.len > 1) args[1] else "game.libyte";
    var frames: u32 = 960;
    if (args.len > 2) frames = std.fmt.parseInt(u32, args[2], 10) catch 960;

    var dir = try std.Io.Dir.cwd().openDir(io, ".", .{});
    defer dir.close(io);
    var blob_buf: [256 * 1024]u8 = undefined;
    const blob = dir.readFile(io, in_path, &blob_buf) catch return;
    _ = engine.init(blob);

    const tfVy = entTf(2, 3);
    const tfVx = entTf(2, 2);
    const tfBeh = entTf(2, 5);
    const tfStep = entTf(2, 7);
    const tfAge = entTf(2, 9);

    var pcm: [735]i16 = undefined;
    var frame: u32 = 0;
    while (frame < frames) : (frame += 1) {
        c.wr32(c.ADDR_INPUT, autopilot(frame));
        engine.frame();
        _ = audio.pull(&pcm);

        if (frame >= 820) {
            var idx = ent.nextLive(2, 0);
            if (idx != 0xFFFFFFFF) std.debug.print("f{d}: ", .{frame});
            while (idx != 0xFFFFFFFF) {
                const h = idx | (@as(u32, ent.genOf(2, idx)) << 20);
                const beh = ent.fieldLoad(tfBeh, h);
                const step = ent.fieldLoad(tfStep, h);
                const age = ent.fieldLoad(tfAge, h);
                var vy = ent.fieldLoad(tfVy, h);
                // 17-bit wrap domain -> signed
                if (vy > 131071 / 2) vy -= 131072;
                var vx = ent.fieldLoad(tfVx, h);
                if (vx > 131071 / 2) vx -= 131072;
                std.debug.print("  [#{d} beh={d} age={d} step={d} vx={d} vy={d}]", .{ idx, beh, age, step, vx, vy });
                idx = ent.nextLive(2, idx + 1);
            }
            if (ent.count(2) > 0) std.debug.print("\n", .{});
        }
    }
}
