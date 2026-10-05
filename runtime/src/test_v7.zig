//! v7 engine capability tests: CORDIC atan2, isqrt/dist, camera clamping,
//! SRAM save/saved semantics, and v1 starfield compatibility at camera 0.
//! Run: zig test src/test_v7.zig (from runtime/).

const std = @import("std");
const c = @import("core.zig");
const render = @import("render.zig");

test "atan2: screen-convention compass points" {
    // Q24.8 unit vectors: rot 0 = up, clockwise (vx = sin, vy = -cos).
    // 4096 ang units = 360 degrees; CORDIC quantization is <= 2 units (0.17 deg).
    const px: i32 = 256; // 1.0
    const near = struct {
        fn eq(a: i32, b: i32) !void {
            const diff = @min(@abs(a - b), 4096 - @abs(a - b));
            try std.testing.expect(diff <= 2);
        }
    };
    try near.eq(c.atan2ang(-px, 0), 0); // up
    try near.eq(c.atan2ang(0, px), 1024); // right
    try near.eq(c.atan2ang(px, 0), 2048); // down
    try near.eq(c.atan2ang(0, -px), 3072); // left
    try near.eq(c.atan2ang(-px, px), 512); // up-right diagonal = 45 deg
    try near.eq(c.atan2ang(px, px), 1536); // down-right diagonal = 135 deg
    try near.eq(c.atan2ang(px, -px), 2560); // down-left = 225 deg
    try near.eq(c.atan2ang(-px, -px), 3584); // up-left = 315 deg
}

test "atan2: roundtrips with sin/cos across the full circle" {
    var a: i32 = 0;
    while (a < 4096) : (a += 37) {
        const s = c.sinA(a);
        const co = c.cosA(a);
        const back = c.atan2ang(-co, s); // dy = -cos, dx = sin
        // tolerance 6 ang units (0.53 deg): the sine LUT quantizes to Q24.8
        // (about 1.4 units of direction noise) on top of CORDIC's ~2
        const diff = @min(@abs(back - a), 4096 - @abs(back - a));
        try std.testing.expect(diff <= 6);
    }
}

test "atan2: magnitude invariance (CORDIC is scale-free)" {
    // small and large inputs of the same direction give the same angle
    const a1 = c.atan2ang(300, 700);
    const a2 = c.atan2ang(300 * 4096, 700 * 4096);
    const diff = @min(@abs(a1 - a2), 4096 - @abs(a1 - a2));
    try std.testing.expect(diff <= 2);
    // zero vector is defined as 0
    try std.testing.expectEqual(@as(i32, 0), c.atan2ang(0, 0));
}

test "isqrt: exact squares and floors" {
    try std.testing.expectEqual(@as(u32, 0), c.isqrt64(0));
    try std.testing.expectEqual(@as(u32, 1), c.isqrt64(1));
    try std.testing.expectEqual(@as(u32, 3), c.isqrt64(9));
    try std.testing.expectEqual(@as(u32, 3), c.isqrt64(15)); // floor(3.87)
    try std.testing.expectEqual(@as(u32, 4), c.isqrt64(16));
    try std.testing.expectEqual(@as(u32, 65535), c.isqrt64(65535 * 65535));
    // brute force 1..4096: floor(sqrt(v^2 - 1)) = v - 1
    var v: u64 = 1;
    while (v <= 4096) : (v += 1) {
        const r = c.isqrt64(v * v - 1);
        try std.testing.expectEqual(@as(u64, v - 1), r);
    }
}

test "dist: 3-4-5 triangle in Q24.8" {
    // (0,0) -> (3,4): dist = 5 px = 1280 raw
    const d = c.distF(0, 0, 3 * 256, 4 * 256);
    try std.testing.expectEqual(@as(i32, 5 * 256), d);
    // large-world coordinates must not overflow (8192 px apart)
    const big = c.distF(0, 0, 8192 * 256, 0);
    try std.testing.expectEqual(@as(i32, 8192 * 256), big);
}

test "camera clamp: world == screen pins to top-left 0 (v1 behavior)" {
    c.game_w = 512;
    c.game_h = 512;
    c.game_world_w = 512;
    c.game_world_h = 512;
    // any request clamps to top-left (0,0) -> render offset 0
    const cl = c.clampCam(256 * 256, 256 * 256);
    try std.testing.expectEqual(@as(i32, 0), cl.x);
    try std.testing.expectEqual(@as(i32, 0), cl.y);
    const cl2 = c.clampCam(100000, -5000);
    try std.testing.expectEqual(@as(i32, 0), cl2.x);
    try std.testing.expectEqual(@as(i32, 0), cl2.y);
}

test "camera clamp: scrolling world keeps viewport inside bounds" {
    c.game_w = 512;
    c.game_h = 512;
    c.game_world_w = 4096;
    c.game_world_h = 2048;
    // far-left center request: top-left clamps to 0
    const cl = c.clampCam(-100000, -100000);
    try std.testing.expectEqual(@as(i32, 0), cl.x);
    try std.testing.expectEqual(@as(i32, 0), cl.y);
    // far-right: top-left clamps to world - viewport
    const cr = c.clampCam(5000000, 5000000);
    try std.testing.expectEqual(@as(i32, (4096 - 512) * 256), cr.x);
    try std.testing.expectEqual(@as(i32, (2048 - 512) * 256), cr.y);
    // interior request: center 1000,700 -> top-left 744,444
    const cm = c.clampCam(1000 * 256, 700 * 256);
    try std.testing.expectEqual(@as(i32, 744 * 256), cm.x);
    try std.testing.expectEqual(@as(i32, 444 * 256), cm.y);
}

test "sram: save region survives the init zeroing pass" {
    // SRAM starts zeroed
    try std.testing.expectEqual(@as(i32, 0), @as(i32, @bitCast(c.rd32(c.ADDR_SRAM))));
    c.wr32(c.ADDR_SRAM + 3 * 4, @bitCast(@as(i32, 12345)));
    c.wr32(c.ADDR_SRAM + 60 * 4, @bitCast(@as(i32, -99)));
    // engine re-init must NOT wipe it (battery-backed semantics)
    @memset(c.mem[0..c.ADDR_SRAM], 0);
    @memset(c.mem[c.ADDR_SRAM + c.SRAM_BYTES ..], 0);
    try std.testing.expectEqual(@as(i32, 12345), @as(i32, @bitCast(c.rd32(c.ADDR_SRAM + 12))));
    try std.testing.expectEqual(@as(i32, -99), @as(i32, @bitCast(c.rd32(c.ADDR_SRAM + 240))));
}

test "starfield: camera 0 is bit-identical to the v1 field" {
    // deterministic: same frame -> same star vertices, with camera
    // registers explicitly zeroed (the v1 condition)
    c.game_w = 512;
    c.game_h = 512;
    c.game_world_w = 512;
    c.game_world_h = 512;
    c.wr32(c.ADDR_CAM_X, 0);
    c.wr32(c.ADDR_CAM_Y, 0);
    c.wr32(c.ADDR_FRAME, 777);
    render.streamReset();
    render.emitStarfield();
    const n = render.streamCount();
    const a = try std.heap.page_allocator.dupe(u8, c.mem[c.ADDR_STREAM + 4 .. c.ADDR_STREAM + 4 + n * 14]);

    // camera() on a world == screen game: clamp pins registers to 0 ->
    // identical starfield (and identical world-layer draws)
    const cl = c.clampCam(42 * 256, 17 * 256);
    c.wr32(c.ADDR_CAM_X, @bitCast(cl.x));
    c.wr32(c.ADDR_CAM_Y, @bitCast(cl.y));
    render.streamReset();
    render.emitStarfield();
    const b = c.mem[c.ADDR_STREAM + 4 .. c.ADDR_STREAM + 4 + n * 14];
    try std.testing.expectEqualSlices(u8, a, b);

    // a REAL camera move must move the stars (parallax proof)
    c.game_world_w = 4096;
    const cl2 = c.clampCam(2000 * 256, 600 * 256);
    c.wr32(c.ADDR_CAM_X, @bitCast(cl2.x));
    c.wr32(c.ADDR_CAM_Y, @bitCast(cl2.y));
    render.streamReset();
    render.emitStarfield();
    const d2 = c.mem[c.ADDR_STREAM + 4 .. c.ADDR_STREAM + 4 + n * 14];
    var moved: u32 = 0;
    for (a, d2) |x, y| {
        if (x != y) moved += 1;
    }
    try std.testing.expect(moved > 100); // the field scrolled, not jittered
    std.heap.page_allocator.free(a);
}
