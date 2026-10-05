//! v11 engine capability tests: arrays (wrap bounds, entity-array spawn
//! zeroing), fixed-point 3D projection goldens, negative-angle trig wrap,
//! and draw3d determinism + budget clamps.
//! Run: zig test src/test_v11.zig (from runtime/).

const std = @import("std");
const c = @import("core.zig");
const ent = @import("entity.zig");
const render = @import("render.zig");
const render3d = @import("render3d.zig");

// ---- minimal loader stand-in: install an arrays schema by hand ----
fn installArrays() void {
    // arr 0: global fixed[32] (pow2 cap -> mask path; fits the 18-cell quad)
    // arr 1: global u8[5]  (non-pow2 cap -> modulo path)
    // arr 2: entity array for type 0, cap 4 (entity plane)
    c.n_arrs = 3;
    c.arr_ent = .{ 0xFF, 0xFF, 0 } ++ [_]u8{0xFF} ** (c.MAX_ARRS - 3);
    c.arr_cap = .{ 32, 5, 4 } ++ [_]u16{0} ** (c.MAX_ARRS - 3);
    c.arr_base = .{ c.ADDR_GARR, c.ADDR_GARR + 32 * 4, c.ADDR_EARR } ++
        [_]u32{0} ** (c.MAX_ARRS - 3);
    @memset(c.mem[c.ADDR_GARR..(c.ADDR_EARR + c.EARR_BYTES)], 0);
    // one entity type with maxLive 8 so earr addressing has room
    c.n_entities = 1;
    c.wr16(c.ENT_SCHEMA_BASE + 0 * 6 + 4, 8); // maxLive(0) = 8
}

test "arrays: pow2 caps mask and non-pow2 caps modulo-wrap" {
    installArrays();
    // pow2 cap 32: raw index -1 wraps to 31 (two's complement mask)
    try std.testing.expectEqual(@as(u32, 31), c.arrIndex(0, -1));
    try std.testing.expectEqual(@as(u32, 9 % 32), c.arrIndex(0, 9));
    try std.testing.expectEqual(@as(u32, 0), c.arrIndex(0, 0));
    // non-pow2 cap 5: RAW u32 modulo semantics (the documented contract):
    // -1 wraps to 0xFFFFFFFF, and 0xFFFFFFFF % 5 == 0
    try std.testing.expectEqual(@as(u32, 0), c.arrIndex(1, -1));
    try std.testing.expectEqual(@as(u32, 4), c.arrIndex(1, -2));
    try std.testing.expectEqual(@as(u32, 2), c.arrIndex(1, 7));
    try std.testing.expectEqual(@as(u32, 3), c.arrIndex(1, 3));
}

test "arrays: global cells store and load raw i32" {
    installArrays();
    const base = c.garrElem(0, 3);
    c.wr32(base, @bitCast(@as(i32, -123456)));
    const back: i32 = @bitCast(c.rd32(c.garrElem(0, 3)));
    try std.testing.expectEqual(@as(i32, -123456), back);
    // planes do not overlap: array 1 base is past array 0's 8 cells
    try std.testing.expect(c.arr_base[1] >= c.arr_base[0] + 8 * 4);
}

test "arrays: entity rows address by slot and zero on spawn" {
    installArrays();
    ent.resetAll();
    // slot 3 row of entity-array 2
    const row = c.earrRow(2, 3);
    c.wr32(row + 2 * 4, 0xBEEF);
    try std.testing.expectEqual(@as(u32, 0xBEEF), c.rd32(c.earrRow(2, 3) + 8));
    // rows of different slots are distinct planes
    try std.testing.expect(c.earrRow(2, 4) >= row + 4 * 4);
    // spawn zeroes the dense row AND the entity-array rows of that slot
    // (write junk into slot 0's array row first, then spawn into slot 0)
    c.wr32(c.earrRow(2, 0), 0xDEAD);
    c.wr32(c.earrRow(2, 0) + 4, 0xDEAD);
    const h = ent.spawn(0);
    try std.testing.expect(h != 0xFFFFFFFF);
    try std.testing.expectEqual(@as(u32, 0), c.rd32(c.earrRow(2, 0)));
    try std.testing.expectEqual(@as(u32, 0), c.rd32(c.earrRow(2, 0) + 4));
}

test "trig: negative angles wrap two's-complement (sin(-x) == sin(4096-x))" {
    // regression: intCast panicked on negatives; bitCast wraps exactly
    try std.testing.expectEqual(c.sinA(3072), c.sinA(-1024));
    try std.testing.expectEqual(c.cosA(3072), c.cosA(-1024));
    try std.testing.expectEqual(c.sinA(4095), c.sinA(-1));
    // non-negative domain is bit-identical to the v1..v10 behavior
    // (LUT quantization: |sin| peaks at 255, not 256 — pre-existing v1 trait)
    try std.testing.expectEqual(@as(i32, 0), c.sinA(0));
    try std.testing.expectEqual(@as(i32, 255), c.sinA(1024));
    try std.testing.expectEqual(@as(i32, -256), c.sinA(3072));
    try std.testing.expectEqual(@as(i32, 255), c.cosA(0));
    try std.testing.expectEqual(@as(i32, 0), c.cosA(1024));
}

test "proj3: golden screen positions through a known camera" {
    // camera at origin looking +Y (yaw 0), pitch 0, 512x512 => focal 256px
    c.game_w = 512;
    c.game_h = 512;
    c.wr32(c.ADDR_CAM3_X, 0);
    c.wr32(c.ADDR_CAM3_Y, 0);
    c.wr32(c.ADDR_CAM3_Z, 0);
    c.wr32(c.ADDR_CAM3_YAW, 0);
    c.wr32(c.ADDR_CAM3_PITCH, 0);
    // point straight ahead at 2 units: center screen, scale ~= focal/2
    // (LUT/rounding noise: cos(0) quantizes to 255/256, so fwd = 510 not 512)
    {
        const p = c.proj3(0, 2 * 256, 0);
        try std.testing.expect(p.ok);
        try std.testing.expectEqual(@as(i32, 256 << 8), p.sx);
        try std.testing.expectEqual(@as(i32, 256 << 8), p.sy);
        const diff = @abs(p.scale - (@as(i32, 256) << 8) / 2);
        try std.testing.expect(diff < 400);
    }
    // point 1 unit right at 2 forward: sx = 256 + right/fwd*focal = 256+128
    {
        const p = c.proj3(256, 2 * 256, 0);
        try std.testing.expectEqual(@as(i32, 384), p.sx >> 8);
    }
    // point 1 unit up at 2 forward: sy = 256 - up/fwd*focal ~= 256-128
    // (cpit quantizes to 255/256 -> 127 instead of 128; tolerance 2px)
    {
        const p = c.proj3(0, 2 * 256, 256);
        const diff = @abs((p.sy >> 8) - 128);
        try std.testing.expect(diff <= 2);
    }
    // behind the camera: !ok and everything clamps to 0
    {
        const p = c.proj3(0, -256, 0);
        try std.testing.expect(!p.ok);
        try std.testing.expectEqual(@as(i32, 0), p.scale);
    }
}

test "proj3: yaw 90 looks along +X (yaw convention consistent)" {
    c.game_w = 512;
    c.game_h = 512;
    c.wr32(c.ADDR_CAM3_X, 0);
    c.wr32(c.ADDR_CAM3_Y, 0);
    c.wr32(c.ADDR_CAM3_Z, 0);
    c.wr32(c.ADDR_CAM3_YAW, 1024); // 90 degrees
    c.wr32(c.ADDR_CAM3_PITCH, 0);
    // 2 units along +X is now dead ahead: center screen
    const p = c.proj3(2 * 256, 0, 0);
    try std.testing.expect(p.ok);
    try std.testing.expectEqual(@as(i32, 256), p.sx >> 8);
    try std.testing.expectEqual(@as(i32, 256), p.sy >> 8);
    // 1 unit along the OLD forward (+Y) is now edge-on (fwd ~ 0): !ok
    const q = c.proj3(0, 256, 0);
    try std.testing.expect(!q.ok);
}

test "draw3d: budget clamps, determinism, and wireframe mode" {
    installArrays();
    ent.resetAll();
    c.game_w = 512;
    c.game_h = 512;
    c.wr32(c.ADDR_CAM3_X, 0);
    c.wr32(c.ADDR_CAM3_Y, 0);
    c.wr32(c.ADDR_CAM3_Z, @bitCast(@as(i32, -6 * 256)));
    c.wr32(c.ADDR_CAM3_YAW, 0);
    c.wr32(c.ADDR_CAM3_PITCH, 0);
    // build a 2-tri quad facing the camera in global array 0 (cap 8)
    const q = [_]i32{
        -256, 256, -256, // v0 (-1, 1, -1)
        256,  256, -256, // v1 ( 1, 1, -1)
        256,  256, 256,  // v2 ( 1, 1,  1)
        -256, 256, -256,
        256,  256, 256,
        -256, 256, 256,
    };
    for (q, 0..) |v, i| c.wr32(c.arr_base[0] + @as(u32, @intCast(i)) * 4, @bitCast(v));

    // solid: 6 verts on the stream
    render.streamReset();
    render3d.draw3d(0, 6, 0, 0, 0, 0, 0x8080FFFF);
    try std.testing.expectEqual(@as(u32, 6), render.streamCount());
    // determinism: the same call emits the identical stream
    render.streamReset();
    render3d.draw3d(0, 6, 0, 0, 0, 0, 0x8080FFFF);
    try std.testing.expectEqual(@as(u32, 6), render.streamCount());
    // n clamps to whole tris (7 -> 6) and to the array capacity
    render.streamReset();
    render3d.draw3d(0, 7, 0, 0, 0, 0, 0x8080FFFF);
    try std.testing.expectEqual(@as(u32, 6), render.streamCount());
    // wireframe (alpha 0): 3 edges x 6 verts per tri = 36
    render.streamReset();
    render3d.draw3d(0, 6, 0, 0, 0, 0, 0x40E0FF00);
    try std.testing.expectEqual(@as(u32, 36), render.streamCount());
    // model yaw rotates deterministically: same pose, same stream
    render.streamReset();
    render3d.draw3d(0, 6, 0, 0, 0, 777, 0x8080FFFF);
    const a = render.streamCount();
    render.streamReset();
    render3d.draw3d(0, 6, 0, 0, 0, 777, 0x8080FFFF);
    try std.testing.expectEqual(a, render.streamCount());
}
