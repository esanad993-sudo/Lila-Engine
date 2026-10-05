//! v12 engine capability tests: mat4/quat builtins over arrays (axis-angle
//! goldens, composition identities, convention checks), the indexed mesh
//! pass (draw3di == draw3d on equal geometry, out-of-range index drops),
//! and the z-buffer profile knob (token encoding + raster determinism).
//! Run: zig test src/test_v12.zig (from runtime/).

const std = @import("std");
const c = @import("core.zig");
const ent = @import("entity.zig");
const render = @import("render.zig");
const raster = @import("raster.zig");
const render3d = @import("render3d.zig");

// ---- minimal loader stand-in: install an arrays schema by hand ----
fn installArrays() void {
    // arr 0: global fixed[32] verts   (8 verts x 3 cells)
    // arr 1: global u8[16]    indices (4 tris x 3)
    // arr 2: global fixed[16] mat A
    // arr 3: global fixed[16] mat B
    // arr 4: global fixed[4]  quat
    // arr 5: global fixed[32] bind pose
    c.n_arrs = 6;
    c.arr_ent = .{ 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF } ++ [_]u8{0xFF} ** (c.MAX_ARRS - 6);
    c.arr_cap = .{ 32, 16, 16, 16, 4, 32 } ++ [_]u16{0} ** (c.MAX_ARRS - 6);
    var base: u32 = c.ADDR_GARR;
    for (0..6) |i| {
        c.arr_base[i] = base;
        base += @as(u32, c.arr_cap[i]) * 4;
    }
    @memset(c.mem[c.ADDR_GARR..(c.ADDR_EARR + c.EARR_BYTES)], 0);
    c.n_entities = 0;
}

fn rdCell(arr: u32, idx: u32) i32 {
    return @bitCast(c.rd32(c.arr_base[arr] + idx * 4));
}

fn wrCell(arr: u32, idx: u32, v: i32) void {
    c.wr32(c.arr_base[arr] + idx * 4, @bitCast(v));
}

test "quatAA: axis-angle goldens and normalization" {
    installArrays();
    // 90 degrees about +Z: (0, 0, sin45, cos45) in Q24.8 (~181, ~181)
    render3d.quatAA(0, 0, 0, 0, 256, 1024);
    try std.testing.expectEqual(@as(i32, 0), rdCell(0, 0));
    try std.testing.expectEqual(@as(i32, 0), rdCell(0, 1));
    try std.testing.expect(@abs(rdCell(0, 2) - 181) <= 2);
    try std.testing.expect(@abs(rdCell(0, 3) - 181) <= 2);
    // unit magnitude: x^2+y^2+z^2+w^2 ~= 256^2 in Q16 (mulF twice)
    const x = rdCell(0, 0);
    const y = rdCell(0, 1);
    const z = rdCell(0, 2);
    const w = rdCell(0, 3);
    const m2 = c.mulF(x, x) +% c.mulF(y, y) +% c.mulF(z, z) +% c.mulF(w, w);
    try std.testing.expect(@abs(m2 - 256) <= 4);
    // 90 degrees about +X: (sin45, 0, 0, cos45)
    render3d.quatAA(0, 0, 256, 0, 0, 1024);
    try std.testing.expect(@abs(rdCell(0, 0) - 181) <= 2);
    try std.testing.expectEqual(@as(i32, 0), rdCell(0, 1));
    // zero axis -> identity quat (exact)
    render3d.quatAA(0, 0, 0, 0, 0, 1024);
    try std.testing.expectEqual(@as(i32, 0), rdCell(0, 0));
    try std.testing.expectEqual(@as(i32, 256), rdCell(0, 3));
    // negative angle: q(-90 deg about +Z) rotates +Y to -X. A quat and its
    // negation are the SAME rotation, so verify via the rotated point.
    render3d.quatAA(4, 0, 0, 0, 256, -1024);
    render3d.m4QT(2, 0, 4, 0, 0, 0, 0);
    render3d.skin3(0, 0, 2, 0, 0, 256, 0); // p = (0, 1, 0)
    try std.testing.expect(@abs(rdCell(0, 0) - 256) <= 4); // x' ~= +1
    try std.testing.expect(@abs(rdCell(0, 1)) <= 4); // y' ~= 0
}

test "m4QT: identity quat gives pure translation (column-major layout)" {
    installArrays();
    render3d.quatAA(4, 0, 0, 0, 0, 0); // identity
    render3d.m4QT(2, 0, 4, 0, 1 * 256, -2 * 256, 3 * 256);
    // rotation block = identity exactly (zero quat components)
    try std.testing.expectEqual(@as(i32, 256), rdCell(2, 0));
    try std.testing.expectEqual(@as(i32, 256), rdCell(2, 5));
    try std.testing.expectEqual(@as(i32, 256), rdCell(2, 10));
    try std.testing.expectEqual(@as(i32, 256), rdCell(2, 15));
    try std.testing.expectEqual(@as(i32, 0), rdCell(2, 1));
    try std.testing.expectEqual(@as(i32, 0), rdCell(2, 4));
    // translation in cells 12..14, bottom-right 1
    try std.testing.expectEqual(@as(i32, 256), rdCell(2, 12));
    try std.testing.expectEqual(@as(i32, -512), rdCell(2, 13));
    try std.testing.expectEqual(@as(i32, 768), rdCell(2, 14));
}

test "quat rotZ(90): skin maps +X to +Y (right-handed convention)" {
    installArrays();
    render3d.quatAA(4, 0, 0, 0, 256, 1024); // 90 deg about +Z
    render3d.m4QT(2, 0, 4, 0, 0, 0, 0);
    render3d.skin3(0, 0, 2, 0, 256, 0, 0); // p = (1, 0, 0)
    try std.testing.expect(@abs(rdCell(0, 0)) <= 2); // x' ~= 0
    try std.testing.expect(@abs(rdCell(0, 1) - 256) <= 3); // y' ~= 1
    try std.testing.expect(@abs(rdCell(0, 2)) <= 2); // z' ~= 0
}

test "skin3 with translation: p + t exact" {
    installArrays();
    render3d.m4QT(2, 0, 4, 0, 10 * 256, 0, -3 * 256); // identity quat + t
    render3d.skin3(0, 3, 2, 0, 256, 2 * 256, 4 * 256);
    try std.testing.expectEqual(@as(i32, 11 * 256), rdCell(0, 3));
    try std.testing.expectEqual(@as(i32, 2 * 256), rdCell(0, 4));
    try std.testing.expectEqual(@as(i32, 1 * 256), rdCell(0, 5));
}

test "m4Mul: two translations compose to the summed translation (alias-safe)" {
    installArrays();
    render3d.m4QT(2, 0, 4, 0, 1 * 256, 2 * 256, 3 * 256); // A = T(1,2,3)
    render3d.m4QT(3, 0, 4, 0, 10 * 256, 20 * 256, 30 * 256); // B = T(10,20,30)
    // alias: dst = A (in place) must still read B fully
    render3d.m4Mul(2, 0, 2, 0, 3, 0);
    try std.testing.expectEqual(@as(i32, 11 * 256), rdCell(2, 12));
    try std.testing.expectEqual(@as(i32, 22 * 256), rdCell(2, 13));
    try std.testing.expectEqual(@as(i32, 33 * 256), rdCell(2, 14));
    try std.testing.expectEqual(@as(i32, 256), rdCell(2, 0));
    try std.testing.expectEqual(@as(i32, 256), rdCell(2, 5));
}

test "qMul: two 90-degree Z rotations compose to 180 (with golden)" {
    installArrays();
    render3d.quatAA(4, 0, 0, 0, 256, 1024); // 90 deg about +Z
    render3d.qMul(2, 0, 4, 0, 4, 0); // q*q = 180 deg about +Z
    // 180 deg: (0, 0, sin90 = 1, cos90 = 0) -> (0, 0, 256, ~0)
    try std.testing.expect(@abs(rdCell(2, 2) - 256) <= 3);
    try std.testing.expect(@abs(rdCell(2, 3)) <= 3);
    // rotate (1,0,0) by it: should map to (-1, 0, 0)
    render3d.m4QT(3, 0, 2, 0, 0, 0, 0);
    render3d.skin3(0, 0, 3, 0, 256, 0, 0);
    // (1,0,0) rotated 180 deg about Z -> (-1,0,0); two chained LUT squarings
    // quantize sin45=181/256, so allow ~4% drift
    try std.testing.expect(@abs(rdCell(0, 0) + 256) <= 10);
    try std.testing.expect(@abs(rdCell(0, 1)) <= 10);
}

test "draw3di: indexed quad matches soup, bad index drops, wireframe rides" {
    installArrays();
    ent.resetAll();
    c.game_w = 512;
    c.game_h = 512;
    c.wr32(c.ADDR_CAM3_X, 0);
    c.wr32(c.ADDR_CAM3_Y, 0);
    c.wr32(c.ADDR_CAM3_Z, @bitCast(@as(i32, -6 * 256)));
    c.wr32(c.ADDR_CAM3_YAW, 0);
    c.wr32(c.ADDR_CAM3_PITCH, 0);
    // 4 unique verts of a quad at y=+1 (facing the camera at -Y), 2 tris
    const q = [_]i32{
        -256, 256, -256, // v0
        256,  256, -256, // v1
        256,  256, 256,  // v2
        -256, 256, 256,  // v3
    };
    for (q, 0..) |v, i| wrCell(0, @intCast(i), v);
    // same winding the soup version uses (kept tris have area < 0)
    const ix = [_]i32{ 0, 1, 2, 0, 2, 3 };
    for (ix, 0..) |v, i| wrCell(1, @intCast(i), v);

    // soup reference: 6 verts
    const soup = [_]i32{ -256, 256, -256, 256, 256, -256, 256, 256, 256, -256, 256, -256, 256, 256, 256, -256, 256, 256 };
    for (soup, 0..) |v, i| wrCell(5, @intCast(i), v);
    render.streamReset();
    render3d.draw3d(5, 6, 0, 0, 0, 0, 0x8080FFFF);
    const soup_count = render.streamCount();
    // indexed: same geometry, 6 verts on the stream
    render.streamReset();
    render3d.draw3di(0, 1, 4, 6, 0, 0, 0, 0, 0x8080FFFF);
    try std.testing.expectEqual(@as(u32, 6), render.streamCount());
    try std.testing.expectEqual(soup_count, render.streamCount());
    // determinism
    render.streamReset();
    render3d.draw3di(0, 1, 4, 6, 0, 0, 0, 0, 0x8080FFFF);
    try std.testing.expectEqual(@as(u32, 6), render.streamCount());
    // out-of-range index drops exactly that tri (no trap)
    wrCell(1, 2, 99); // tri 0 = (0, 1, 99) -> dropped; tri 1 survives
    render.streamReset();
    render3d.draw3di(0, 1, 4, 6, 0, 0, 0, 0, 0x8080FFFF);
    try std.testing.expectEqual(@as(u32, 3), render.streamCount());
    // BOTH tris bad -> empty stream
    wrCell(1, 5, 99); // tri 1 = (0, 2, 99) -> dropped
    render.streamReset();
    render3d.draw3di(0, 1, 4, 6, 0, 0, 0, 0, 0x8080FFFF);
    try std.testing.expectEqual(@as(u32, 0), render.streamCount());
    wrCell(1, 2, 2);
    wrCell(1, 5, 3);
    // wireframe (alpha 0): 2 tris x 3 edges x 6 verts = 36
    render.streamReset();
    render3d.draw3di(0, 1, 4, 6, 0, 0, 0, 0, 0x40E0FF00);
    try std.testing.expectEqual(@as(u32, 36), render.streamCount());
    // nv/ni clamp to capacity silently: ni 7 -> floored to 6 tris-worth
    render.streamReset();
    render3d.draw3di(0, 1, 4, 7, 0, 0, 0, 0, 0x8080FFFF);
    try std.testing.expectEqual(@as(u32, 6), render.streamCount());
}

test "z-buffer knob: token encoding flips, raster stays deterministic" {
    installArrays();
    ent.resetAll();
    c.game_w = 512;
    c.game_h = 512;
    c.wr32(c.ADDR_CAM3_X, 0);
    c.wr32(c.ADDR_CAM3_Y, 0);
    c.wr32(c.ADDR_CAM3_Z, @bitCast(@as(i32, -6 * 256)));
    c.wr32(c.ADDR_CAM3_YAW, 0);
    c.wr32(c.ADDR_CAM3_PITCH, 0);
    const q = [_]i32{
        -256, 256, -256,
        256,  256, -256,
        256,  256, 256,
        -256, 256, 256,
    };
    for (q, 0..) |v, i| wrCell(0, @intCast(i), v);
    const ix = [_]i32{ 0, 1, 2, 0, 2, 3 };
    for (ix, 0..) |v, i| wrCell(1, @intCast(i), v);

    // knob OFF (default): token 0 — v11 byte-identical stream
    c.zbuf3d = false;
    render.streamReset();
    render3d.draw3di(0, 1, 4, 6, 0, 0, 0, 0, 0x8080FFFF);
    const tok_off = c.rd16(c.ADDR_STREAM + 4 + 12);
    try std.testing.expectEqual(@as(u16, 0), tok_off);

    // knob ON: token carries bit3 + 12-bit depth (>= 1)
    c.zbuf3d = true;
    render.streamReset();
    render3d.draw3di(0, 1, 4, 6, 0, 0, 0, 0, 0x8080FFFF);
    const tok_on = c.rd16(c.ADDR_STREAM + 4 + 12);
    try std.testing.expect((tok_on & 8) != 0);
    try std.testing.expect(((tok_on >> 4) & 0xFFF) >= 1);
    // 2D content keeps tok bit3 clear even in z mode
    render.appendVert(10, 10, 0, 0, 0xFFFFFFFF, 0);
    try std.testing.expectEqual(@as(u16, 0), c.rd16(c.ADDR_STREAM + 4 + 6 * 14 + 12));

    // z-mode raster is deterministic frame over frame
    raster.rasterize();
    var h0: u64 = 0;
    for (raster.fb) |b| h0 = h0 *% 31 +% b;
    raster.rasterize();
    var h1: u64 = 0;
    for (raster.fb) |b| h1 = h1 *% 31 +% b;
    try std.testing.expectEqual(h0, h1);

    // restore the default for other tests
    c.zbuf3d = false;
}
