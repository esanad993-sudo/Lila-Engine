//! v11/v12 3D MESH PIPELINE — the Burst/UE "engine crunches, game orchestrates"
//! pattern. The game fills a GLOBAL array with local-space vertices (3 fixed
//! per vertex) once (static geometry) or rarely, then calls
//!   draw3d(mesh, n, x, y, z, yaw, rgba);        // triangle soup
//!   draw3di(verts, nv, idx, ni, x, y, z, yaw, rgba); // indexed (shared verts)
//! per frame. THIS module owns the whole hot path in native Zig:
//!   model yaw-rotate + translate -> view (cam3 un-rotate) -> perspective
//!   project (fov 90, focal = half screen) -> per-tri backface cull +
//!   near-clip skip -> painter's depth sort (stable insertion, ties keep
//!   array order) -> depth-shaded raster via the shared vertex stream.
//!
//! v12 ARTICULATION: mat4/quat builtins OVER ARRAYS (FMatrix/FQuat as data,
//! one native opcode per operation — the game orchestrates the skeleton,
//! the engine crunches the linear algebra):
//!   quatAA — axis-angle quaternion (axis normalized in-engine, unit quat out)
//!   qMul   — Hamilton product (composes rotations)
//!   m4QT   — M = T(t) * R(q) in ONE op (the per-bone local transform)
//!   m4Mul  — full 4x4 product (world = parent * local), alias-safe
//!   skin3  — verts[voff..+2] = M * (x, y, z, 1) (rigid skinning)
//! Storage is COLUMN-MAJOR in the array cells (m[c*4+r], translation in
//! cells 12..14) — the GL/UE convention, so composition order reads
//! naturally: world = parent * local, point = M * p.
//!
//! Everything is fixed-point integer math + the shared ang LUT: bit-exact
//! determinism on native and wasm, captured by the rewind/rollback hash
//! chain like every other render output.
//!
//! Budgets (mirroring the engine's loud-limit philosophy):
//!   768 verts / 256 tris per call (the checker rejects bigger literals;
//!   the runtime clamps dynamic counts — never a crash, never a corrupt).
//!   rgba alpha == 0 selects WIREFRAME (3 edge bands per tri, RGB @ alpha FF).
//!
//! v12 Z-BUFFER VARIANT (profile knob c.zbuf3d <- engine_config.ZBUFFER3D /
//! runner --zbuffer): when on, each 3D tri also ships a 12-bit view depth in
//! the vertex token (bit 3 = zflag, bits 4..15 = depth, 1/32-unit steps,
//! clamped at 512 units) and the rasterizer depth-tests per pixel instead of
//! trusting the painter's sort — correct triangle-triangle occlusion for
//! intersecting/nested geometry. When off, the token is 0 and the stream is
//! byte-identical to v11.

const std = @import("std");
const c = @import("core.zig");
const render = @import("render.zig");

const MAX3D_VERTS: u32 = 768;
const MAX3D_TRIS: u32 = MAX3D_VERTS / 3;

// ---------------- v12: mat4/quat builtins over arrays ----------------
// Column-major storage in array cells: m[c*4 + r]; translation m12..m14.
// Every product runs in i64 and truncates >> 8 ONCE (same precision model
// as mulF, better than summing pre-truncated terms). All (array, offset)
// pairs go through arrIndex — mask|mod wrap, never a trap, always
// deterministic (the same contract as LD/ST_GARR).

/// Address of cell `idx` (wrapped) of global array `id`. Caller validates
/// the id (global) first — mirrors the VM's LD_GARR guard.
inline fn cellOf(id: u32, idx: i32) u32 {
    return c.arr_base[id] + @as(u32, c.arrIndex(id, idx)) * 4;
}

inline fn rdCell(id: u32, idx: i32) i32 {
    return @bitCast(c.rd32(cellOf(id, idx)));
}

inline fn wrCell(id: u32, idx: i32, v: i32) void {
    c.wr32(cellOf(id, idx), @bitCast(v));
}

/// q = axis-angle quaternion. Axis (Q24.8, normalized here), angle in ang
/// units. Zero axis -> identity. Odd angles round the half-angle by <= 1
/// ang unit (0.088 degrees) — deterministic everywhere.
pub fn quatAA(q_aid: u32, off: i32, ax: i32, ay: i32, az: i32, ang: i32) void {
    if (q_aid >= c.n_arrs or c.arr_ent[q_aid] != 0xFF) return;
    // squared axis length in i64 (Q24.8 values are up to 2^31 — i32 squares
    // would overflow); i64 squares of i32 sign-extended can't
    const l: i64 = @as(i64, ax) * ax + @as(i64, ay) * ay + @as(i64, az) * az;
    if (l == 0) {
        wrCell(q_aid, off, 0);
        wrCell(q_aid, off + 1, 0);
        wrCell(q_aid, off + 2, 0);
        wrCell(q_aid, off + 3, 256);
        return;
    }
    const len: i32 = @intCast(c.isqrt64(@abs(l)));
    const nx = c.divF(ax, len);
    const ny = c.divF(ay, len);
    const nz = c.divF(az, len);
    // exact halving mod 4096: (w + (w & 1)) >> 1, max +/-1 unit rounding
    const w: u32 = @as(u32, @bitCast(ang)) & 0xFFF;
    const half: i32 = @intCast((w + (w & 1)) >> 1);
    const s = c.sinA(half);
    wrCell(q_aid, off, c.mulF(nx, s));
    wrCell(q_aid, off + 1, c.mulF(ny, s));
    wrCell(q_aid, off + 2, c.mulF(nz, s));
    wrCell(q_aid, off + 3, c.cosA(half));
}

/// d = a * b (Hamilton). Reads all 8 source cells first: d may alias a or b.
pub fn qMul(d_aid: u32, doff: i32, a_aid: u32, aoff: i32, b_aid: u32, boff: i32) void {
    if (d_aid >= c.n_arrs or a_aid >= c.n_arrs or b_aid >= c.n_arrs) return;
    if (c.arr_ent[d_aid] != 0xFF or c.arr_ent[a_aid] != 0xFF or c.arr_ent[b_aid] != 0xFF) return;
    const ax = rdCell(a_aid, aoff);
    const ay = rdCell(a_aid, aoff + 1);
    const az = rdCell(a_aid, aoff + 2);
    const aw = rdCell(a_aid, aoff + 3);
    const bx = rdCell(b_aid, boff);
    const by = rdCell(b_aid, boff + 1);
    const bz = rdCell(b_aid, boff + 2);
    const bw = rdCell(b_aid, boff + 3);
    wrCell(d_aid, doff, c.mulF(aw, bx) +% c.mulF(ax, bw) +% c.mulF(ay, bz) -% c.mulF(az, by));
    wrCell(d_aid, doff + 1, c.mulF(aw, by) -% c.mulF(ax, bz) +% c.mulF(ay, bw) +% c.mulF(az, bx));
    wrCell(d_aid, doff + 2, c.mulF(aw, bz) +% c.mulF(ax, by) -% c.mulF(ay, bx) +% c.mulF(az, bw));
    wrCell(d_aid, doff + 3, c.mulF(aw, bw) -% c.mulF(ax, bx) -% c.mulF(ay, by) -% c.mulF(az, bz));
}

/// M = T(t) * R(q) — the per-bone LOCAL transform in ONE opcode (a full
/// compose would cost two temps + a 4x4 product per bone per frame).
pub fn m4QT(m_aid: u32, moff: i32, q_aid: u32, qoff: i32, tx: i32, ty: i32, tz: i32) void {
    if (m_aid >= c.n_arrs or q_aid >= c.n_arrs) return;
    if (c.arr_ent[m_aid] != 0xFF or c.arr_ent[q_aid] != 0xFF) return;
    const x = rdCell(q_aid, qoff);
    const y = rdCell(q_aid, qoff + 1);
    const z = rdCell(q_aid, qoff + 2);
    const w = rdCell(q_aid, qoff + 3);
    const xx = c.mulF(x, x);
    const yy = c.mulF(y, y);
    const zz = c.mulF(z, z);
    const xy = c.mulF(x, y);
    const xz = c.mulF(x, z);
    const yz = c.mulF(y, z);
    const wx = c.mulF(w, x);
    const wy = c.mulF(w, y);
    const wz = c.mulF(w, z);
    // col 0
    wrCell(m_aid, moff, 256 -% ((yy +% zz) << 1));
    wrCell(m_aid, moff + 1, (xy +% wz) << 1);
    wrCell(m_aid, moff + 2, (xz -% wy) << 1);
    wrCell(m_aid, moff + 3, 0);
    // col 1
    wrCell(m_aid, moff + 4, (xy -% wz) << 1);
    wrCell(m_aid, moff + 5, 256 -% ((xx +% zz) << 1));
    wrCell(m_aid, moff + 6, (yz +% wx) << 1);
    wrCell(m_aid, moff + 7, 0);
    // col 2
    wrCell(m_aid, moff + 8, (xz +% wy) << 1);
    wrCell(m_aid, moff + 9, (yz -% wx) << 1);
    wrCell(m_aid, moff + 10, 256 -% ((xx +% yy) << 1));
    wrCell(m_aid, moff + 11, 0);
    // col 3 = translation
    wrCell(m_aid, moff + 12, tx);
    wrCell(m_aid, moff + 13, ty);
    wrCell(m_aid, moff + 14, tz);
    wrCell(m_aid, moff + 15, 256);
}

/// D = A * B (column-major), alias-safe: reads all 32 source cells, then
/// writes. D[c*4+r] = sum_k A[k*4+r] * B[c*4+k], i64 accumulate >> 8 once.
pub fn m4Mul(d_aid: u32, doff: i32, a_aid: u32, aoff: i32, b_aid: u32, boff: i32) void {
    if (d_aid >= c.n_arrs or a_aid >= c.n_arrs or b_aid >= c.n_arrs) return;
    if (c.arr_ent[d_aid] != 0xFF or c.arr_ent[a_aid] != 0xFF or c.arr_ent[b_aid] != 0xFF) return;
    var a: [16]i64 = undefined;
    var b: [16]i64 = undefined;
    var k: u32 = 0;
    while (k < 16) : (k += 1) {
        a[k] = rdCell(a_aid, aoff + @as(i32, @intCast(k)));
        b[k] = rdCell(b_aid, boff + @as(i32, @intCast(k)));
    }
    var col: u32 = 0;
    while (col < 4) : (col += 1) {
        var row: u32 = 0;
        while (row < 4) : (row += 1) {
            var acc: i64 = 0;
            k = 0;
            while (k < 4) : (k += 1) {
                acc += a[k * 4 + row] * b[col * 4 + k];
            }
            wrCell(d_aid, doff + @as(i32, @intCast(col * 4 + row)), @truncate(acc >> 8));
        }
    }
}

/// Rigid skin: verts[voff..+2] = M * (x, y, z, 1). Row i: sum_k M[i][k]*p_k
/// with the i64 accumulate >> 8 once, translation added in Q24.8.
pub fn skin3(v_aid: u32, voff: i32, m_aid: u32, moff: i32, x: i32, y: i32, z: i32) void {
    if (v_aid >= c.n_arrs or m_aid >= c.n_arrs) return;
    if (c.arr_ent[v_aid] != 0xFF or c.arr_ent[m_aid] != 0xFF) return;
    var m: [16]i64 = undefined;
    var k: u32 = 0;
    while (k < 16) : (k += 1) {
        m[k] = rdCell(m_aid, moff + @as(i32, @intCast(k)));
    }
    wrCell(v_aid, voff, @as(i32, @truncate((m[0] * x + m[4] * y + m[8] * z) >> 8)) +% @as(i32, @truncate(m[12])));
    wrCell(v_aid, voff + 1, @as(i32, @truncate((m[1] * x + m[5] * y + m[9] * z) >> 8)) +% @as(i32, @truncate(m[13])));
    wrCell(v_aid, voff + 2, @as(i32, @truncate((m[2] * x + m[6] * y + m[10] * z) >> 8)) +% @as(i32, @truncate(m[14])));
}

// draw-pass scratch (draw fns never run on parallel workers — only update
// fns are group-scheduled — so a plain static is race-free by construction)
var sx_px: [MAX3D_VERTS]i32 = undefined;
var sy_px: [MAX3D_VERTS]i32 = undefined;
var vfwd: [MAX3D_VERTS]i32 = undefined;
var tri_depth: [MAX3D_TRIS]i32 = undefined;
var tri_order: [MAX3D_TRIS]u16 = undefined;

inline fn mul256(a: i32, b: i32) i32 {
    return @intCast((@as(i64, a) * @as(i64, b)) >> 8);
}

// ---- v15 depth-cue curve -------------------------------------------------
//
// The v1 curve was linear-in-depth with a hard clamp: full brightness
// inside 1 world unit, then dark = min((d-1)/8, 176) — a 31% floor at
// 6.5 units and NOTHING beyond. But LILA scenes are authored at HUNDREDS
// of units (the z token encodes 12 bits of depth, horizon 512; the demo
// camera sits ~250-340 units from the hero cube) — so every object in a
// real game rendered permanently on the floor: the demo cube filled a
// third of the screen at 31% brightness, with zero depth separation
// anywhere (the actual reported symptom).
//
// The replacement is the physical inverse-square law with a fog floor:
//
//   shade(d) = FLOOR + (256 - FLOOR) * K / (K + e^2),   e = max(d - NEAR, 0)
//
//   NEAR  = 1 unit    — closer than this, full brightness
//   REF   = 400 units — K = REF^2, the falloff scale: calibrated so the
//                        demo hero cube (~293 units mean view depth) renders
//                        at ~78%, its near/far faces at 83%/74% (internal
//                        depth gradient, visible for the first time)
//   FLOOR = 96 / 256  — 37.5% far-field asymptote; distant geometry stays
//                        readable (the point of a floor) but clearly dimmer
//
// Why this shape: smooth (no clamp knee anywhere), monotone, and
// perceptually well spaced — like real light attenuation it front-loads
// the contrast in the near field and keeps separating depth forever,
// approaching the floor instead of hitting it. Calibrated against the
// engine's actual scene scale (view depths in units):
//
//   depth:     1     8    50   150   293(hero)  512(z horizon)  units
//   shade:   100%  ~100%  99%   92%    78%        61%      (v15)
//   v1:      100%   31%   31%   31%    31%        31%      (!!)
//
// All math is i64 with ONE division per TRIANGLE (never per pixel), so
// the rasterizer cost is unchanged and the result is bit-deterministic.
// e comes from Q24.8 depth: |e| < 2^31, e^2 < 2^62 — no i64 overflow.
pub const DEPTH_SHADE_NEAR: i64 = 256; // 1 world unit (Q24.8)
pub const DEPTH_SHADE_REF: i64 = 400; // falloff scale, world units (see table)
pub const DEPTH_SHADE_FLOOR: u32 = 96; // 37.5% far-field floor, Q8

const DS_K: i64 = DEPTH_SHADE_REF * DEPTH_SHADE_REF * 256 * 256; // (REF units)^2 in Q8^2
const DS_SPAN: i64 = 256 - @as(i64, DEPTH_SHADE_FLOOR); // floor-to-full span

/// Shade a solid tri by view depth: nearer = brighter (Q8 modulation).
/// `depth` is the mean view-forward of the tri (Q24.8). Smooth
/// inverse-square falloff onto a 37.5% fog floor — see the curve table
/// above. Pure integer, one division; used per tri by draw3d/draw3di.
pub fn depthShade(rgba: u32, depth: i32) u32 {
    var e: i64 = @as(i64, depth) - DEPTH_SHADE_NEAR;
    if (e < 0) e = 0;
    const e2 = e * e;
    const fall: u32 = @intCast(@divTrunc(DS_SPAN * DS_K, DS_K + e2)); // 0..160
    const shade: u32 = DEPTH_SHADE_FLOOR + fall; // 96..256
    const cm = struct {
        fn f(x: u32, y: u32) u32 {
            return @min((x * y) >> 8, 255);
        }
    }.f;
    return (cm((rgba >> 24) & 0xFF, shade) << 24) |
        (cm((rgba >> 16) & 0xFF, shade) << 16) |
        (cm((rgba >> 8) & 0xFF, shade) << 8) |
        (rgba & 0xFF);
}

/// v12 z-buffer variant: ship the per-vertex view depth in the token so the
/// rasterizer can depth-test per pixel. bit 3 = zflag, bits 4..15 = 12-bit
/// depth in 1/32-unit steps (clamped at 512 units). Off-knob -> 0: the
/// stream stays byte-identical to v11. Only called with fwd >= PROJ_NEAR.
inline fn tok3d(fwd: i32) u32 {
    if (!c.zbuf3d) return 0;
    var d: u32 = @intCast((fwd >> 5) + 1);
    if (d > 4095) d = 4095;
    return 0x8 | (d << 4);
}

/// One edge as a thin quad (2 tris, 6 verts): extrude 1px along the
/// dominant-axis perpendicular — deterministic, always visible, never the
/// fixed-point/pixel mixup that blew bands up to screen size. `tok` rides
/// through so wireframe tris depth-test too (z variant).
fn edgeBand(x0: i32, y0: i32, x1: i32, y1: i32, rgba: u32, tok: u32) void {
    const dx = x1 - x0;
    const dy = y1 - y0;
    if (dx == 0 and dy == 0) return; // degenerate edge
    const ox: i32 = if (@abs(dx) > @abs(dy)) 0 else 1;
    const oy: i32 = if (@abs(dx) > @abs(dy)) 1 else 0;
    const col = (rgba & 0xFFFFFF00) | 0xFF; // alpha-0 mode color -> solid stroke
    render.appendVert(x0, y0, 0, 0, col, tok);
    render.appendVert(x1, y1, 0, 0, col, tok);
    render.appendVert(x1 + ox, y1 + oy, 0, 0, col, tok);
    render.appendVert(x0, y0, 0, 0, col, tok);
    render.appendVert(x1 + ox, y1 + oy, 0, 0, col, tok);
    render.appendVert(x0 + ox, y0 + oy, 0, 0, col, tok);
}

/// The whole draw3d pipeline for one call. Stack args arrived through the VM
/// (popped rgba, yaw, z, y, x, n); `arr_id` rides the opcode operand.
pub fn draw3d(arr_id: u32, n_in: i32, mx: i32, my: i32, mz: i32, yaw: i32, rgba: u32) void {
    if (arr_id >= c.n_arrs or c.arr_ent[arr_id] != 0xFF) return;
    const cap_cells: u32 = c.arr_cap[arr_id];
    // n counts VERTICES; the array holds 3 cells per vertex. Clamp to both
    // the per-call budget and the array capacity, floored to whole tris.
    var n: u32 = @bitCast(n_in);
    const max_by_cells: u32 = cap_cells / 3;
    if (n > MAX3D_VERTS) n = MAX3D_VERTS;
    if (n > max_by_cells) n = max_by_cells;
    n -= n % 3;
    if (n < 3) return;

    // camera + model trig once per call (shared ang LUT — deterministic)
    // MODEL yaw drives the mesh transform; the VIEW un-rotation must use the
    // CAMERA's yaw (the bug that made every oblique view face backwards —
    // top-down happened to work because both yaws were 0).
    const msy = c.sinA(yaw);
    const mcy = c.cosA(yaw);
    const cam_yaw: i32 = @as(i32, @bitCast(c.rd32(c.ADDR_CAM3_YAW)));
    const vsy = c.sinA(cam_yaw);
    const vcy = c.cosA(cam_yaw);
    const spit = c.sinA(@as(i32, @bitCast(c.rd32(c.ADDR_CAM3_PITCH))));
    const cpit = c.cosA(@as(i32, @bitCast(c.rd32(c.ADDR_CAM3_PITCH))));
    const camx: i32 = @bitCast(c.rd32(c.ADDR_CAM3_X));
    const camy: i32 = @bitCast(c.rd32(c.ADDR_CAM3_Y));
    const camz: i32 = @bitCast(c.rd32(c.ADDR_CAM3_Z));
    const half_w: i32 = @intCast((c.game_w / 2) << 8); // focal AND screen center
    const half_h: i32 = @intCast((c.game_h / 2) << 8);
    const base = c.arr_base[arr_id];

    // ---- model -> view -> project (per vertex, native hot loop) ----
    var vi: u32 = 0;
    while (vi < n) : (vi += 1) {
        const lx: i32 = @bitCast(c.rd32(base + vi * 12));
        const ly: i32 = @bitCast(c.rd32(base + vi * 12 + 4));
        const lz: i32 = @bitCast(c.rd32(base + vi * 12 + 8));
        // model yaw around Z + translate to world
        const wx = mul256(lx, mcy) -% mul256(ly, msy) +% mx;
        const wy = mul256(lx, msy) +% mul256(ly, mcy) +% my;
        const wz = lz +% mz;
        // view: translate by camera, un-rotate CAMERA yaw, un-rotate pitch
        const dx = wx -% camx;
        const dy = wy -% camy;
        const dz = wz -% camz;
        const right = mul256(dx, vcy) -% mul256(dy, vsy);
        const fwd0 = mul256(dx, vsy) +% mul256(dy, vcy);
        const fwd = mul256(fwd0, cpit) +% mul256(dz, spit);
        const up = mul256(dz, cpit) -% mul256(fwd0, spit);
        vfwd[vi] = fwd;
        if (fwd < c.PROJ_NEAR) {
            sx_px[vi] = 0;
            sy_px[vi] = 0;
        } else {
            const sxf = half_w +% c.divF(mul256(right, half_w), fwd);
            const syf = half_h -% c.divF(mul256(up, half_w), fwd);
            sx_px[vi] = sxf >> 8;
            sy_px[vi] = syf >> 8;
        }
    }

    // ---- per-tri culling + depth keys ----
    const wire = (rgba & 0xFF) == 0;
    var n_tris: u32 = 0;
    var t: u32 = 0;
    while (t < n) : (t += 3) {
        // near plane: whole-tri skip if any vertex is behind (conservative v1)
        if (vfwd[t] < c.PROJ_NEAR or vfwd[t + 1] < c.PROJ_NEAR or vfwd[t + 2] < c.PROJ_NEAR) continue;
        const x0 = sx_px[t];
        const y0 = sy_px[t];
        const x1 = sx_px[t + 1];
        const y1 = sy_px[t + 1];
        const x2 = sx_px[t + 2];
        const y2 = sy_px[t + 2];
        // backface: signed area in PIXEL space. Screen y grows downward, so
        // the front face (counter-clockwise in math coords) flips sign:
        // keep area < 0, cull area >= 0 (verified against the demo cube).
        const area = @as(i64, x1 - x0) * @as(i64, y2 - y0) -
            @as(i64, x2 - x0) * @as(i64, y1 - y0);
        if (area >= 0) continue;
        tri_depth[n_tris] = @intCast(@divTrunc(@as(i64, vfwd[t]) + vfwd[t + 1] + vfwd[t + 2], 3));
        tri_order[n_tris] = @intCast(t);
        n_tris += 1;
    }

    // ---- painter's sort: far (large fwd) first; stable insertion keeps
    // array order on ties — the sort is deterministic end to end ----
    var i: u32 = 1;
    while (i < n_tris) : (i += 1) {
        const d = tri_depth[i];
        const o = tri_order[i];
        var j: u32 = i;
        while (j > 0 and tri_depth[j - 1] < d) : (j -= 1) {
            tri_depth[j] = tri_depth[j - 1];
            tri_order[j] = tri_order[j - 1];
        }
        tri_depth[j] = d;
        tri_order[j] = o;
    }

    // ---- emit into the shared stream (raster paints in stream order) ----
    var k: u32 = 0;
    while (k < n_tris) : (k += 1) {
        const t0 = tri_order[k];
        const x0 = sx_px[t0];
        const y0 = sy_px[t0];
        const x1 = sx_px[t0 + 1];
        const y1 = sy_px[t0 + 1];
        const x2 = sx_px[t0 + 2];
        const y2 = sy_px[t0 + 2];
        const ztok = tok3d(tri_depth[k]);
        if (wire) {
            edgeBand(x0, y0, x1, y1, rgba, ztok);
            edgeBand(x1, y1, x2, y2, rgba, ztok);
            edgeBand(x2, y2, x0, y0, rgba, ztok);
        } else {
            const col = depthShade(rgba, tri_depth[k]);
            render.appendVert(x0, y0, 0, 0, col, ztok);
            render.appendVert(x1, y1, 0, 0, col, ztok);
            render.appendVert(x2, y2, 0, 0, col, ztok);
        }
    }
}

/// v12: the INDEXED mesh pass — the GPU vertex/index-buffer vocabulary.
/// `verts` holds 3 fixed cells per UNIQUE vertex (nv of them, shared by
/// triangles and animated in place by skinv); `idx` holds vertex indices,
/// 3 per tri. The transform/view/project/cull/sort/emit pipeline is
/// identical to draw3d; the only difference is the vertex fetch through
/// the index buffer. Out-of-range indices DROP the triangle (the same
/// no-trap contract as every array access).
pub fn draw3di(v_aid: u32, i_aid: u32, nv_in: i32, ni_in: i32, mx: i32, my: i32, mz: i32, yaw: i32, rgba: u32) void {
    if (v_aid >= c.n_arrs or i_aid >= c.n_arrs) return;
    if (c.arr_ent[v_aid] != 0xFF or c.arr_ent[i_aid] != 0xFF) return;
    var nv: u32 = @bitCast(nv_in);
    const max_by_cells: u32 = c.arr_cap[v_aid] / 3;
    if (nv > MAX3D_VERTS) nv = MAX3D_VERTS;
    if (nv > max_by_cells) nv = max_by_cells;
    if (nv < 3) return;
    var ni: u32 = @bitCast(ni_in);
    if (ni > c.arr_cap[i_aid]) ni = c.arr_cap[i_aid];
    ni -= ni % 3;
    if (ni < 3) return;

    // camera + model trig once per call — identical convention to draw3d
    const msy = c.sinA(yaw);
    const mcy = c.cosA(yaw);
    const cam_yaw: i32 = @as(i32, @bitCast(c.rd32(c.ADDR_CAM3_YAW)));
    const vsy = c.sinA(cam_yaw);
    const vcy = c.cosA(cam_yaw);
    const spit = c.sinA(@as(i32, @bitCast(c.rd32(c.ADDR_CAM3_PITCH))));
    const cpit = c.cosA(@as(i32, @bitCast(c.rd32(c.ADDR_CAM3_PITCH))));
    const camx: i32 = @bitCast(c.rd32(c.ADDR_CAM3_X));
    const camy: i32 = @bitCast(c.rd32(c.ADDR_CAM3_Y));
    const camz: i32 = @bitCast(c.rd32(c.ADDR_CAM3_Z));
    const half_w: i32 = @intCast((c.game_w / 2) << 8);
    const half_h: i32 = @intCast((c.game_h / 2) << 8);
    const vbase = c.arr_base[v_aid];

    // ---- model -> view -> project per UNIQUE vertex (native hot loop) ----
    var vi: u32 = 0;
    while (vi < nv) : (vi += 1) {
        const lx: i32 = @bitCast(c.rd32(vbase + vi * 12));
        const ly: i32 = @bitCast(c.rd32(vbase + vi * 12 + 4));
        const lz: i32 = @bitCast(c.rd32(vbase + vi * 12 + 8));
        const wx = mul256(lx, mcy) -% mul256(ly, msy) +% mx;
        const wy = mul256(lx, msy) +% mul256(ly, mcy) +% my;
        const wz = lz +% mz;
        const dx = wx -% camx;
        const dy = wy -% camy;
        const dz = wz -% camz;
        const right = mul256(dx, vcy) -% mul256(dy, vsy);
        const fwd0 = mul256(dx, vsy) +% mul256(dy, vcy);
        const fwd = mul256(fwd0, cpit) +% mul256(dz, spit);
        const up = mul256(dz, cpit) -% mul256(fwd0, spit);
        vfwd[vi] = fwd;
        if (fwd < c.PROJ_NEAR) {
            sx_px[vi] = 0;
            sy_px[vi] = 0;
        } else {
            const sxf = half_w +% c.divF(mul256(right, half_w), fwd);
            const syf = half_h -% c.divF(mul256(up, half_w), fwd);
            sx_px[vi] = sxf >> 8;
            sy_px[vi] = syf >> 8;
        }
    }

    // ---- per-tri cull + depth keys, vertices fetched through the index
    // buffer; any out-of-range index drops the tri (no trap, ever) ----
    const wire = (rgba & 0xFF) == 0;
    const ibase = c.arr_base[i_aid];
    var n_tris: u32 = 0;
    var t: u32 = 0;
    while (t < ni) : (t += 3) {
        const v0: u32 = @bitCast(c.rd32(ibase + t * 4));
        const v1: u32 = @bitCast(c.rd32(ibase + (t + 1) * 4));
        const v2: u32 = @bitCast(c.rd32(ibase + (t + 2) * 4));
        if (v0 >= nv or v1 >= nv or v2 >= nv) continue;
        if (vfwd[v0] < c.PROJ_NEAR or vfwd[v1] < c.PROJ_NEAR or vfwd[v2] < c.PROJ_NEAR) continue;
        const x0 = sx_px[v0];
        const y0 = sy_px[v0];
        const x1 = sx_px[v1];
        const y1 = sy_px[v1];
        const x2 = sx_px[v2];
        const y2 = sy_px[v2];
        // backface: same screen-winding rule as draw3d (keep area < 0)
        const area = @as(i64, x1 - x0) * @as(i64, y2 - y0) -
            @as(i64, x2 - x0) * @as(i64, y1 - y0);
        if (area >= 0) continue;
        tri_depth[n_tris] = @intCast(@divTrunc(@as(i64, vfwd[v0]) + vfwd[v1] + vfwd[v2], 3));
        tri_order[n_tris] = @intCast(t);
        n_tris += 1;
    }

    // ---- painter's sort (far first, stable) — with the z-buffer knob on
    // this order only fixes blend layering; occlusion is per-pixel ----
    var i: u32 = 1;
    while (i < n_tris) : (i += 1) {
        const d = tri_depth[i];
        const o = tri_order[i];
        var j: u32 = i;
        while (j > 0 and tri_depth[j - 1] < d) : (j -= 1) {
            tri_depth[j] = tri_depth[j - 1];
            tri_order[j] = tri_order[j - 1];
        }
        tri_depth[j] = d;
        tri_order[j] = o;
    }

    // ---- emit: the stream tri is (idx[t], idx[t+1], idx[t+2]) ----
    var k: u32 = 0;
    while (k < n_tris) : (k += 1) {
        const t0 = tri_order[k];
        const v0: u32 = @bitCast(c.rd32(ibase + t0 * 4));
        const v1: u32 = @bitCast(c.rd32(ibase + (t0 + 1) * 4));
        const v2: u32 = @bitCast(c.rd32(ibase + (t0 + 2) * 4));
        const x0 = sx_px[v0];
        const y0 = sy_px[v0];
        const x1 = sx_px[v1];
        const y1 = sy_px[v1];
        const x2 = sx_px[v2];
        const y2 = sy_px[v2];
        const ztok = tok3d(tri_depth[k]);
        if (wire) {
            edgeBand(x0, y0, x1, y1, rgba, ztok);
            edgeBand(x1, y1, x2, y2, rgba, ztok);
            edgeBand(x2, y2, x0, y0, rgba, ztok);
        } else {
            const col = depthShade(rgba, tri_depth[k]);
            render.appendVert(x0, y0, 0, 0, col, ztok);
            render.appendVert(x1, y1, 0, 0, col, ztok);
            render.appendVert(x2, y2, 0, 0, col, ztok);
        }
    }
}

// ---------------- v15 tests (aggregated by test_raster.zig) ----------------

test "depthShade: near field is exact passthrough (alpha untouched)" {
    // at or inside 1 world unit (and behind the camera): full brightness
    try std.testing.expectEqual(@as(u32, 0xAABBCCFF), depthShade(0xAABBCCFF, 0));
    try std.testing.expectEqual(@as(u32, 0xAABBCCFF), depthShade(0xAABBCCFF, 256));
    try std.testing.expectEqual(@as(u32, 0xAABBCCFF), depthShade(0xAABBCCFF, -9999));
    // alpha is coverage — never modulated
    try std.testing.expectEqual(@as(u32, 0x07), depthShade(0xAABBCC07, 100000 * 256) & 0xFF);
}

test "depthShade: monotone, 96/256 floor, demo cube at ~78%" {
    // non-increasing brightness over the whole 0..512-unit z range
    var last: u32 = 255;
    var units: i32 = 0;
    while (units <= 512) : (units += 8) {
        const v = (depthShade(0xFFFFFFFF, units * 256) >> 24) & 0xFF;
        try std.testing.expect(v <= last);
        last = v;
    }
    // far-field asymptote 96/256: channel 95 (255*96>>8)
    try std.testing.expectEqual(@as(u32, 95), (depthShade(0xFFFFFFFF, 100000 * 256) >> 24) & 0xFF);
    // the demo hero cube (~293 units mean view depth): ~200/256 = 78%
    const hero = (depthShade(0xFFFFFFFF, 293 * 256) >> 24) & 0xFF;
    try std.testing.expect(hero >= 197 and hero <= 201);
    // v1 pinned it to the 31% floor (80/256) — the fix, quantified
    try std.testing.expect(hero > 160);
}
