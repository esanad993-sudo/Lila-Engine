//! Software rasterizer: consumes the LILA vertex stream (14B quads/tris) and
//! fills a RGBA framebuffer. Implements the uber-shader's feature set in CPU
//! form: flat/gradient fill, additive blend, noise pattern, per-vertex color.
//! The WebGL host implements the identical feature set as a GPU uber-shader
//! (v15: including the optional gamma-correct blend mode — see S2L/L2S).

const std = @import("std");
const c = @import("core.zig");

pub const W: u32 = 512;
pub const H: u32 = 512;

pub var fb: [W * H * 4]u8 = undefined; // RGBA

// v12 Z-BUFFER VARIANT (profile knob c.zbuf3d): per-pixel depth for
// zflagged 3D tris. Depth is the 12-bit view-forward from the vertex token
// (bit 3 = zflag, bits 4..15 = depth). 2D tris (bit 3 clear) never touch
// it and paint in stream order exactly like the painter's path, so HUD /
// sprites / starfield are unaffected. Cleared to 0xFFFF (far) per frame —
// only when the knob is on; the default path never reads or writes this.
pub var zbuf: [W * H]u16 = undefined;

// ---------------- v15: gamma-correct blending (opt-in, c.gamma_blend) ----------------
//
// 8-bit sRGB values are PERCEPTUAL codes, not light: mixing them directly
// darkens every intermediate color (a 50/50 blend of black and white lands
// on sRGB 128 = 22% light instead of 50%). The gamma mode linearizes RGB
// before any mix and re-encodes after — that 50/50 blend lands on 188, and
// gradient midtones / glow falloffs stop sagging toward mud. Alpha is
// COVERAGE, not light: it is never converted (the convention every
// premultiplied-alpha pipeline uses).
//
// Implementation: the sRGB transfer functions are baked at comptime into
// two LUTs (the exact technique as SIN_LUT / NOTE_STEPS — evaluated once
// per build, then pure integer lookups at runtime; the hot path never
// touches a float). S2L: sRGB code -> linear light on a 0..65535 scale
// (u16 keeps dark-band precision: adjacent codes near black sit ~5 linear
// LSBs apart). L2S: the full 64K-entry inverse, so encoding is one load.
// The pair round-trips exactly — L2S[S2L[c]] == c for all 256 c, verified
// by test_raster.zig — so flat colors pass through both directions
// unchanged and the opaque fast path needs no conversion at all.
//
// Cost: blended pixels only (alpha 1..254 or additive) pay 6 LUT loads;
// gradient tris interpolate in linear (12 setup loads, 3 encodes/pixel).
// Opaque pixels are unchanged. Knob off: every path byte-identical to v14.
// The knob is pixel-output-only — fb is not part of the state hash, so
// determinism / replay / rollback are untouched either way.

const S2L: [256]u16 = blk: {
    @setEvalBranchQuota(10000);
    var t: [256]u16 = undefined;
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        const s = @as(f64, @floatFromInt(i)) / 255.0;
        // sRGB EOTF; x^2.4 via exp/log (comptime-cheap, same libm-baked
        // determinism as SIN_LUT's @sin)
        const lin = if (s <= 0.04045) s / 12.92 else @exp(2.4 * @log((s + 0.055) / 1.055));
        var v = @round(lin * 65535.0);
        if (v < 0) v = 0;
        if (v > 65535.0) v = 65535.0;
        t[i] = @intFromFloat(v);
    }
    break :blk t;
};

const L2S: [65536]u8 = blk: {
    @setEvalBranchQuota(500000);
    var t: [65536]u8 = undefined;
    var i: u32 = 0;
    while (i < 65536) : (i += 1) {
        const lin = @as(f64, @floatFromInt(i)) / 65535.0;
        // inverse EOTF; x^(1/2.4) via exp/log
        const s = if (lin <= 0.0031308) lin * 12.92 else 1.055 * @exp(@log(lin) / 2.4) - 0.055;
        var v = @round(s * 255.0);
        if (v < 0) v = 0;
        if (v > 255.0) v = 255.0;
        t[i] = @intFromFloat(v);
    }
    break :blk t;
};

/// The two background row patterns (the 64px checker), precomputed at
/// comptime: clear() becomes 512 row memcpys instead of a 262k-iteration
/// pixel loop — ~3x faster on a full-screen clear, and identical bytes.
const BG_ROW_A: [W * 4]u8 = blk: {
    var row: [W * 4]u8 = undefined;
    var x: u32 = 0;
    while (x < W) : (x += 1) {
        const v: u8 = if ((x ^ @as(u32, 0)) & 64 == 0) 12 else 8;
        row[x * 4] = v;
        row[x * 4 + 1] = v + 2;
        row[x * 4 + 2] = v + 10;
        row[x * 4 + 3] = 255;
    }
    break :blk row;
};
const BG_ROW_B: [W * 4]u8 = blk: {
    var row: [W * 4]u8 = undefined;
    var x: u32 = 0;
    while (x < W) : (x += 1) {
        const v: u8 = if ((x ^ @as(u32, 64)) & 64 == 0) 12 else 8;
        row[x * 4] = v;
        row[x * 4 + 1] = v + 2;
        row[x * 4 + 2] = v + 10;
        row[x * 4 + 3] = 255;
    }
    break :blk row;
};

pub fn clear() void {
    // dark blue-black backdrop (64px checker) via precomputed row templates
    var y: u32 = 0;
    while (y < H) : (y += 1) {
        const row = if ((y & 64) == 0) &BG_ROW_A else &BG_ROW_B;
        @memcpy(fb[y * W * 4 .. (y + 1) * W * 4], row);
    }
}

/// Rasterize the current stream into fb. Triangles come in groups of 3
/// vertices (font quads emit 2 tris = 6 verts; sprites emit their baked
/// triangle soup). Vertex tokens: bit0 additive, bits1-2 pattern.
pub fn rasterize() void {
    clear();
    if (c.zbuf3d) {
        // v12 z-buffer variant: same stream, but zflagged tris depth-test
        // per pixel; everything else paints in stream order (painter's)
        @memset(&zbuf, 0xFFFF);
        const n = c.rd32(c.ADDR_STREAM);
        var i: u32 = 0;
        while (i + 3 <= n) : (i += 3) {
            triZ(loadVert(i), loadVert(i + 1), loadVert(i + 2));
        }
        return;
    }
    const n = c.rd32(c.ADDR_STREAM);
    var i: u32 = 0;
    while (i + 3 <= n) : (i += 3) {
        tri(loadVert(i), loadVert(i + 1), loadVert(i + 2));
    }
}

const Vert = struct { x: i32, y: i32, rgba: u32, tok: u32 };

fn loadVert(idx: u32) Vert {
    const base = c.ADDR_STREAM + 4 + idx * 14;
    return .{
        .x = @as(i16, @bitCast(c.rd16(base))),
        .y = @as(i16, @bitCast(c.rd16(base + 2))),
        .rgba = c.rd32(base + 8),
        .tok = c.rd16(base + 12),
    };
}

// ---------------- v14 rasterizer core ----------------
//
// FIDELITY CONTRACT (all fixes are pixel-output-only; the engine-state
// hash never covers the framebuffer, so determinism guarantees are
// unchanged):
//
//   1. Half-pixel sample centers. Edge functions are evaluated at
//      (x+0.5, y+0.5) via 2x internal coordinates, so integer quad
//      boundaries never coincide with sample points — the two triangles
//      of a sprite/font quad no longer double-blend their shared diagonal
//      (alpha < 255 used to paint a visible seam), and adjacent quads
//      tile without gaps or overlap: a 0..N quad covers exactly N pixels.
//
//   2. Top-left tie-break. Exact sample-on-edge hits (45-degree diagonals
//      through half-pixel centers) resolve with the antisymmetric rule
//      "dy > 0, or dy == 0 and dx > 0" — shared edges are owned by exactly
//      one of the two adjacent triangles. Never both, never neither.
//
//   3. Gouraud shading. Per-vertex RGBA is interpolated barycentrically.
//      The v1 path used vertex 0's color for the whole tri, so gradient
//      sprites (fill/grad pairs) rendered flat. Interpolation is affine and
//      fully incremental: 3 i64 adds per pixel for the edges, 4 more for
//      the channels — no per-pixel multiplies at all (the old inside test
//      alone cost 6). Flat-color triangles are bit-exact flat (the three
//      interpolation steps are identically zero).
//
//   4. Z interpolation is incremental too (the v12 path divided per pixel).
//
//   5. Exact blending. Source-over uses true /255 convex weights with
//      round-to-nearest (the old >>8 dropped up to a full LSB and
//      under-blended every pixel); additive SATURATES at 255 (the old
//      @truncate wrapped overlapping glows back to black).
//
//   6. v15 optional gamma-correct mode (c.gamma_blend): RGB mixes in
//      LINEAR light through the comptime sRGB LUTs above — blends,
//      additive accumulation, gradient interpolation and the noise
//      darkening all switch space. Alpha stays coverage. Off = the v14
//      paths, bit for bit.
//
// Everything stays integer and bit-deterministic on every target.

/// One triangle, both modes. `zmode` = the v12 z-buffer profile knob.
fn rasterTri(a_in: Vert, b_in: Vert, cc_in: Vert, zmode: bool) void {
    // token bits describe the whole tri (stream contract: vertex 0's token)
    const additive = (a_in.tok & 1) != 0;
    const pattern = (a_in.tok >> 1) & 3;
    const zflag = zmode and (a_in.tok & 8) != 0;

    // CCW normalization in 2x space: positive area = counter-clockwise in
    // the y-down screen convention = interior on the positive side of every
    // directed edge. A negative-area input just swaps two vertices.
    var p = [3]Vert{ a_in, b_in, cc_in };
    var area: i64 = triArea2x(p[0], p[1], p[2]);
    if (area == 0) return; // degenerate
    if (area < 0) {
        const t = p[1];
        p[1] = p[2];
        p[2] = t;
        area = -area;
    }

    // bounding box in whole pixels (vertices are pixel-quantized)
    var minx = @min(@min(p[0].x, p[1].x), p[2].x);
    var maxx = @max(@max(p[0].x, p[1].x), p[2].x);
    var miny = @min(@min(p[0].y, p[1].y), p[2].y);
    var maxy = @max(@max(p[0].y, p[1].y), p[2].y);
    minx = @max(minx, 0);
    miny = @max(miny, 0);
    maxx = @min(maxx, @as(i32, W) - 1);
    maxy = @min(maxy, @as(i32, H) - 1);
    if (minx > maxx or miny > maxy) return;

    // edge vectors in 2x space: e[i] goes p[i] -> p[(i+1)%3]
    var ex: [3]i64 = undefined;
    var ey: [3]i64 = undefined;
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        const j = (i + 1) % 3;
        ex[i] = @as(i64, p[j].x) * 2 - @as(i64, p[i].x) * 2;
        ey[i] = @as(i64, p[j].y) * 2 - @as(i64, p[i].y) * 2;
    }

    // top-left rule, folded into the edge functions: pixels exactly on an
    // edge (s == 0, possible because 2x-space edge values are even) resolve
    // by the antisymmetric rule "dy > 0, or dy == 0 and dx > 0" — the edge
    // owner draws, the neighbor skips. Precomputing a +1/-1 bias per edge
    // turns the rule into a plain sign test (s + bias > 0), branch-free:
    //   top-left edge, s == 0 : 0 + 1  = 1  > 0  -> draw
    //   other edge,    s == 0 : 0 - 1  = -1     -> skip
    //   any edge,      s > 0  (even, so >= 2): stays positive after -1
    //   any edge,      s < 0  (even, so <= -2): stays negative after +1
    var bias: [3]i64 = undefined;
    i = 0;
    while (i < 3) : (i += 1) {
        bias[i] = if (ey[i] > 0 or (ey[i] == 0 and ex[i] > 0)) 1 else -1;
    }

    // edge-function accumulators at the first sample center (2*minx+1, ...)
    var s: [3]i64 = undefined;
    i = 0;
    while (i < 3) : (i += 1) {
        const sx: i64 = @as(i64, minx) * 2 + 1;
        const sy: i64 = @as(i64, miny) * 2 + 1;
        const xi: i64 = @as(i64, p[i].x) * 2;
        const yi: i64 = @as(i64, p[i].y) * 2;
        s[i] = ex[i] * (sy - yi) - ey[i] * (sx - xi);
    }

    // ---- per-channel color + depth interpolation setup (Q16.16) ----
    // color = (s1*c0 + s2*c1 + s0*c2) / area (barycentric: edge i is
    // opposite vertex i's weight... e[i]=p[i]->p[i+1], so weight of p0 is
    // the edge p1->p2 = s[1], of p1 is s[2], of p2 is s[0]).
    // One PIXEL is 2 units in 2x space: every per-pixel slope below carries
    // the x2 factor. dchannel/dx = -2*ey-weighted mix / area, etc.
    // Flat triangles: the ey (or ex) terms sum to zero -> steps are 0.
    //
    // FLAT FAST PATH: text glyphs, stars, 3D tris and non-gradient sprites
    // carry the same color on every vertex. For those we skip the 4-channel
    // interpolation entirely (the per-pixel work collapses to the old
    // constant-color cost) — only gradient sprites pay for Gouraud.
    const flat_col: ?u32 =
        if (p[0].rgba == p[1].rgba and p[1].rgba == p[2].rgba) p[0].rgba else null;
    var colx: [4]i64 = undefined; // R,G,B,A at row start, Q16.16
    var stepcx: [4]i64 = undefined; // per +1 pixel
    var stepcy: [4]i64 = undefined; // per +1 row
    if (flat_col == null) {
        const cols = [3][4]u32{
            unpackRgba(p[0].rgba),
            unpackRgba(p[1].rgba),
            unpackRgba(p[2].rgba),
        };
        // v15 gamma mode: RGB interpolates in LINEAR light (the whole point
        // — an sRGB-space lerp puts every gradient midpoint in the mud);
        // alpha stays on the 0..255 coverage scale.
        const lin = c.gamma_blend;
        var ch: usize = 0;
        while (ch < 4) : (ch += 1) {
            const c0: i64 = if (lin and ch < 3) S2L[@intCast(cols[0][ch])] else cols[0][ch];
            const c1: i64 = if (lin and ch < 3) S2L[@intCast(cols[1][ch])] else cols[1][ch];
            const c2: i64 = if (lin and ch < 3) S2L[@intCast(cols[2][ch])] else cols[2][ch];
            const mix_x: i64 = ey[1] * c0 + ey[2] * c1 + ey[0] * c2;
            const mix_y: i64 = ex[1] * c0 + ex[2] * c1 + ex[0] * c2;
            stepcx[ch] = @divTrunc(-(mix_x << 17), area);
            stepcy[ch] = @divTrunc(mix_y << 17, area);
            colx[ch] = @divTrunc((s[1] * c0 + s[2] * c1 + s[0] * c2) << 16, area);
        }
    }

    // depth interpolation (z variant): 12-bit view depth, Q16.16 affine.
    // Same x2-per-pixel factor as the color slopes.
    var z_acc: i64 = 0;
    var z_stepx: i64 = 0;
    var z_stepy: i64 = 0;
    if (zflag) {
        const z0: i64 = @as(i64, (p[0].tok >> 4) & 0xFFF);
        const z1: i64 = @as(i64, (p[1].tok >> 4) & 0xFFF);
        const z2: i64 = @as(i64, (p[2].tok >> 4) & 0xFFF);
        z_stepx = @divTrunc(-(ey[1] * z0 + ey[2] * z1 + ey[0] * z2) << 17, area);
        z_stepy = @divTrunc((ex[1] * z0 + ex[2] * z1 + ex[0] * z2) << 17, area);
        z_acc = @divTrunc((s[1] * z0 + s[2] * z1 + s[0] * z2) << 16, area);
    }

    // apply the top-left biases AFTER the color/depth setup (those need the
    // pure barycentric edge values)
    s[0] += bias[0];
    s[1] += bias[1];
    s[2] += bias[2];

    // ---- scan ----
    var py: i32 = miny;
    while (py <= maxy) : (py += 1) {
        const sx0: [3]i64 = s; // row-start edge values (bias rides along)
        const c0r: [4]i64 = colx; // row-start colors
        const zr: i64 = z_acc;
        var px: i32 = minx;
        while (px <= maxx) : (px += 1) {
            // inside test: branch-free sign check (bias encodes the fill rule)
            const inside = (s[0] > 0 and s[1] > 0 and s[2] > 0);
            if (inside) {
                var zpass = true;
                if (zflag) {
                    // depth REJECT happens before any blending
                    var z = z_acc >> 16;
                    if (z < 0) z = 0;
                    if (z > 4095) z = 4095;
                    const zi: u32 = @intCast(@as(u32, @bitCast(py)) * W + @as(u32, @bitCast(px)));
                    if (z >= zbuf[zi]) {
                        zpass = false;
                    } else if (!additive) {
                        // additive glows test but never write depth
                        zbuf[zi] = @intCast(z);
                    }
                }
                if (zpass) {
                    var col: u32 = undefined;
                    if (flat_col) |fc| {
                        col = fc;
                    } else if (c.gamma_blend) {
                        // RGB accumulated in linear light: encode each
                        // channel, alpha keeps the raw coverage scale
                        col = packRgba(
                            L2S[clampChan16(colx[0])],
                            L2S[clampChan16(colx[1])],
                            L2S[clampChan16(colx[2])],
                            clampChan(colx[3]),
                        );
                    } else {
                        col = packRgba(
                            clampChan(colx[0]),
                            clampChan(colx[1]),
                            clampChan(colx[2]),
                            clampChan(colx[3]),
                        );
                    }
                    if (pattern == 2) {
                        const h = @as(u32, @intCast(px *% 7 + py *% 13)) & 7;
                        if (h < 4) col = if (c.gamma_blend) darkenLin(col, 0x70) else darken(col, 0x70);
                    }
                    putPx(px, py, col, additive);
                }
            }
            // step right (one pixel = 2 units in 2x space); flat tris skip
            // the channel interpolation entirely
            s[0] -= ey[0] * 2;
            s[1] -= ey[1] * 2;
            s[2] -= ey[2] * 2;
            if (flat_col == null) {
                colx[0] += stepcx[0];
                colx[1] += stepcx[1];
                colx[2] += stepcx[2];
                colx[3] += stepcx[3];
            }
            if (zflag) z_acc += z_stepx;
        }
        // step down: restore row-start values, advance one row (= 2 units)
        s = sx0;
        s[0] += ex[0] * 2;
        s[1] += ex[1] * 2;
        s[2] += ex[2] * 2;
        colx = c0r;
        if (flat_col == null) {
            colx[0] += stepcy[0];
            colx[1] += stepcy[1];
            colx[2] += stepcy[2];
            colx[3] += stepcy[3];
        }
        if (zflag) z_acc = zr + z_stepy;
    }
}

inline fn triArea2x(a: Vert, b: Vert, cc: Vert) i64 {
    const ax: i64 = @as(i64, a.x) * 2;
    const ay: i64 = @as(i64, a.y) * 2;
    return (@as(i64, b.x) * 2 - ax) * (@as(i64, cc.y) * 2 - ay) -
        (@as(i64, cc.x) * 2 - ax) * (@as(i64, b.y) * 2 - ay);
}

inline fn unpackRgba(rgba: u32) [4]u32 {
    return .{ (rgba >> 24) & 0xFF, (rgba >> 16) & 0xFF, (rgba >> 8) & 0xFF, rgba & 0xFF };
}

inline fn packRgba(r: u32, g: u32, b: u32, al: u32) u32 {
    return (r << 24) | (g << 16) | (b << 8) | al;
}

inline fn clampChan(v: i64) u32 {
    // Q16.16 accumulator -> 0..255 with round-to-nearest
    const q = (v + (1 << 15)) >> 16;
    if (q < 0) return 0;
    if (q > 255) return 255;
    return @intCast(q);
}

inline fn clampChan16(v: i64) u32 {
    // Q16.16 accumulator -> 0..65535 linear-light scale, round-to-nearest
    const q = (v + (1 << 15)) >> 16;
    if (q < 0) return 0;
    if (q > 65535) return 65535;
    return @intCast(q);
}

inline fn darken(col: u32, amt: u32) u32 {
    const r = ((col >> 24) & 0xFF) * amt >> 8;
    const g = ((col >> 16) & 0xFF) * amt >> 8;
    const b = ((col >> 8) & 0xFF) * amt >> 8;
    const a = col & 0xFF;
    return (r << 24) | (g << 16) | (b << 8) | a;
}

/// v15: same multiply, but in LINEAR light (darkening IS a light-space
/// operation — doing it in sRGB over-darkens). Round-trips through the
/// LUT pair, so flat colors stay exact.
inline fn darkenLin(col: u32, amt: u32) u32 {
    const r: u32 = L2S[(@as(u32, S2L[@intCast((col >> 24) & 0xFF)]) * amt) >> 8];
    const g: u32 = L2S[(@as(u32, S2L[@intCast((col >> 16) & 0xFF)]) * amt) >> 8];
    const b: u32 = L2S[(@as(u32, S2L[@intCast((col >> 8) & 0xFF)]) * amt) >> 8];
    const a = col & 0xFF;
    return (r << 24) | (g << 16) | (b << 8) | a;
}

/// Blending, hot-path ordered by frequency:
///   - a == 255 (opaque sprites, text, 3D tris — the vast majority): exact
///     straight copy, cheaper than the v1 >>8 blend — and identical in
///     BOTH blend modes (the LUT pair round-trips exactly, so there is
///     nothing to convert)
///   - a == 0: nothing to write
///   - otherwise: 256-weight convex blend with round-to-nearest. The blend
///     weight is a/256 instead of a/255 — at most half an output LSB of
///     difference from the exact /255 mix, invisible in 8-bit channels.
///     (Source-over never wraps; additive SATURATES — the v1 @truncate
///     wrapped overlapping glows back to near-black, which 60-star fields
///     could hit and flicker.)
///   - v15 gamma mode (c.gamma_blend): the a!=0/255 and additive paths mix
///     in LINEAR light — decode src+dst, blend, saturate, re-encode. RGB
///     only; alpha is coverage and never converted.
inline fn putPx(x: i32, y: i32, rgba: u32, additive: bool) void {
    if (x < 0 or y < 0 or x >= W or y >= H) return;
    const i = (@as(u32, @intCast(y)) * W + @as(u32, @intCast(x))) * 4;
    const r = (rgba >> 24) & 0xFF;
    const g = (rgba >> 16) & 0xFF;
    const b = (rgba >> 8) & 0xFF;
    const a = rgba & 0xFF;
    if (additive) {
        if (c.gamma_blend) {
            fb[i] = L2S[@min(@as(u32, S2L[fb[i]]) + ((@as(u32, S2L[r]) * a) >> 8), 65535)];
            fb[i + 1] = L2S[@min(@as(u32, S2L[fb[i + 1]]) + ((@as(u32, S2L[g]) * a) >> 8), 65535)];
            fb[i + 2] = L2S[@min(@as(u32, S2L[fb[i + 2]]) + ((@as(u32, S2L[b]) * a) >> 8), 65535)];
        } else {
            fb[i] = @min(@as(u32, fb[i]) + ((r * a) >> 8), 255);
            fb[i + 1] = @min(@as(u32, fb[i + 1]) + ((g * a) >> 8), 255);
            fb[i + 2] = @min(@as(u32, fb[i + 2]) + ((b * a) >> 8), 255);
        }
    } else if (a == 255) {
        fb[i] = @intCast(r);
        fb[i + 1] = @intCast(g);
        fb[i + 2] = @intCast(b);
    } else if (a != 0) {
        if (c.gamma_blend) {
            const na: u32 = 256 - a;
            fb[i] = L2S[(@as(u32, S2L[r]) * a + @as(u32, S2L[fb[i]]) * na + 128) >> 8];
            fb[i + 1] = L2S[(@as(u32, S2L[g]) * a + @as(u32, S2L[fb[i + 1]]) * na + 128) >> 8];
            fb[i + 2] = L2S[(@as(u32, S2L[b]) * a + @as(u32, S2L[fb[i + 2]]) * na + 128) >> 8];
        } else {
            const na: u32 = 256 - a;
            fb[i] = @truncate((r * a + @as(u32, fb[i]) * na + 128) >> 8);
            fb[i + 1] = @truncate((g * a + @as(u32, fb[i + 1]) * na + 128) >> 8);
            fb[i + 2] = @truncate((b * a + @as(u32, fb[i + 2]) * na + 128) >> 8);
        }
    }
}

// ---------------- v12: z-buffer tri variant ----------------
// Same rasterTri core with zmode=true: zflagged tris depth-test per pixel
// (12-bit token depth, now interpolated incrementally); everything else
// paints in stream order (painter's), so HUD/sprites keep their semantics.

fn triZ(a: Vert, b: Vert, cc: Vert) void {
    rasterTri(a, b, cc, true);
}

fn tri(a: Vert, b: Vert, cc: Vert) void {
    rasterTri(a, b, cc, false);
}

// ---------------- PNG writer (zero-dep, zlib stored blocks + CRC32) ----------------

fn crc32(buf: []const u8) u32 {
    var crc: u32 = 0xFFFFFFFF;
    for (buf) |b| {
        crc ^= b;
        var k: u32 = 0;
        while (k < 8) : (k += 1) {
            const mask: u32 = 0 -% (crc & 1); // 0xFFFFFFFF when bit set
            crc = (crc >> 1) ^ (0xEDB88320 & mask);
        }
    }
    return ~crc;
}

fn adler32(buf: []const u8) u32 {
    var a: u32 = 1;
    var b: u32 = 0;
    for (buf) |byte| {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

fn chunk(out: *std.ArrayList(u8), gpa: std.mem.Allocator, tag: *const [4]u8, data: []const u8) !void {
    var lenb: [4]u8 = undefined;
    std.mem.writeInt(u32, &lenb, @intCast(data.len), .big);
    try out.appendSlice(gpa, &lenb);
    try out.appendSlice(gpa, tag);
    try out.appendSlice(gpa, data);
    // CRC over tag+data
    const crc_input = try gpa.alloc(u8, 4 + data.len);
    defer gpa.free(crc_input);
    @memcpy(crc_input[0..4], tag);
    @memcpy(crc_input[4..], data);
    var crcb: [4]u8 = undefined;
    std.mem.writeInt(u32, &crcb, crc32(crc_input), .big);
    try out.appendSlice(gpa, &crcb);
}

/// Write fb as RGBA PNG (stored/uncompressed zlib blocks — simple + valid).
pub fn writePng(gpa: std.mem.Allocator, path: []const u8, io: std.Io) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);

    try out.appendSlice(gpa, &[_]u8{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n' });

    // IHDR
    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], W, .big);
    std.mem.writeInt(u32, ihdr[4..8], H, .big);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 6; // RGBA
    ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    try chunk(&out, gpa, "IHDR", &ihdr);

    // raw scanlines: filter byte 0 + W*4 bytes each
    const stride = W * 4 + 1;
    const raw = try gpa.alloc(u8, stride * H);
    defer gpa.free(raw);
    var y: u32 = 0;
    while (y < H) : (y += 1) {
        raw[y * stride] = 0;
        @memcpy(raw[y * stride + 1 .. (y + 1) * stride], fb[y * W * 4 .. (y + 1) * W * 4]);
    }

    // zlib wrapper: 0x78 0x01 + stored blocks (max 65535 each) + adler32
    var z: std.ArrayList(u8) = .empty;
    defer z.deinit(gpa);
    try z.append(gpa, 0x78);
    try z.append(gpa, 0x01);
    var off: usize = 0;
    while (off < raw.len) {
        const n = @min(raw.len - off, 65535);
        const last: u8 = if (off + n >= raw.len) 1 else 0;
        try z.append(gpa, last);
        var nb: [2]u8 = undefined;
        std.mem.writeInt(u16, &nb, @intCast(n), .little);
        try z.appendSlice(gpa, &nb);
        var nbi: [2]u8 = undefined;
        std.mem.writeInt(u16, &nbi, @intCast(n ^ 0xFFFF), .little);
        try z.appendSlice(gpa, &nbi);
        try z.appendSlice(gpa, raw[off .. off + n]);
        off += n;
    }
    var adb: [4]u8 = undefined;
    std.mem.writeInt(u32, &adb, adler32(raw), .big);
    try z.appendSlice(gpa, &adb);

    try chunk(&out, gpa, "IDAT", z.items);
    try chunk(&out, gpa, "IEND", &[_]u8{});

    // write file
    var dir = try std.Io.Dir.cwd().openDir(io, ".", .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = path, .data = out.items });
}

// ---------------- v15 tests (aggregated by test_raster.zig) ----------------

const testing = std.testing;

test "sRGB LUTs: exact round-trip, monotonicity, endpoints" {
    // the LUT pair is a true inverse: every code survives decode->encode.
    // This is what lets flat colors and the opaque fast path skip
    // conversion entirely in gamma mode.
    var code: u32 = 0;
    while (code < 256) : (code += 1) {
        try testing.expectEqual(@as(u8, @intCast(code)), L2S[S2L[@intCast(code)]]);
    }
    var i: u32 = 1;
    while (i < 256) : (i += 1) try testing.expect(S2L[i] >= S2L[i - 1]);
    i = 1;
    while (i < 65536) : (i += 1) try testing.expect(L2S[i] >= L2S[i - 1]);
    try testing.expectEqual(@as(u16, 0), S2L[0]);
    try testing.expectEqual(@as(u16, 65535), S2L[255]);
    try testing.expectEqual(@as(u8, 0), L2S[0]);
    try testing.expectEqual(@as(u8, 255), L2S[65535]);
}

test "sRGB LUTs: gamma anchors — 50% light is sRGB 188, not 128" {
    // THE gamma fact, pinned: half linear light encodes to 188; the naive
    // sRGB average (128) is only 22% as much light.
    try testing.expectEqual(@as(u8, 188), L2S[32768]);
    // sRGB 128 decodes to ~21.6% light (14143/65535)
    try testing.expect(S2L[128] > 14100 and S2L[128] < 14200);
}

test "putPx: 50% black over white — legacy 128, gamma 188" {
    const i: u32 = (0 * W + 10) * 4;
    c.gamma_blend = false;
    fb[i] = 255;
    fb[i + 1] = 255;
    fb[i + 2] = 255;
    putPx(10, 0, 0x00000080, false);
    try testing.expectEqual(@as(u8, 128), fb[i]);
    try testing.expectEqual(@as(u8, 128), fb[i + 1]);
    try testing.expectEqual(@as(u8, 128), fb[i + 2]);
    // gamma: (0*128 + 65535*128 + 128) >> 8 = 32768 -> L2S = 188
    c.gamma_blend = true;
    fb[i] = 255;
    fb[i + 1] = 255;
    fb[i + 2] = 255;
    putPx(10, 0, 0x00000080, false);
    try testing.expectEqual(@as(u8, 188), fb[i]);
    try testing.expectEqual(@as(u8, 188), fb[i + 1]);
    try testing.expectEqual(@as(u8, 188), fb[i + 2]);
    c.gamma_blend = false;
}

test "putPx: additive — legacy 254, gamma saturates at 255" {
    const i: u32 = (0 * W + 20) * 4;
    c.gamma_blend = false;
    fb[i] = 0;
    fb[i + 1] = 0;
    fb[i + 2] = 0;
    putPx(20, 0, 0xFFFFFF80, true);
    putPx(20, 0, 0xFFFFFF80, true);
    // legacy: (255*128)>>8 = 127, twice -> 254
    try testing.expectEqual(@as(u8, 254), fb[i]);
    c.gamma_blend = true;
    fb[i] = 0;
    fb[i + 1] = 0;
    fb[i + 2] = 0;
    putPx(20, 0, 0xFFFFFF80, true);
    putPx(20, 0, 0xFFFFFF80, true);
    // gamma: 32767 + 32767 linear -> 65534 -> encodes 255
    try testing.expectEqual(@as(u8, 255), fb[i]);
    try testing.expectEqual(@as(u8, 255), fb[i + 1]);
    try testing.expectEqual(@as(u8, 255), fb[i + 2]);
    c.gamma_blend = false;
}

test "putPx: opaque copy needs no conversion in either mode" {
    const i: u32 = (0 * W + 30) * 4;
    c.gamma_blend = true;
    putPx(30, 0, 0x07C863FF, false);
    try testing.expectEqual(@as(u8, 0x07), fb[i]);
    try testing.expectEqual(@as(u8, 0xC8), fb[i + 1]);
    try testing.expectEqual(@as(u8, 0x63), fb[i + 2]);
    c.gamma_blend = false;
}

test "rasterize: gradient midpoint — sRGB lerp 127 vs linear-light lerp 188" {
    const render = @import("render.zig");
    render.streamReset();
    render.appendVert(0, 0, 0, 0, 0x000000FF, 0);
    render.appendVert(511, 0, 0, 0, 0xFFFFFFFF, 0);
    render.appendVert(0, 511, 0, 0, 0xFFFFFFFF, 0);
    const i: u32 = (127 * W + 127) * 4;
    c.gamma_blend = false;
    rasterize();
    const legacy: u32 = fb[i];
    c.gamma_blend = true;
    rasterize();
    const gamma: u32 = fb[i];
    c.gamma_blend = false;
    // sample center (127.5, 127.5) sits at 49.9% of the black->white ramp:
    // the sRGB-space lerp lands on 127, the linear-light lerp on ~188 —
    // the whole visual reason the knob exists
    try testing.expect(legacy >= 126 and legacy <= 128);
    try testing.expect(gamma >= 186 and gamma <= 190);
    try testing.expect(gamma > legacy + 50);
}

test "rasterize: flat colors identical in both blend modes" {
    const render = @import("render.zig");
    render.streamReset();
    render.appendVert(64, 64, 0, 0, 0x336699FF, 0);
    render.appendVert(192, 64, 0, 0, 0x336699FF, 0);
    render.appendVert(64, 192, 0, 0, 0x336699FF, 0);
    c.gamma_blend = false;
    rasterize();
    const h_legacy = std.hash.Wyhash.hash(0, &fb);
    c.gamma_blend = true;
    rasterize();
    const h_gamma = std.hash.Wyhash.hash(0, &fb);
    c.gamma_blend = false;
    // opaque flat tris take the fast path in both modes -> identical bytes
    try testing.expectEqual(h_legacy, h_gamma);
}

test "rasterize: pixel-output-only — c.mem untouched in both modes" {
    const render = @import("render.zig");
    render.streamReset();
    render.appendVert(0, 0, 0, 0, 0x000000FF, 0);
    render.appendVert(300, 0, 0, 0, 0xFFFFFFFF, 0);
    render.appendVert(0, 300, 0, 0, 0x80808080, 0); // gradient + alpha blend
    render.appendVert(400, 400, 0, 0, 0xFFFFFF40, 1); // additive star
    render.appendVert(420, 400, 0, 0, 0xFFFFFF40, 1);
    render.appendVert(410, 420, 0, 0, 0xFFFFFF40, 1);
    const h0 = std.hash.Wyhash.hash(0, &c.mem);
    c.gamma_blend = false;
    rasterize();
    try testing.expectEqual(h0, std.hash.Wyhash.hash(0, &c.mem));
    c.gamma_blend = true;
    rasterize();
    try testing.expectEqual(h0, std.hash.Wyhash.hash(0, &c.mem));
    c.gamma_blend = false;
}
