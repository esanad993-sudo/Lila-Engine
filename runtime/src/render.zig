//! LILA render stream (manual section 4):
//! - 14-byte quantized vertices: i16 pos (screen px), u16 uv, u32 rgba, u16 token
//! - draws transform sprite verts CPU-side in Q24.8, quantize to i16 pixels
//! - token: bit0 additive blend, bits1-2 pattern (0 flat, 2 noise)
//! - comptime 4x5 vector font for draw_text / draw_num

const std = @import("std");
const c = @import("core.zig");

pub fn streamReset() void {
    c.wr32(c.ADDR_STREAM, 0);
}

pub fn streamCount() u32 {
    return c.rd32(c.ADDR_STREAM);
}

pub inline fn appendVert(x: i32, y: i32, u: u32, v: u32, rgba: u32, tok: u32) void {
    const n = c.rd32(c.ADDR_STREAM);
    if (n >= c.MAX_VERTS) {
        // vertex budget exceeded: drop, but COUNT it — the host surfaces
        // stream_drop_total so budgets get raised instead of pixels vanishing
        c.stream_drop_total +%= 1;
        return;
    }
    const base = c.ADDR_STREAM + 4 + n * 14;
    c.wr16(base, @bitCast(@as(u16, @truncate(@as(u32, @bitCast(x))))));
    c.wr16(base + 2, @bitCast(@as(u16, @truncate(@as(u32, @bitCast(y))))));
    c.wr16(base + 4, @truncate(u));
    c.wr16(base + 6, @truncate(v));
    c.wr32(base + 8, rgba);
    c.wr16(base + 12, @truncate(tok));
    c.wr32(c.ADDR_STREAM, n + 1);
}

inline fn mul256(a: i32, b: i32) i32 {
    // (a*b)>>8 in i64
    return @intCast((@as(i64, a) * @as(i64, b)) >> 8);
}

inline fn chanMul(x: u32, y: u32) u32 {
    return @min((x * y) >> 8, 255);
}

inline fn modColor(va: u32, ca: u32) u32 {
    // per-channel va*ca>>8 (white modulate = identity)
    return (chanMul((va >> 24) & 0xFF, (ca >> 24) & 0xFF) << 24) |
        (chanMul((va >> 16) & 0xFF, (ca >> 16) & 0xFF) << 16) |
        (chanMul((va >> 8) & 0xFF, (ca >> 8) & 0xFF) << 8) |
        chanMul(va & 0xFF, ca & 0xFF);
}

/// draw(#sprite, x, y, rot, scale, rgba) — VM stack args.
/// v7: x/y are WORLD coords; the viewport offset (camera) is subtracted
/// here, so the vertex stream stays screen-space and every host (GL, GL1,
/// native rasterizer) is camera-agnostic. Camera 0 = v1 behavior.
pub fn drawSprite(spr: u32, x: i32, y: i32, rot: i32, scale: i32, rgba: u32) void {
    if (spr >= c.n_sprites) return;
    const base = c.ADDR_SPRITES + spr * c.SPRITE_STRIDE;
    const nv = c.rd16(base);
    const s = c.sinA(rot);
    const co = c.cosA(rot);
    const so = shakeOffset(); // world layer only — HUD text + starfield stay put
    const cam = c.camPx(); // world-space camera center -> screen pixels
    var i: u32 = 0;
    while (i < nv) : (i += 1) {
        const vd = base + 2 + i * 14;
        const px: i32 = @as(i16, @bitCast(c.rd16(vd)));
        const py: i32 = @as(i16, @bitCast(c.rd16(vd + 2)));
        const u: u32 = c.rd16(vd + 4);
        const v: u32 = c.rd16(vd + 6);
        const vrgba = c.rd32(vd + 8);
        const tok = c.rd16(vd + 12);
        // rotate (screen convention: rot 0 = up, clockwise) + scale
        const pxq = px *% 256;
        const pyq = py *% 256;
        var rx = mul256(co, pxq) -% mul256(s, pyq);
        var ry = mul256(s, pxq) +% mul256(co, pyq);
        rx = mul256(rx, scale);
        ry = mul256(ry, scale);
        const sx = ((x +% rx) >> 8) -% cam.x +% so.dx;
        const sy = ((y +% ry) >> 8) -% cam.y +% so.dy;
        appendVert(sx, sy, u, v, modColor(vrgba, rgba), tok);
    }
}

// ---------------- screen shake (world layer) ----------------

/// deterministic 32-bit hash — consumes no rand state (game stays reproducible)
inline fn h32(x: u32) u32 {
    var v = x;
    v ^= v >> 16; v *%= 0x7feb352d;
    v ^= v >> 15; v *%= 0x846ca68b;
    v ^= v >> 16;
    return v;
}

pub fn shakeOffset() struct { dx: i32, dy: i32 } {
    const amp = c.rd32(c.ADDR_SHAKE) & 0xFF;
    if (amp == 0) return .{ .dx = 0, .dy = 0 };
    const f = c.rd32(c.ADDR_FRAME);
    const span: u32 = 2 * amp + 1;
    const dx: i32 = @intCast(h32(f *% 0x9E3779B1) % span);
    const dy: i32 = @intCast(h32(f *% 0x85EBCA6B +% 1) % span);
    return .{ .dx = dx - @as(i32, @intCast(amp)), .dy = dy - @as(i32, @intCast(amp)) };
}

// ---------------- engine starfield layer (background) ----------------
// Three parallax layers, hashed positions, drifting down at Q24.8 speeds,
// additive blend, per-star twinkle. Zero game code — pure engine service.
// v7: each layer also scrolls AGAINST the camera at a per-layer depth
// factor (far 1/8, mid 1/4, near 1/2) — the free parallax proof that the
// world moves. Camera 0 reproduces the v1 starfield exactly.

const StarLayer = struct { n: u32, speed: u32, col: u32, alpha: u32, size: i32, par: u5 };

const STAR_LAYERS = [_]StarLayer{
    .{ .n = 26, .speed = 64, .col = 0x3A4A5E, .alpha = 0x90, .size = 2, .par = 3 }, // far: cam/8
    .{ .n = 20, .speed = 128, .col = 0x5E7891, .alpha = 0xA8, .size = 2, .par = 2 }, // mid: cam/4
    .{ .n = 14, .speed = 256, .col = 0x9DB8D9, .alpha = 0xC0, .size = 3, .par = 1 }, // near: cam/2
};

pub fn emitStarfield() void {
    const f = c.rd32(c.ADDR_FRAME);
    const w: u32 = if (c.game_w > 0) c.game_w else 512;
    const h: u32 = if (c.game_h > 0) c.game_h else 512;
    // camera Q24.8 (clamped >= 0 by OP_CAMERA; mask defensively to unsigned)
    const camx: u32 = c.rd32(c.ADDR_CAM_X) & 0x7FFFFFFF;
    const camy: u32 = c.rd32(c.ADDR_CAM_Y) & 0x7FFFFFFF;
    var layer: u32 = 0;
    while (layer < STAR_LAYERS.len) : (layer += 1) {
        const L = STAR_LAYERS[layer];
        // Q24.8 scroll offset -> integer px, wrapped
        const off: u32 = ((f *% L.speed) >> 8) % h;
        // parallax: layer-depth fraction of the camera, in screen px
        const parx: u32 = (camx >> L.par) >> 8;
        const pary: u32 = (camy >> L.par) >> 8;
        var i: u32 = 0;
        while (i < L.n) : (i += 1) {
            const hx = h32(i *% 0x9E37 +% layer *% 0x85EB +% 0x2545);
            const hy = h32(hx ^ 0x1234);
            // stars keep their downward ambient drift (+off) and scroll
            // opposite the camera (-parallax); camera 0 = the v1 field
            const x: i32 = @intCast((hx % w + w - parx % w) % w);
            const y: i32 = @intCast((hy % h + off + h - pary % h) % h);
            // twinkle: brightness 0.55..1.0, phase-hashed per star & slow time
            const tw = h32(i ^ (f >> 4) *% 0x2722 +% layer);
            const b: u32 = 140 + (tw % 116); // 140..255 -> Q8 brightness
            var col = L.col;
            // ~1/16 stars are warm-tinted for variety
            if ((hx >> 8) & 15 == 0) col = 0xD9B08C;
            const r = chanMul((col >> 16) & 0xFF, b);
            const g = chanMul((col >> 8) & 0xFF, b);
            const bch = chanMul(col & 0xFF, b);
            const rgba = (r << 24) | (g << 16) | (bch << 8) | L.alpha;
            const sz = L.size;
            // one quad (2 tris), additive token bit0
            appendVert(x, y, 0, 0, rgba, 1);
            appendVert(x + sz, y, 0, 0, rgba, 1);
            appendVert(x + sz, y + sz, 0, 0, rgba, 1);
            appendVert(x, y, 0, 0, rgba, 1);
            appendVert(x + sz, y + sz, 0, 0, rgba, 1);
            appendVert(x, y + sz, 0, 0, rgba, 1);
        }
    }
}

// ---------------- comptime 4x5 font (each glyph 20 bits) ----------------

const Glyph = struct { ch: u8, bits: u32 };

const GLYPHS = [_]Glyph{
    .{ .ch = 'A', .bits = 0b0110_1001_1111_1001_1001 },
    .{ .ch = 'B', .bits = 0b1110_1001_1110_1001_1110 },
    .{ .ch = 'C', .bits = 0b0111_1000_1000_1000_0111 },
    .{ .ch = 'D', .bits = 0b1110_1001_1001_1001_1110 },
    .{ .ch = 'E', .bits = 0b1111_1000_1110_1000_1111 },
    .{ .ch = 'F', .bits = 0b1111_1000_1110_1000_1000 },
    .{ .ch = 'G', .bits = 0b0111_1000_1011_1001_0111 },
    .{ .ch = 'H', .bits = 0b1001_1001_1111_1001_1001 },
    .{ .ch = 'I', .bits = 0b1110_0100_0100_0100_1110 },
    .{ .ch = 'J', .bits = 0b0011_0001_0001_1001_0110 },
    .{ .ch = 'K', .bits = 0b1001_1010_1100_1010_1001 },
    .{ .ch = 'L', .bits = 0b1000_1000_1000_1000_1111 },
    .{ .ch = 'M', .bits = 0b1001_1111_1111_1001_1001 },
    .{ .ch = 'N', .bits = 0b1001_1101_1011_1001_1001 },
    .{ .ch = 'O', .bits = 0b0110_1001_1001_1001_0110 },
    .{ .ch = 'P', .bits = 0b1110_1001_1110_1000_1000 },
    .{ .ch = 'Q', .bits = 0b0110_1001_1001_1010_0101 },
    .{ .ch = 'R', .bits = 0b1110_1001_1110_1010_1001 },
    .{ .ch = 'S', .bits = 0b0111_1000_0110_0001_1110 },
    .{ .ch = 'T', .bits = 0b1110_0100_0100_0100_0100 },
    .{ .ch = 'U', .bits = 0b1001_1001_1001_1001_0110 },
    .{ .ch = 'V', .bits = 0b1001_1001_1001_0110_0110 },
    .{ .ch = 'W', .bits = 0b1001_1001_1111_1111_1001 },
    .{ .ch = 'X', .bits = 0b1001_1001_0110_1001_1001 },
    .{ .ch = 'Y', .bits = 0b1001_1001_0110_0100_0100 },
    .{ .ch = 'Z', .bits = 0b1111_0001_0110_1000_1111 },
    .{ .ch = '0', .bits = 0b0110_1001_1011_1001_0110 },
    .{ .ch = '1', .bits = 0b0100_1100_0100_0100_1110 },
    .{ .ch = '2', .bits = 0b1110_0001_0110_1000_1111 },
    .{ .ch = '3', .bits = 0b1110_0001_0110_0001_1110 },
    .{ .ch = '4', .bits = 0b1001_1001_1111_0001_0001 },
    .{ .ch = '5', .bits = 0b1111_1000_1110_0001_1110 },
    .{ .ch = '6', .bits = 0b0111_1000_1110_1001_0110 },
    .{ .ch = '7', .bits = 0b1111_0001_0010_0100_0100 },
    .{ .ch = '8', .bits = 0b0110_1001_0110_1001_0110 },
    .{ .ch = '9', .bits = 0b0110_1001_0111_0001_1110 },
    .{ .ch = ':', .bits = 0b0000_0100_0000_0100_0000 },
    .{ .ch = '.', .bits = 0b0000_0000_0000_0000_0100 },
    .{ .ch = '!', .bits = 0b0100_0100_0100_0000_0100 },
    .{ .ch = '-', .bits = 0b0000_0000_1110_0000_0000 },
    .{ .ch = '/', .bits = 0b0001_0010_0100_1000_0000 },
};

fn glyphOf(ch: u8) u32 {
    if (ch == ' ') return 0;
    for (GLYPHS) |g| {
        if (g.ch == ch) return g.bits;
    }
    return 0;
}

const FONT_PX: i32 = 2; // screen pixels per font pixel
const CHAR_ADV: i32 = 10; // 4*2 + 2 spacing

/// Emit one font pixel as a 2x2 quad (2 triangles) at screen pos.
fn fontQuad(gx: i32, gy: i32, rgba: u32) void {
    const x0 = gx;
    const y0 = gy;
    const x1 = gx + FONT_PX;
    const y1 = gy + FONT_PX;
    // two triangles: (x0,y0)(x1,y0)(x1,y1) and (x0,y0)(x1,y1)(x0,y1)
    appendVert(x0, y0, 0, 0, rgba, 0);
    appendVert(x1, y0, 0, 0, rgba, 0);
    appendVert(x1, y1, 0, 0, rgba, 0);
    appendVert(x0, y0, 0, 0, rgba, 0);
    appendVert(x1, y1, 0, 0, rgba, 0);
    appendVert(x0, y1, 0, 0, rgba, 0);
}

fn drawString(s: []const u8, x: i32, y: i32, rgba: u32) void {
    var cx = x;
    for (s) |ch| {
        const bits = glyphOf(ch);
        if (bits != 0) {
            var row: u32 = 0;
            while (row < 5) : (row += 1) {
                var col: u32 = 0;
                while (col < 4) : (col += 1) {
                    const bit = (bits >> @intCast((4 - row) * 4 + (3 - col))) & 1;
                    if (bit != 0) {
                        fontQuad(cx + @as(i32, @intCast(col)) * FONT_PX, y + @as(i32, @intCast(row)) * FONT_PX, rgba);
                    }
                }
            }
        }
        cx += CHAR_ADV;
    }
}

pub fn drawText(text_atom: u32, x: i32, y: i32, rgba: u32) void {
    if (text_atom >= c.n_texts) return;
    const base = c.ADDR_TEXTS + text_atom * c.TEXT_STRIDE;
    const n = c.mem[base];
    drawString(c.mem[base + 1 .. base + 1 + n], x >> 8, y >> 8, rgba);
}

pub fn drawNum(val: i32, x: i32, y: i32, rgba: u32) void {
    var buf: [16]u8 = undefined;
    const neg = val < 0;
    var v: u32 = if (neg) @as(u32, @intCast(-(val + 1))) + 1 else @as(u32, @intCast(val)); // INT_MIN safe
    var i: usize = buf.len;
    if (v == 0) {
        i -= 1;
        buf[i] = '0';
    }
    while (v != 0) {
        i -= 1;
        buf[i] = @intCast('0' + @as(u8, @intCast(v % 10)));
        v /= 10;
    }
    if (neg) {
        if (i > 0) {
            i -= 1;
            buf[i] = '-';
        }
    }
    drawString(buf[i..], x >> 8, y >> 8, rgba);
}
