//! .libyte loader: parses the binary into flat-memory structures and decodes
//! the Huffman bitstream into the instruction array (decode-once; index-based
//! jumps at runtime). Mirrors compiler/src/libyte.rs EXACTLY.

const std = @import("std");
const c = @import("core.zig");
const ent = @import("entity.zig");

/// scratch for entropy-coded text payloads (MAX_TEXTS x (1 + TEXT_STRIDE))
var text_buf: [4608]u8 = undefined;

/// Byte reader with truncation detection. The v1 reader returned 0 for
/// reads past EOF, which let a truncated file decode to garbage with
/// ERR_OK. Now any read that walks off the end sets `truncated`, and the
/// section loops check it — a short file fails loudly as ERR_TRUNCATED.
const Br = struct {
    b: []const u8,
    pos: usize,
    truncated: bool = false,

    fn u8r(r: *Br) u8 {
        if (r.pos >= r.b.len) {
            r.truncated = true;
            return 0;
        }
        const v = r.b[r.pos];
        r.pos += 1;
        return v;
    }
    fn u16r(r: *Br) u16 {
        return @as(u16, r.u8r()) | (@as(u16, r.u8r()) << 8);
    }
    fn u32r(r: *Br) u32 {
        return @as(u32, r.u16r()) | (@as(u32, r.u16r()) << 16);
    }
    fn i32r(r: *Br) i32 { return @bitCast(r.u32r()); }
    fn take(r: *Br, n: usize) []const u8 {
        if (r.pos + n > r.b.len) {
            // clamp AND flag: callers that need the exact length check
            // `truncated` or compare the slice length
            const s = r.b[r.pos..];
            r.pos = r.b.len;
            r.truncated = true;
            return s;
        }
        const s = r.b[r.pos .. r.pos + n];
        r.pos += n;
        return s;
    }
    fn str8(r: *Br) []const u8 {
        const n = r.u8r();
        return r.take(n);
    }
};

/// Load and decode. Returns 0 on success or negative error code.
/// v6 contract: over-limit games fail LOUDLY with a specific ERR_LIMIT_*
/// code (silent clamping was a toy-engine behavior — it corrupted games
/// that outgrew the engine build without telling anyone).
///
/// Transactional (v14 hardening): a failed load leaves the engine counts
/// zeroed, so a host that ignores the error code runs an empty game
/// instead of executing a half-parsed schema. `c.last_error` carries the
/// code either way.
pub fn load(blob: []const u8) i32 {
    const rc = loadInner(blob);
    if (rc != c.ERR_OK) {
        // roll back the section counts a partial parse had published —
        // every runtime accessor gates on these, so the soft-landing
        // contract holds even for hosts that soldier on after failure
        c.n_tfs = 0; c.n_entities = 0; c.n_fns = 0; c.n_tables = 0; c.n_scenes = 0;
        c.n_sprites = 0; c.n_tracks = 0; c.n_sfx = 0; c.n_anims = 0; c.n_texts = 0;
        c.n_keys = 0; c.pool_len = 0; c.instr_count = 0;
        c.n_fsms = 0;
        c.n_arrs = 0;
        c.n_par_groups = 0;
    }
    c.last_error = rc;
    return rc;
}

fn loadInner(blob: []const u8) i32 {
    // reset engine-global state
    c.n_tfs = 0; c.n_entities = 0; c.n_fns = 0; c.n_tables = 0; c.n_scenes = 0;
    c.n_sprites = 0; c.n_tracks = 0; c.n_sfx = 0; c.n_anims = 0; c.n_texts = 0;
    c.n_keys = 0; c.pool_len = 0; c.instr_count = 0;
    c.n_fsms = 0;
    c.n_arrs = 0;
    // v8 regions zero every load (fitted-kind 0 = absent, span 0 = no table)
    @memset(c.mem[c.ADDR_FSM..c.MAP_END], 0);
    // v11: array planes zero every load (fresh-game semantics; entity slots
    // additionally zero their array rows at spawn). NOT inside the FSM range:
    // arrays live after MAP_END and must not perturb legacy memory hashes.
    @memset(c.mem[c.ADDR_GARR..(c.ADDR_EARR + c.EARR_BYTES)], 0);

    var r = Br{ .b = blob, .pos = 0 };
    if (r.take(4).len < 4 or !std.mem.eql(u8, blob[0..4], "LIBY")) return c.ERR_MAGIC;
    const ver = r.u16r();
    // v3 = the optimizer format (v8): FSM bit-plane tables + fitted sfx.
    if (ver != 1 and ver != 2 and ver != 3) return c.ERR_VERSION;
    const flags = r.u16r();
    c.game_w = r.u16r();
    c.game_h = r.u16r();
    c.game_wrap = (flags & 1) != 0;
    c.n_entities = r.u8r();
    const n_fns = r.u16r();
    c.n_fns = n_fns;
    const n_tables = r.u8r();
    c.n_tables = n_tables;
    const n_scenes = r.u8r();
    c.n_scenes = n_scenes;

    // v2 capacity + world header: the game declares the entity slots it
    // needs and (v7) its simulation bounds; the engine build must provide
    // them (ERR_LIMIT_ENTITIES / world defaults to screen otherwise).
    // v1 files carry no capacity: they need the default 512, world = screen.
    var cap_ent: u32 = 512;
    if (ver >= 2) {
        cap_ent = r.u32r();
        const world_packed = r.u32r(); // v7: world_w | world_h << 16 (0 = screen)
        const pw: u32 = world_packed & 0xFFFF;
        const ph: u32 = (world_packed >> 16) & 0xFFFF;
        // world defaults to the screen per axis (old v2 files: reserved=0)
        c.game_world_w = if (pw != 0) pw else c.game_w;
        c.game_world_h = if (ph != 0) ph else c.game_h;
    } else {
        c.game_world_w = c.game_w;
        c.game_world_h = c.game_h;
    }
    if (cap_ent > c.MAX_ENT) return c.ERR_LIMIT_ENTITIES;
    if (c.n_entities > c.MAX_TYPES) return c.ERR_LIMIT_TYPES; // more entity types than this build's profile
    if (n_fns > c.MAX_FNS) return c.ERR_LIMIT_FNS;
    if (n_tables > c.MAX_TABLES) return c.ERR_LIMIT_TABLES;
    if (n_scenes > c.MAX_SCENES) return c.ERR_LIMIT_SCENES;

    // atom name table (skip — debug only)
    const n_atoms = r.u16r();
    var i: u32 = 0;
    while (i < n_atoms) : (i += 1) { _ = r.str8(); }
    if (r.truncated) return c.ERR_TRUNCATED;

    // domain lists (skip — operands embed domain positions)
    var d: u32 = 0;
    while (d < 7) : (d += 1) {
        const n = r.u16r();
        var k: u32 = 0;
        while (k < n) : (k += 1) { _ = r.u16r(); }
    }
    if (r.truncated) return c.ERR_TRUNCATED;

    // texts
    const n_texts = r.u16r();
    if (r.truncated) return c.ERR_TRUNCATED;
    if (n_texts > c.MAX_TEXTS) return c.ERR_LIMIT_TEXTS;
    c.n_texts = n_texts;
    if (flags & 2 != 0) {
        // dictionary synthesis (v8): text payload entropy-coded with a
        // per-game canonical Huffman dictionary shipped inline. The decode
        // LUT builder is IDENTICAL to the opcode one — one tiny decoder
        // serves every packed section; no generic inflate library exists.
        const n_dict = r.u16r();
        var dlens = [_]u8{0} ** 256;
        var dcodes = [_]u16{0} ** 256;
        var k: u32 = 0;
        while (k < n_dict) : (k += 1) {
            const sym = r.u8r();
            const l = r.u8r();
            dlens[sym] = l;
            // canonical code assignment mirrors huffman.canonical_codes:
            // syms arrive sorted by (len, sym); assign sequential codes
        }
        // recompute canonical codes from lengths
        {
            var code: u16 = 0;
            var l: u8 = 1;
            while (l <= 8) : (l += 1) {
                var s: u32 = 0;
                while (s < 256) : (s += 1) {
                    if (dlens[s] == l) {
                        dcodes[s] = code;
                        code += 1;
                    }
                }
                code <<= 1;
            }
        }
        const total = r.u32r();
        const packed_len = r.u32r();
        const packed_bytes = r.take(packed_len);
        if (total > text_buf.len) return c.ERR_HUFF;
        // decode (MSB-first bit peek, 8-bit LUT — same as opcodes)
        var bitpos: u32 = 0;
        var tdec: u32 = 0;
        var tdec_lut = [_]u16{0} ** 256;
        {
            var s: u32 = 0;
            while (s < 256) : (s += 1) {
                const l = dlens[s];
                if (l == 0 or l > 8) continue;
                var base: u32 = 0;
                var b: u32 = 0;
                while (b < l) : (b += 1) {
                    base |= ((@as(u32, dcodes[s]) >> @intCast(l - 1 - b)) & 1) << @intCast(b);
                }
                var j: u32 = 0;
                const span = @as(u32, 1) << @intCast(8 - l);
                while (j < span) : (j += 1) {
                    tdec_lut[base + (j << @intCast(l))] = (@as(u16, @intCast(s)) << 8) | l;
                }
            }
        }
        // a truncated entropy payload would decode to garbage zeros: reject
        // when the packed bytes are shorter than the file promised
        if (packed_bytes.len < packed_len) return c.ERR_TRUNCATED;
        while (tdec < total) : (tdec += 1) {
            var peek: u32 = 0;
            var b: u32 = 0;
            while (b < 8) : (b += 1) {
                const byte = (bitpos + b) >> 3;
                if (byte < packed_bytes.len) {
                    peek |= @as(u32, (packed_bytes[byte] >> @intCast((bitpos + b) & 7)) & 1) << @intCast(b);
                }
            }
            const entry = tdec_lut[peek];
            const oplen: u8 = @intCast(entry & 0xFF);
            if (oplen == 0) return c.ERR_HUFF;
            bitpos += oplen;
            text_buf[tdec] = @intCast(entry >> 8);
        }
        // parse the decoded payload exactly like the raw layout
        var tr: Br = .{ .b = text_buf[0..total], .pos = 0 };
        i = 0;
        while (i < n_texts) : (i += 1) {
            const s = tr.str8();
            const dst = c.ADDR_TEXTS + i * c.TEXT_STRIDE;
            const n = @min(s.len, c.TEXT_STRIDE - 2);
            c.mem[dst] = @intCast(n);
            @memcpy(c.mem[dst + 1 .. dst + 1 + n], s[0..n]);
        }
    } else {
        i = 0;
        while (i < n_texts) : (i += 1) {
            const s = r.str8();
            const dst = c.ADDR_TEXTS + i * c.TEXT_STRIDE;
            const n = @min(s.len, c.TEXT_STRIDE - 2);
            c.mem[dst] = @intCast(n);
            @memcpy(c.mem[dst + 1 .. dst + 1 + n], s[0..n]);
        }
    }

    // entity schemas + integration pairs
    i = 0;
    while (i < c.n_entities) : (i += 1) {
        const rb = r.u16r();
        const cb = r.u16r();
        const ml = r.u16r();
        const n_integ = r.u8r();
        // validate BEFORE writing: a crafted schema row must not reach the
        // entity accessors (rowBytes/coldBytes are derived from these)
        if (rb > 32 or cb > 8) return c.ERR_BAD_HEADER;
        c.wr16(c.ENT_SCHEMA_BASE + i * 6 + 0, rb);
        c.wr16(c.ENT_SCHEMA_BASE + i * 6 + 2, cb);
        c.wr16(c.ENT_SCHEMA_BASE + i * 6 + 4, ml);
        c.wr8(c.INTEG_BASE + i * c.INTEG_STRIDE, @min(n_integ, 8));
        var k: u32 = 0;
        while (k < n_integ and k < 8) : (k += 1) {
            const off = c.INTEG_BASE + i * c.INTEG_STRIDE + 4 + k * 10;
            c.wr16(off, r.u16r());
            c.wr16(off + 2, r.u16r());
            c.wr32(off + 4, @bitCast(r.i32r()));
        }
        while (k < n_integ) : (k += 1) { _ = r.u16r(); _ = r.u16r(); _ = r.i32r(); }
    }
    if (r.truncated) return c.ERR_TRUNCATED;

    // typed fields
    const n_tf = r.u16r();
    if (r.truncated) return c.ERR_TRUNCATED;
    if (n_tf > c.MAX_TF) return c.ERR_LIMIT_TF;
    c.n_tfs = n_tf;
    i = 0;
    while (i < n_tf) : (i += 1) {
        const ent_b = r.u8r();
        const off = r.u16r();
        const coff = r.u16r();
        const bits = r.u8r();
        const vt = r.u8r();
        const cold = r.u8r();
        const default = r.i32r();
        // v14 hardening — the tf registry drives raw bit-offset addressing
        // in entity.zig; a crafted record could reach outside the flat
        // buffer. Validate the geometry BEFORE it lands in TF_BASE:
        //   - ent must be a declared entity type or the globals sentinel
        //   - width must fit an i32 payload (0x80 is the sign flag)
        //   - off+width must land inside the storage the record targets
        const width: u32 = bits & 0x7F;
        if (width == 0 or width > 32) return c.ERR_BAD_HEADER;
        if (ent_b == 0xFF) {
            if (off + width > c.GLOBALS_BYTES * 8) return c.ERR_BAD_HEADER;
        } else {
            if (ent_b >= c.n_entities) return c.ERR_BAD_HEADER;
            // the schema stores ROW BYTES (rb) / COLD BYTES (cb)
            const sb = c.ENT_SCHEMA_BASE + @as(u32, ent_b) * 6;
            const row_bits: u32 = @as(u32, c.rd16(sb)) * 8;
            const cold_bits: u32 = @as(u32, c.rd16(sb + 2)) * 8;
            const storage_bits = if (cold != 0) cold_bits else row_bits;
            if (storage_bits == 0 or off + width > storage_bits) return c.ERR_BAD_HEADER;
        }
        const base = c.TF_BASE + i * c.TF_SIZE;
        c.mem[base] = ent_b;
        c.wr16(base + 2, off);
        c.wr16(base + 2 + 2, coff); // note: unused by runtime (slot-indexed cold)
        c.mem[base + 4] = bits;
        c.mem[base + 5] = vt;
        c.mem[base + 6] = cold;
        c.wr32(base + 8, @bitCast(default));
    }
    if (r.truncated) return c.ERR_TRUNCATED;

    // sprites
    const n_sprites = r.u16r();
    if (n_sprites > c.MAX_SPRITES) return c.ERR_LIMIT_SPRITES;
    c.n_sprites = n_sprites;
    i = 0;
    while (i < n_sprites) : (i += 1) {
        const nv = r.u16r();
        if (nv > c.MAX_SPR_VERTS) return c.ERR_LIMIT_SPR_VERTS;
        var j: u32 = 0;
        const dst = c.ADDR_SPRITES + i * c.SPRITE_STRIDE;
        c.wr16(dst, @intCast(nv));
        while (j < nv) : (j += 1) {
            const x = r.u16r();
            const y = r.u16r();
            const u = r.u16r();
            const v = r.u16r();
            const rgba = r.u32r();
            const tok = r.u16r();
            const vd = dst + 2 + j * 14;
            c.wr16(vd, x);
            c.wr16(vd + 2, y);
            c.wr16(vd + 4, u);
            c.wr16(vd + 6, v);
            c.wr32(vd + 8, rgba);
            c.wr16(vd + 12, tok);
        }
    }

    // music
    const n_music = r.u16r();
    if (r.truncated) return c.ERR_TRUNCATED;
    if (n_music > c.MAX_TRACKS) return c.ERR_LIMIT_MUSIC;
    c.n_tracks = n_music;
    // per-voice stride inside MUSIC_STRIDE: 4B header + 40B per voice
    // (16 steps x 2B + 8B headroom). A crafted nv would write through the
    // NEXT track's record (v14: validate against the stride, loudly).
    const MAX_VOICES: u32 = (c.MUSIC_STRIDE - 4) / 40;
    const MAX_STEPS: u32 = 16;
    i = 0;
    while (i < n_music) : (i += 1) {
        const bpm = r.u16r();
        const nv = r.u8r();
        if (nv > MAX_VOICES) return c.ERR_LIMIT_MUSIC;
        const base = c.ADDR_MUSIC + i * c.MUSIC_STRIDE;
        c.wr16(base, bpm);
        c.mem[base + 2] = nv;
        var v: u32 = 0;
        while (v < nv) : (v += 1) {
            const wave = r.u8r();
            const vol = r.u8r();
            const nsteps = r.u8r();
            if (nsteps > MAX_STEPS) return c.ERR_LIMIT_MUSIC; // steps live in the 40B voice slot
            const voff = base + 4 + v * 40; // 16 steps * 2B + 8 headroom
            c.mem[voff] = wave;
            c.mem[voff + 1] = vol;
            c.mem[voff + 2] = nsteps;
            var s: u32 = 0;
            while (s < nsteps) : (s += 1) {
                c.wr16(voff + 4 + s * 2, r.u16r());
            }
        }
    }
    if (r.truncated) return c.ERR_TRUNCATED;

    // sfx programs (sound domain order)
    const n_sfx = r.u16r();
    if (n_sfx > c.MAX_SFX) return c.ERR_LIMIT_SFX;
    c.n_sfx = n_sfx;
    i = 0;
    while (i < n_sfx) : (i += 1) {
        const base = c.ADDR_SFX + i * 8;
        c.mem[base] = r.u8r();       // wave
        c.wr16(base + 1, r.u16r());  // freq
        const sw = r.u16r();
        c.wr16(base + 3, sw);        // sweep (raw bits; i16)
        c.mem[base + 5] = r.u8r();   // decay
        c.mem[base + 6] = r.u8r();   // vol
        // v8 SUBSYSTEM 4: fitted synthesis programs ride in v3 files as an
        // in-place extension of the base record. Kind byte >= 4 marks them
        // (4 = FM, 5 = additive); the compiler zero-pads nothing — the fit
        // region stays zeroed for legacy wave programs.
        const wave_b = c.mem[base];
        if (ver >= 3 and wave_b >= 4) {
            const fb = c.ADDR_SFX_FIT + i * c.SFX_FIT_STRIDE;
            c.mem[fb] = wave_b;
            const f0 = r.u16r();
            c.wr16(fb + 4, f0);
            if (wave_b == 4) {
                // FM: f0 u16, sweep i16, ratio_q4, index_q4, decay, vol, noise
                const sweep16 = r.u16r();
                c.wr16(fb + 8, sweep16);
                c.mem[fb + 10] = r.u8r(); // ratio_q4
                c.mem[fb + 11] = r.u8r(); // index_q4
                c.mem[fb + 12] = r.u8r(); // decay
                c.mem[fb + 13] = r.u8r(); // vol
                c.mem[fb + 14] = r.u8r(); // noise mix 0..16
            } else {
                // ADD: f0 u16, decay, vol, npart, then 3B per partial.
                // Partials pack 3 bytes each at fb+16 (12B total) — the
                // old pk*4 stride overflowed the 28B record into the next
                // sfx fit slot (partial 4's ratio got clobbered by the
                // neighbor's kind byte; deterministic but WRONG audio).
                c.mem[fb + 12] = r.u8r(); // decay
                c.mem[fb + 13] = r.u8r(); // vol
                const npart = r.u8r();
                // compiler contract: 1..=4 partials (opt/audio.rs truncates);
                // a bigger count would desync the whole section stream
                if (npart < 1 or npart > 4) return c.ERR_BAD_HEADER;
                c.mem[fb + 15] = npart;
                var pk: u32 = 0;
                while (pk < npart) : (pk += 1) {
                    const po = fb + 16 + pk * 3;
                    c.mem[po] = r.u8r();     // ratio_q6
                    c.mem[po + 1] = r.u8r(); // amp 0..16
                    c.mem[po + 2] = r.u8r(); // phase 0..255
                }
            }
        }
    }
    if (r.truncated) return c.ERR_TRUNCATED;

    // anims
    const n_anim = r.u16r();
    if (r.truncated) return c.ERR_TRUNCATED;
    if (n_anim > c.MAX_ANIMS) return c.ERR_LIMIT_ANIMS;
    c.n_anims = n_anim;
    i = 0;
    while (i < n_anim) : (i += 1) {
        const dur = r.u16r();
        const nseg = r.u8r();
        // ANIM_STRIDE = 4 + MAX_SEGS*20: a crafted nseg would write 8x
        // past the record into the text region (v14: loud, like the rest)
        if (nseg > c.MAX_SEGS) return c.ERR_LIMIT_ANIMS;
        const base = c.ADDR_ANIM + i * c.ANIM_STRIDE;
        c.wr16(base, dur);
        c.mem[base + 2] = nseg;
        var s: u32 = 0;
        while (s < nseg) : (s += 1) {
            const t0 = r.u16r();
            const p0 = r.i32r();
            const m0 = r.i32r();
            const p1 = r.i32r();
            const m1 = r.i32r();
            const off = base + 4 + s * 20;
            c.wr16(off, t0);
            c.wr32(off + 2, @bitCast(p0));
            c.wr32(off + 6, @bitCast(m0));
            c.wr32(off + 10, @bitCast(p1));
            c.wr32(off + 14, @bitCast(m1));
        }
    }

    // fn table
    i = 0;
    while (i < n_fns) : (i += 1) {
        const kind = r.u8r();
        const entity = r.u8r();
        const nparams = r.u8r();
        const nlocals = r.u8r();
        const start = r.u32r();
        const len = r.u32r();
        // v14: kind drives fnKind dispatch (0..7); start/len address the
        // decoded instruction array — validate both against MAX_INSTR
        if (kind > 7) return c.ERR_BAD_HEADER;
        if (start > c.MAX_INSTR or len > c.MAX_INSTR or start + len > c.MAX_INSTR)
            return c.ERR_LIMIT_INSTR;
        const base = c.ADDR_FNS + i * c.FN_SIZE;
        c.mem[base] = kind;
        c.mem[base + 1] = entity;
        c.mem[base + 2] = nparams;
        c.mem[base + 3] = nlocals;
        c.wr32(base + 4, start);
        c.wr32(base + 8, len);
    }
    if (r.truncated) return c.ERR_TRUNCATED;

    // tables
    i = 0;
    while (i < n_tables) : (i += 1) {
        const ne = r.u8r();
        const base = c.ADDR_TABLES + i * c.TABLE_STRIDE;
        var k: u32 = 0;
        while (k < ne) : (k += 1) {
            const key = r.u8r();
            const fi = r.u16r();
            if (fi >= n_fns) return c.ERR_BAD_HEADER;
            c.wr16(base + @as(u32, key) * 2, fi);
        }
    }
    // scenes
    i = 0;
    while (i < n_scenes) : (i += 1) {
        const fi = r.u16r();
        if (fi >= n_fns) return c.ERR_BAD_HEADER;
        c.wr16(c.ADDR_SCENES + i * 2, fi);
    }
    if (r.truncated) return c.ERR_TRUNCATED;

    // pool
    const n_pool = r.u16r();
    if (n_pool > c.MAX_POOL) return c.ERR_LIMIT_POOL;
    c.pool_len = n_pool;
    i = 0;
    while (i < c.pool_len) : (i += 1) {
        c.wr32(c.ADDR_POOL + i * 4, @bitCast(r.i32r()));
    }

    // huffman table -> decode LUT
    var lens = [_]u8{0} ** 256;
    const n_huff = r.u16r();
    i = 0;
    while (i < n_huff) : (i += 1) {
        const sym = r.u8r();
        const len = r.u8r();
        // v14 hardening: a crafted len byte > 8 previously reached the
        // canonical shift math below as a negative/overflowing shift and
        // trapped in ReleaseSafe. Zero it here — the symbol decodes as
        // "invalid code" (oplen 0) exactly like an unused symbol.
        lens[sym] = if (len <= 8) len else 0;
    }
    // canonical codes (mirror of Rust canonical_codes)
    var codes = [_]u16{0} ** 256;
    {
        // symbols sorted by (len, sym)
        var order: [256]u8 = undefined;
        var n_syms: u32 = 0;
        var s: u32 = 0;
        while (s < 256) : (s += 1) {
            if (lens[s] > 0) { order[n_syms] = @intCast(s); n_syms += 1; }
        }
        // insertion sort by (len, sym) — syms already ascending within equal len
        var a: u32 = 1;
        while (a < n_syms) : (a += 1) {
            var ka: u32 = a;
            while (ka > 0 and lens[order[ka]] < lens[order[ka - 1]]) {
                const tmp = order[ka];
                order[ka] = order[ka - 1];
                order[ka - 1] = tmp;
                ka -= 1;
            }
        }
        var code: u32 = 0;
        var prev_len: u8 = 0;
        var first = true;
        for (order[0..n_syms]) |sym| {
            const l = lens[sym];
            if (first) {
                first = false;
            } else {
                code = (code + 1) << @intCast(l - prev_len);
            }
            // pathological (over-subscribed) tables can walk code past u16;
            // legit trees never do — reject loudly instead of trapping
            if (code > 65535) return c.ERR_HUFF;
            codes[sym] = @intCast(code);
            prev_len = l;
        }
    }
    // decode table (256 entries)
    var decode = [_]u16{0} ** 256; // packed: sym << 8 | len (0 len = invalid)
    {
        var s: u32 = 0;
        while (s < 256) : (s += 1) {
            const l = lens[s];
            if (l == 0 or l > 8) continue;
            var base: u32 = 0;
            var b: u32 = 0;
            while (b < l) : (b += 1) {
                base |= ((codes[s] >> @intCast(l - 1 - b)) & 1) << @intCast(b);
            }
            var j: u32 = 0;
            const span = @as(u32, 1) << @intCast(8 - l);
            while (j < span) : (j += 1) {
                decode[base + (j << @intCast(l))] = (@as(u16, @intCast(s)) << 8) | l;
            }
        }
    }

    // code bitstream
    const n_instr = r.u32r();
    _ = r.u32r(); // total bits (informational)
    const byte_len = r.u32r();
    const code_bytes = r.take(byte_len);
    if (n_instr > c.MAX_INSTR) return c.ERR_LIMIT_INSTR;
    // v14: a short entropy section used to decode to zero-opcode garbage
    // with ERR_OK; now the truncation is loud
    if (code_bytes.len < byte_len) return c.ERR_TRUNCATED;

    var bitpos: u32 = 0;
    var extra_off: u32 = 0;
    var ii: u32 = 0;
    while (ii < n_instr) : (ii += 1) {
        // decode opcode via LUT
        var peek: u32 = 0;
        var b: u32 = 0;
        while (b < 8) : (b += 1) {
            const byte = (bitpos + b) >> 3;
            if (byte < code_bytes.len) {
                peek |= @as(u32, (code_bytes[byte] >> @intCast((bitpos + b) & 7)) & 1) << @intCast(b);
            }
        }
        const entry = decode[peek];
        const oplen: u8 = @intCast(entry & 0xFF);
        if (oplen == 0) return c.ERR_HUFF;
        bitpos += oplen;
        const op: u8 = @intCast(entry >> 8);

        var a: i32 = 0;
        var bb: i32 = 0;
        var cc: i32 = 0;
        // operand signatures (mirror compiler codegen.sig)
        const OP_PUSH_S8: u8 = 0x01;
        const OP_PUSH_POOL: u8 = 0x02;
        const OP_LD_LOCAL: u8 = 0x03; const OP_ST_LOCAL: u8 = 0x04;
        const OP_KILL: u8 = 0x32;
        const OP_LD_GLBL: u8 = 0x05; const OP_ST_GLBL: u8 = 0x06;
        const OP_LD_ENT: u8 = 0x07; const OP_ST_ENT: u8 = 0x08;
        const OP_JMP: u8 = 0x2C; const OP_JZ: u8 = 0x2D; const OP_JNZ: u8 = 0x2E;
        const OP_FOR_BGN: u8 = 0x2F; const OP_FOR_ADV: u8 = 0x30;
        const OP_SPAWN: u8 = 0x31;
        const OP_CALL_FN: u8 = 0x3D; const OP_CALL_TBL: u8 = 0x3A;
        const OP_COUNT: u8 = 0x3B;
        const OP_SHAKE: u8 = 0x40;
        const OP_MUSIC_VOL: u8 = 0x42; // v7: u8 master volume operand
        // v8 optimizer opcodes (mirror compiler codegen.sig)
        const OP_SWEPT: u8 = 0x49;         // (None, None, None)
        const OP_COLLIDE_MASK: u8 = 0x4A;  // (U8, PoolU16, PoolU16) + extra
        const OP_FOR_MASK_BGN: u8 = 0x4B;  // (U8, U8, S16)
        const OP_FOR_MASK_ADV: u8 = 0x4C;  // (S16, None, None)
        const OP_FSM_NEXT: u8 = 0x4D;      // (U16, None, None)
        const OP_SEL: u8 = 0x4E;           // (None, None, None) — branchless cond-move
        // v11 arrays + 3D (mirror compiler codegen.sig)
        const OP_LD_GARR: u8 = 0x4F;       // (U8, None, None)
        const OP_ST_GARR: u8 = 0x50;       // (U8, None, None)
        const OP_LD_EARR: u8 = 0x51;       // (U8, U8, None)
        const OP_ST_EARR: u8 = 0x52;       // (U8, U8, None)
        const OP_DUP: u8 = 0x53;           // (None, None, None)
        const OP_CAM3: u8 = 0x54;          // (None, None, None)
        const OP_PROJ3: u8 = 0x55;         // (None, None, None)
        const OP_PROJ_X: u8 = 0x56;        // (None, None, None)
        const OP_PROJ_Y: u8 = 0x57;        // (None, None, None)
        const OP_PROJ_OK: u8 = 0x58;       // (None, None, None)
        const OP_DRAW3D: u8 = 0x59;        // (U8, None, None)
        // v12 articulation (mirror compiler codegen.sig)
        const OP_QUAT_AA: u8 = 0x5A;       // (None, None, None) — stack args only
        const OP_Q_MUL: u8 = 0x5B;         // (None, None, None)
        const OP_M4_QT: u8 = 0x5C;         // (None, None, None)
        const OP_M4_MUL: u8 = 0x5D;        // (None, None, None)
        const OP_SKIN3: u8 = 0x5E;         // (None, None, None)
        const OP_DRAW3DI: u8 = 0x5F;       // (U8, U8, None) — vert + idx arr ids
        const domops = [_]u8{ 0x33, 0x34, 0x36, 0x37, 0x38, 0x2A, 0x2B }; // SFX MUSIC GOTO DRAW DRAW_TEXT ANIM KEY

        switch (op) {
            OP_PUSH_S8 => a = rdS(code_bytes, &bitpos, 8),
            OP_PUSH_POOL => a = @bitCast(rdU(code_bytes, &bitpos, 16)),
            OP_LD_LOCAL, OP_ST_LOCAL, OP_COUNT => a = @bitCast(rdU(code_bytes, &bitpos, 8)),
            OP_KILL => {
                a = @bitCast(rdU(code_bytes, &bitpos, 8));
                bb = @bitCast(rdU(code_bytes, &bitpos, 8));
            },
            OP_LD_GLBL, OP_ST_GLBL => a = @bitCast(rdU(code_bytes, &bitpos, 16)),
            OP_LD_ENT, OP_ST_ENT => {
                a = @bitCast(rdU(code_bytes, &bitpos, 8));
                bb = @bitCast(rdU(code_bytes, &bitpos, 16));
            },
            OP_JMP, OP_JZ, OP_JNZ, OP_FOR_ADV => a = rdS(code_bytes, &bitpos, 16),
            OP_FOR_BGN => {
                a = @bitCast(rdU(code_bytes, &bitpos, 8));
                bb = @bitCast(rdU(code_bytes, &bitpos, 8));
                cc = rdS(code_bytes, &bitpos, 16);
            },
            OP_CALL_FN => {
                a = @bitCast(rdU(code_bytes, &bitpos, 16));
                bb = @bitCast(rdU(code_bytes, &bitpos, 8));
            },
            OP_CALL_TBL => {
                a = @bitCast(rdU(code_bytes, &bitpos, 8));
                bb = @bitCast(rdU(code_bytes, &bitpos, 8));
            },
            OP_SPAWN => {
                a = @bitCast(rdU(code_bytes, &bitpos, 8));
                bb = @bitCast(rdU(code_bytes, &bitpos, 8));
            },
            OP_SHAKE, OP_MUSIC_VOL => a = @bitCast(rdU(code_bytes, &bitpos, 8)),
            OP_SWEPT => {},
            OP_COLLIDE_MASK => {
                a = @bitCast(rdU(code_bytes, &bitpos, 8));
                bb = @bitCast(rdU(code_bytes, &bitpos, 16)); // pool hw
                cc = @bitCast(rdU(code_bytes, &bitpos, 16)); // pool hh
            },
            OP_FOR_MASK_BGN => {
                a = @bitCast(rdU(code_bytes, &bitpos, 8));
                bb = @bitCast(rdU(code_bytes, &bitpos, 8));
                cc = rdS(code_bytes, &bitpos, 16);
            },
            OP_FOR_MASK_ADV => a = rdS(code_bytes, &bitpos, 16),
            OP_SEL => {}, // (None, None, None): pure stack op, no operands
            OP_FSM_NEXT => a = @bitCast(rdU(code_bytes, &bitpos, 16)),
            // v11: arrays + 3D
            OP_LD_GARR, OP_ST_GARR, OP_DRAW3D => a = @bitCast(rdU(code_bytes, &bitpos, 8)),
            OP_LD_EARR, OP_ST_EARR, OP_DRAW3DI => {
                a = @bitCast(rdU(code_bytes, &bitpos, 8));
                bb = @bitCast(rdU(code_bytes, &bitpos, 8));
            },
            OP_DUP, OP_CAM3, OP_PROJ3, OP_PROJ_X, OP_PROJ_Y, OP_PROJ_OK => {},
            OP_QUAT_AA, OP_Q_MUL, OP_M4_QT, OP_M4_MUL, OP_SKIN3 => {},
            else => {
                if (std.mem.indexOfScalar(u8, &domops, op) != null) {
                    a = @bitCast(rdU(code_bytes, &bitpos, 16));
                }
            },
        }

        // write decoded instruction
        const base = c.ADDR_CODE + ii * c.INSTR_SIZE;
        c.mem[base] = op;
        c.wr32(base + 4, @bitCast(a));
        c.wr32(base + 8, @bitCast(bb));
        c.wr32(base + 12, @bitCast(cc));
        // SPAWN extra tf list; COLLIDE_MASK extra [x_tf, y_tf] (v8 subsystem 3)
        if (op == OP_SPAWN) {
            const n: u32 = @intCast(bb);
            // extra-pool overflow guard (v6): previously unbounded — a
            // pathological program could write past EXTRA_BYTES
            if (extra_off + n > c.EXTRA_BYTES / 2) return c.ERR_LIMIT_EXTRA;
            var j: u32 = 0;
            const ex = c.ADDR_EXTRA + extra_off * 2;
            while (j < n) : (j += 1) {
                c.wr16(ex + j * 2, @intCast(rdU(code_bytes, &bitpos, 16)));
            }
            c.wr32(base + 16, extra_off);
            extra_off += n;
        } else if (op == OP_COLLIDE_MASK) {
            if (extra_off + 2 > c.EXTRA_BYTES / 2) return c.ERR_LIMIT_EXTRA;
            const ex = c.ADDR_EXTRA + extra_off * 2;
            c.wr16(ex, @intCast(rdU(code_bytes, &bitpos, 16)));     // x_tf
            c.wr16(ex + 2, @intCast(rdU(code_bytes, &bitpos, 16))); // y_tf
            c.wr32(base + 16, extra_off);
            extra_off += 2;
        } else {
            c.wr32(base + 16, 0);
        }
    }
    c.instr_count = n_instr;

    // v3 trailer: FSM bit-plane transition tables (v8 subsystem 5)
    if (ver >= 3) {
        const n_fsm = r.u8r();
        if (n_fsm > c.MAX_FSM) return c.ERR_LIMIT_FSM;
        c.n_fsms = n_fsm;
        var fi: u32 = 0;
        while (fi < n_fsm) : (fi += 1) {
            _ = r.u16r(); // atom (the table is addressed by FSM-domain position)
            const nstates = r.u8r();
            const span = r.u8r();
            const dst = c.ADDR_FSM + fi * c.FSM_STRIDE;
            c.mem[dst + 2] = @min(nstates, c.FSM_MAX_STATES);
            c.mem[dst + 3] = span;
            const tlen: u32 = @as(u32, @min(nstates, c.FSM_MAX_STATES)) * @as(u32, span);
            if (tlen > c.FSM_MAX_STATES * c.FSM_MAX_STATES) return c.ERR_BAD_HEADER;
            const tbytes = r.take(tlen);
            // v14: a truncated trailer made tbytes[0..tlen] an out-of-bounds
            // slice — the single clearest crafted-file crash. Take() clamps;
            // verify the promised length actually arrived.
            if (tbytes.len < tlen) return c.ERR_TRUNCATED;
            @memcpy(c.mem[dst + 4 .. dst + 4 + tlen], tbytes[0..tlen]);
        }
    }

    // v10 trailer: compile-proven parallel-system groups (flags bit 2).
    // Zeroed on every load BEFORE parsing so a re-load with a plain binary
    // never inherits the previous game's groups (hot-swap safety).
    c.n_par_groups = 0;
    c.par_groups = [_]u16{0} ** (c.MAX_PAR_GROUPS * c.PAR_MAX_MEMBERS);
    c.par_group_len = [_]u8{0} ** c.MAX_PAR_GROUPS;
    if (flags & 4 != 0) {
        const n_groups = r.u8r();
        if (n_groups > c.MAX_PAR_GROUPS) return c.ERR_BAD_HEADER;
        var g: u32 = 0;
        while (g < n_groups) : (g += 1) {
            const n_members = r.u8r();
            if (n_members == 0 or n_members > c.PAR_MAX_MEMBERS) return c.ERR_BAD_HEADER;
            c.par_group_len[g] = n_members;
            var m: u32 = 0;
            while (m < n_members) : (m += 1) {
                const fi = r.u16r();
                if (fi >= n_fns) return c.ERR_BAD_HEADER;
                c.par_groups[g * c.PAR_MAX_MEMBERS + m] = fi;
            }
        }
        c.n_par_groups = n_groups;
    }

    // v11 trailer: arrays schema (flags bit 3). Bases are derived in id order:
    // global arrays pack the GARR plane; entity arrays pack max_live(type)
    // rows of cap*4 into the EARR plane. Overflow = LOUD load error.
    c.arr_ent = [_]u8{0xFF} ** c.MAX_ARRS;
    c.arr_cap = [_]u16{0} ** c.MAX_ARRS;
    c.arr_base = [_]u32{0} ** c.MAX_ARRS;
    if (flags & 8 != 0) {
        const n = r.u8r();
        if (n > c.MAX_ARRS) return c.ERR_LIMIT_ARRAYS;
        var gcur: u32 = c.ADDR_GARR;
        var ecur: u32 = c.ADDR_EARR;
        var k: u32 = 0;
        while (k < n) : (k += 1) {
            const kind = r.u8r();
            const et = r.u8r();
            const cap = r.u16r();
            if (cap == 0) return c.ERR_BAD_HEADER;
            c.arr_ent[k] = et;
            c.arr_cap[k] = cap;
            if (kind == 0) {
                c.arr_base[k] = gcur;
                gcur +%= @as(u32, cap) * 4;
                if (gcur > c.ADDR_GARR + c.GARR_BYTES) return c.ERR_LIMIT_ARRAYS;
            } else {
                if (et >= c.n_entities) return c.ERR_BAD_HEADER;
                c.arr_base[k] = ecur;
                ecur +%= c.maxLive(et) * @as(u32, cap) * 4;
                if (ecur > c.ADDR_EARR + c.EARR_BYTES) return c.ERR_LIMIT_ARRAYS;
            }
        }
        c.n_arrs = n;
    }

    // runtime state init
    c.wr32(c.ADDR_INPUT, 0);
    c.wr32(c.ADDR_FRAME, 0);
    c.wr32(c.ADDR_RAND, 0x12345678);
    c.wr32(c.ADDR_SCENE, 0);
    c.wr32(c.ADDR_SHAKE, 0);
    // v7: camera starts at the world origin (viewport shows the top-left
    // corner of the world; world == screen games get offset 0 — v1 behavior)
    c.wr32(c.ADDR_CAM_X, 0);
    c.wr32(c.ADDR_CAM_Y, 0);
    // NOTE: sram_dirty is intentionally NOT cleared here — a save() that
    // happened right before a re-init must still reach the host's persist
    // poll (lila_sram_dirty reads-and-clears; only the host consumes it).

    // zero entity storage + defaults
    ent.resetAll();
    ent.globalsReset();

    // final truncation sweep: any section read that walked off the end set
    // the flag; a file that ends mid-section must not load as "valid but
    // empty" (silent data loss). Valid files end with pos == len, flag clear.
    if (r.truncated) return c.ERR_TRUNCATED;
    return c.ERR_OK;
}


// ---------------- bit-level readers (file scope; Zig has no closures) ----------------

fn rdBit(cb: []const u8, bp: *u32) u1 {
    const byte = bp.* >> 3;
    if (byte >= cb.len) {
        bp.* += 1;
        return 0;
    }
    const v: u1 = @intCast((cb[byte] >> @intCast(bp.* & 7)) & 1);
    bp.* += 1;
    return v;
}

fn rdU(cb: []const u8, bp: *u32, bits: u5) u32 {
    var v: u32 = 0;
    var i: u32 = 0;
    while (i < bits) : (i += 1) {
        v |= @as(u32, rdBit(cb, bp)) << @intCast(i);
    }
    return v;
}

fn rdS(cb: []const u8, bp: *u32, bits: u5) i32 {
    const v = rdU(cb, bp, bits);
    if (bits < 32 and (v >> @intCast(bits - 1)) & 1 == 1) {
        return @bitCast(v | (@as(u32, 0xFFFFFFFF) << @intCast(bits)));
    }
    return @bitCast(v);
}
