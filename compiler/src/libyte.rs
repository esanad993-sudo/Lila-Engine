//! .libyte binary format emitter (and the Huffman bitstream packer).
//! Layout documented in docs/SPEC.md — Zig loader (runtime/src/loader.zig)
//! and disasm.rs must parse in EXACTLY this order.

use crate::checker::{Ctx, FnKind, GLOBAL_ENT};
use crate::codegen::{self, Code, OpK};
use crate::huffman;

pub struct BitWriter {
    pub bytes: Vec<u8>,
    bit: u32, // 0..7
}

impl BitWriter {
    pub fn new() -> Self { BitWriter { bytes: Vec::new(), bit: 0 } }
    pub fn write_bit(&mut self, b: bool) {
        if self.bit == 0 { self.bytes.push(0); }
        if b {
            let last = self.bytes.len() - 1;
            self.bytes[last] |= 1u8 << self.bit;
        }
        self.bit = (self.bit + 1) % 8;
    }
    /// Write `len` bits of `code`, MSB of the code FIRST (sequential order).
    pub fn write_code(&mut self, code: u16, len: u8) {
        for i in (0..len).rev() {
            self.write_bit(((code >> i) & 1) != 0);
        }
    }
    pub fn write_u(&mut self, v: u32, bits: u8) {
        for i in 0..bits {
            self.write_bit(((v >> i) & 1) != 0);
        }
    }
    pub fn write_s(&mut self, v: i32, bits: u8) {
        let u = v as u32;
        self.write_u(u, bits);
    }
    pub fn total_bits(&self) -> u32 {
        (self.bytes.len() as u32).saturating_sub(1) * 8 + self.bit
    }
}

struct Bw {
    buf: Vec<u8>,
}

impl Bw {
    fn new() -> Self { Bw { buf: Vec::new() } }
    fn u8(&mut self, v: u8) { self.buf.push(v); }
    fn u16(&mut self, v: u16) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn u32(&mut self, v: u32) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn i16(&mut self, v: i16) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn i32(&mut self, v: i32) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn bytes(&mut self, v: &[u8]) { self.buf.extend_from_slice(v); }
}

pub struct EmitResult {
    pub bytes: Vec<u8>,
    pub code_uncompressed: usize, // bytes if 1 byte per opcode + operands
    pub code_packed: usize,       // packed bitstream bytes
    pub text_uncompressed: usize, // raw text payload bytes (0 if not packed)
    pub text_packed: usize,       // entropy-coded payload bytes
}

/// v3 = the OPTIMIZER format: FSM bit-plane tables (subsystem 5) and/or
/// fitted sfx programs (subsystem 4). A v3 header ALWAYS carries the v2
/// capacity/world block (linear parsing); v1/v2 files are byte-identical to
/// the pre-optimizer compiler (legacy contract).
pub fn emitted_version(ctx: &Ctx, fsm_tables: &[Vec<u8>], has_parallel: bool) -> u16 {
    let has_fitted = ctx.sfx.iter().any(|p| p.kind >= 4);
    // v11: arrays force v3 (their schema table rides the v3 trailer)
    if !fsm_tables.is_empty() || has_fitted || has_parallel || !ctx.arrays.is_empty() {
        3
    } else {
        legacy_version(ctx)
    }
}

fn legacy_version(ctx: &Ctx) -> u16 {
    let has_v2 = ctx.game.capacity.is_some()
        || ctx.game.world_w.is_some() || ctx.game.world_h.is_some();
    if has_v2 { 2 } else { 1 }
}

pub fn emit(ctx: &Ctx, code: &Code, fsm_tables: &[Vec<u8>], par_groups: &[Vec<u16>]) -> EmitResult {
    let mut w = Bw::new();

    // ---- header ----
    // v1: fixed limits (512 entity slots implied).
    // v2: game-declared capacity + world dims — emitted ONLY when the game
    // declares one of them, so default games stay byte-identical to v1 files.
    // The v2 "reserved" u32 now carries the world: (world_w | world_h << 16)
    // in effective pixels (world defaults to the screen — packed value then
    // equals width|height<<16, never 0 for sane games; old v2 files with
    // reserved=0 fall back to world == screen in the loader).
    // v3: optimizer format — ALWAYS includes the v2 block, then (after the
    // code section) the FSM bit-plane tables; fitted sfx extend their base
    // records in place. See emitted_version().
    let cap = ctx.game.capacity;
    let world_w = ctx.game.world_w.unwrap_or(ctx.game.width);
    let world_h = ctx.game.world_h.unwrap_or(ctx.game.height);
    let ver: u16 = emitted_version(ctx, fsm_tables, !par_groups.is_empty());
    w.bytes(b"LIBY");
    w.u16(ver);
    let mut flags = 0u16;
    if ctx.game.wrap { flags |= 1; }
    // bit 1: texts shipped entropy-coded with a per-game canonical Huffman
    // dictionary (dictionary synthesis; loader inflates with the same
    // canonical decoder it uses for opcodes)
    let pack_texts = ctx.game.optimize && !ctx.texts.is_empty();
    if pack_texts { flags |= 2; }
    // bit 2: v10 parallel-systems trailer present (subsystem 10 — compile-
    // proven disjoint update groups the runtime may execute concurrently)
    if !par_groups.is_empty() { flags |= 4; }
    // bit 3: v11 arrays schema table present (fixed-capacity arrays with
    // their kind/entity/capacity geometry)
    if !ctx.arrays.is_empty() { flags |= 8; }
    w.u16(flags);
    w.u16(ctx.game.width as u16);
    w.u16(ctx.game.height as u16);
    w.u8(ctx.entities.len() as u8);
    w.u16(ctx.fns.len() as u16);
    w.u8(ctx.tables.len() as u8);
    w.u8(ctx.scenes.len() as u8);
    match ver {
        1 => {}
        2 => {
            w.u32(cap.unwrap_or(512)); // capacity: entity slots per type
            w.u32(world_w | (world_h << 16)); // world dims (v7; was reserved=0)
        }
        _ => {
            // v3: the v2 block is ALWAYS present (defaults fill in) so the
            // loader parse stays linear: ver>=2 -> read cap+world.
            w.u32(cap.unwrap_or(512));
            w.u32(world_w | (world_h << 16));
        }
    }

    // ---- atom name table (disasm/debug only; runtime ignores) ----
    w.u16(ctx.atom_names.len() as u16);
    for n in &ctx.atom_names {
        w.u8(n.len() as u8);
        w.bytes(n.as_bytes());
    }

    // ---- domain atom-id lists (disasm/debug; runtime ignores) ----
    let text_atom_ids: Vec<u16> = ctx.texts.iter().map(|(a, _)| *a).collect();
    let doms: [&Vec<u16>; 7] = [
        &ctx.sound, &ctx.keys, &text_atom_ids,
        &ctx.scene_atom_ids, &ctx.sprite_atoms, &ctx.music_atoms, &ctx.anim_atoms,
    ];
    for d in doms {
        w.u16(d.len() as u16);
        for a in d.iter() { w.u16(*a); }
    }

    // ---- texts (runtime) ----
    // SUBSYSTEM 9 — DICTIONARY SYNTHESIS (information theory): the text
    // payload's byte entropy is swept at BUILD time and a canonical Huffman
    // dictionary UNIQUE to this game ships inline with the payload. The
    // runtime's shared ~40-line canonical decoder (the same one that
    // inflates opcodes) is the only decompressor in the engine — there is no
    // generic inflate/LZ4/Zstd library anywhere in the footprint.
    w.u16(ctx.texts.len() as u16);
    let (mut text_raw, mut text_packed) = (0usize, 0usize);
    if pack_texts {
        // payload = the raw layout itself (u8 len + bytes per text) so the
        // loader's post-decode parse is IDENTICAL in both modes
        let mut payload: Vec<u8> = Vec::new();
        for (_, s) in &ctx.texts {
            payload.push(s.len() as u8);
            payload.extend_from_slice(s.as_bytes());
        }
        text_raw = payload.len();
        let mut tfreq = [0u32; 256];
        for b in &payload { tfreq[*b as usize] += 1; }
        let ttable = huffman::build(&tfreq);
        let tcodes = huffman::canonical_codes(&ttable);
        let tnsyms = ttable.lens.iter().filter(|&&l| l > 0).count();
        w.u16(tnsyms as u16);
        for s in 0..256 {
            if ttable.lens[s] > 0 { w.u8(s as u8); w.u8(ttable.lens[s]); }
        }
        w.u32(payload.len() as u32);
        let mut tbw = BitWriter::new();
        for b in &payload {
            let (cv, l) = tcodes[*b as usize];
            if l == 0 { tbw.write_u(*b as u32, 8); } else { tbw.write_code(cv, l); }
        }
        text_packed = tbw.bytes.len();
        w.u32(text_packed as u32);
        w.bytes(&tbw.bytes);
    } else {
        for (_, s) in &ctx.texts {
            w.u8(s.len() as u8);
            w.bytes(s.as_bytes());
        }
    }

    // ---- entity schemas + auto-integration pairs ----
    // Per entity: layout + (pos_tf, vel_tf, mask) pairs for x/vx, y/vy.
    // mask = (dim<<8)-1 when game.wrap (bitwise AND wrap), else 0 (no mask).
    // This is the manual's "metaprogrammed context-driven physics": the
    // compiler emits bespoke integration, the runtime executes 2 ops per axis.
    for (ent_pos, e) in ctx.entities.iter().enumerate() {
        w.u16(e.row_bytes);
        w.u16(e.cold_bytes);
        w.u16(e.max_live as u16);
        // integration pairs: x/vx and y/vy when both exist as dense fixed fields
        let field_tf = |name: &str| -> Option<u16> {
            let fi = e.fields.iter().position(|f| {
                f.name == name && f.vt == crate::checker::VT::Fixed && !f.cold
            })?;
            ctx.typed_fields.iter().position(|t| {
                t.ent as usize == ent_pos && t.field as usize == fi
            }).map(|p| p as u16)
        };
        let mut pairs: Vec<(u16, u16, i32)> = Vec::new();
        if let (Some(xt), Some(vxt)) = (field_tf("x"), field_tf("vx")) {
            // v7: wrap domain is the WORLD (defaults to the screen)
            let mask = if ctx.game.wrap { ((world_w as i64) << 8) - 1 } else { 0 };
            pairs.push((xt, vxt, mask as i32));
        }
        if let (Some(yt), Some(vyt)) = (field_tf("y"), field_tf("vy")) {
            let mask = if ctx.game.wrap { ((world_h as i64) << 8) - 1 } else { 0 };
            pairs.push((yt, vyt, mask as i32));
        }
        w.u8(pairs.len() as u8);
        for (p, v, m) in pairs {
            w.u16(p);
            w.u16(v);
            w.i32(m);
        }
    }

    // ---- typed-field registry ----
    w.u16(ctx.typed_fields.len() as u16);
    for tf in &ctx.typed_fields {
        w.u8(tf.ent);
        w.u16(tf.off_bits);
        w.u16(tf.c_off_bits);
        w.u8(tf.bits | if tf.signed { 0x80 } else { 0 });
        let vt = match tf.vt {
            crate::checker::VT::Fixed => 0u8,
            crate::checker::VT::Int => 1u8,
            crate::checker::VT::Bool => 2u8,
            _ => 3u8, // ang
        };
        w.u8(vt);
        w.u8(tf.cold as u8);
        let default = if tf.ent == GLOBAL_ENT {
            ctx.globals[tf.field as usize].default
        } else {
            ctx.entities[tf.ent as usize].fields[tf.field as usize].default
        };
        w.i32(default);
    }

    // ---- sprites (14B vertices) ----
    w.u16(ctx.sprites.len() as u16);
    for s in &ctx.sprites {
        w.u16(s.verts.len() as u16);
        for v in &s.verts {
            w.i16(v.x);
            w.i16(v.y);
            w.u16(v.u);
            w.u16(v.v);
            w.u32(v.rgba);
            w.u16(v.tok);
        }
    }

    // ---- music ----
    w.u16(ctx.music.len() as u16);
    for m in &ctx.music {
        w.u16(m.bpm as u16);
        w.u8(m.voices.len() as u8);
        for v in &m.voices {
            w.u8(v.wave);
            w.u8(v.vol);
            w.u8(v.rows.len() as u8);
            for r in &v.rows { w.u16(*r); }
        }
    }

    // ---- sfx programs, in SOUND DOMAIN ORDER ----
    // (runtime indexes programs by sound-atom domain position)
    // v8 SUBSYSTEM 4: kind >= 4 = build-time fitted voice. The BASE record
    // keeps the legacy 8-byte shape (kind in the wave byte, zeros elsewhere);
    // v3 files EXTEND it in place with the fitted parameters, so a v3 loader
    // branches once per record and v1/v2 files stay byte-identical.
    w.u16(ctx.sound.len() as u16);
    for atom in &ctx.sound {
        let prog = ctx.sfx.iter().find(|p| &p.atom == atom);
        match prog {
            Some(p) => {
                w.u8(p.wave);
                w.u16(p.freq.max(0).min(65535) as u16);
                w.i16(p.sweep.clamp(-32768, 32767) as i16);
                w.u8(p.decay.clamp(1, 255) as u8);
                w.u8(p.vol as u8);
                if ver >= 3 {
                    match &p.fitted {
                        Some(crate::opt::audio::FittedModel::Fm(m)) => {
                            // FM: out(t) = env(t)·sin(φc + I·sin(φm)) [+ noise]
                            w.u16(m.f0);
                            w.i16(m.sweep);
                            w.u8(m.ratio_q4);
                            w.u8(m.index_q4);
                            w.u8(m.decay);
                            w.u8(m.vol);
                            w.u8(m.noise);
                        }
                        Some(crate::opt::audio::FittedModel::Add(m)) => {
                            // ADD: out(t) = env(t)·Σ aᵢ·sin(2π rᵢ f0 t + φᵢ)
                            w.u16(m.f0);
                            w.u8(m.decay);
                            w.u8(m.vol);
                            w.u8(m.partials.len() as u8);
                            for (r, a, ph) in &m.partials {
                                w.u8(*r); w.u8(*a); w.u8(*ph);
                            }
                        }
                        None => {}
                    }
                }
            }
            None => {
                // undeclared program: silence
                w.u8(2); w.u16(0); w.i16(0); w.u8(1); w.u8(0);
            }
        }
    }

    // ---- anims ----
    w.u16(ctx.anims.len() as u16);
    for a in &ctx.anims {
        w.u16(a.duration);
        w.u8(a.segs.len() as u8);
        for s in &a.segs {
            w.u16(s.t0);
            w.i32(s.p0);
            w.i32(s.m0);
            w.i32(s.p1);
            w.i32(s.m1);
        }
    }

    // ---- fn table ----
    for f in &code.fns {
        let kind = match f.kind {
            FnKind::Init => 0u8,
            FnKind::Update0 => 1,
            FnKind::UpdateEnt => 2,
            FnKind::Draw0 => 3,
            FnKind::DrawEnt => 4,
            FnKind::Helper => 5,
            FnKind::Scene => 6,
            FnKind::Table => 7,
        };
        w.u8(kind);
        w.u8(f.entity);
        w.u8(f.nparams);
        w.u8(f.nlocals);
        w.u32(f.start);
        w.u32(f.len);
    }

    // ---- dispatch tables ----
    for t in &ctx.tables {
        w.u8(t.entries.len() as u8);
        for (k, fi) in &t.entries {
            w.u8(*k);
            w.u16(*fi);
        }
    }
    // scenes (scene-domain order -> fn index)
    for (_, fi) in &ctx.scenes {
        w.u16(*fi);
    }

    // ---- pool ----
    w.u16(code.pool.len() as u16);
    for v in &code.pool { w.i32(*v); }

    // ---- huffman table ----
    let mut freqs = [0u32; 256];
    for ins in &code.instrs { freqs[ins.op as usize] += 1; }
    let table = huffman::build(&freqs);
    let codes = huffman::canonical_codes(&table);
    let nsyms = table.lens.iter().filter(|&&l| l > 0).count();
    w.u16(nsyms as u16);
    for s in 0..256 {
        if table.lens[s] > 0 {
            w.u8(s as u8);
            w.u8(table.lens[s]);
        }
    }

    // ---- code bitstream ----
    let mut bw = BitWriter::new();
    let mut uncompressed = 0usize;
    for (i, ins) in code.instrs.iter().enumerate() {
        let (code_val, len) = codes[ins.op as usize];
        if len == 0 {
            // opcode never in freq (shouldn't happen) — give it 8 bits escape
            bw.write_u(ins.op as u32, 8);
            uncompressed += 1;
        } else {
            bw.write_code(code_val, len);
            uncompressed += 1;
        }
        let (ka, kb, kc) = codegen::sig(ins.op);
        // jump-class operands are ABSOLUTE instr indices -> convert to relative
        let jump_a = matches!(ins.op,
            codegen::OP_JMP | codegen::OP_JZ | codegen::OP_JNZ
            | codegen::OP_FOR_ADV | codegen::OP_FOR_MASK_ADV);
        let jump_c = ins.op == codegen::OP_FOR_BGN || ins.op == codegen::OP_FOR_MASK_BGN;
        for (k, v) in [(ka, ins.a), (kb, ins.b), (kc, ins.c)] {
            match k {
                OpK::None => {}
                OpK::U8 => { bw.write_u(v as u32, 8); uncompressed += 1; }
                OpK::U16 => { bw.write_u(v as u32, 16); uncompressed += 2; }
                OpK::S16 => {
                    let rel = if jump_a {
                        v - (i as i32 + 1)
                    } else if jump_c {
                        v - (i as i32 + 1)
                    } else { v };
                    bw.write_s(rel, 16); uncompressed += 2;
                }
                OpK::S8 => { bw.write_s(v, 8); uncompressed += 1; }
                OpK::PoolU16 => { bw.write_u(v as u32, 16); uncompressed += 2; }
            }
        }
        // SPAWN extra tf list; COLLIDE_MASK extra [x_tf, y_tf] (subsystem 3)
        if ins.op == codegen::OP_SPAWN || ins.op == codegen::OP_COLLIDE_MASK {
            for tf in &ins.extra {
                bw.write_u(*tf as u32, 16);
                uncompressed += 2;
            }
        }
    }

    w.u32(code.instrs.len() as u32);
    w.u32(bw.total_bits());
    w.u32(bw.bytes.len() as u32);
    w.bytes(&bw.bytes);

    // ---- v3: FSM bit-plane transition tables (subsystem 5) ----
    // T[state * span + events] = next state; span = 2^max_guards (<= 2^4).
    // The runtime consults the table with ONE shift+mask (OP_FSM_NEXT);
    // guard priority was precomputed here by opt::fsm::build_table.
    if ver >= 3 {
        w.u8(fsm_tables.len() as u8);
        for (f, t) in ctx.fsms.iter().zip(fsm_tables) {
            let max_g = f.guards.iter().map(|g| g.len()).max().unwrap_or(0);
            let span: u8 = if max_g == 0 { 1 } else { 1u8 << max_g };
            w.u16(f.atom);
            w.u8(f.states.len() as u8);
            w.u8(span);
            w.bytes(t);
        }
    }

    // ---- v10: parallel-system groups (subsystem 10) ----
    // Groups of UpdateEnt fn indices the compiler PROVED mutually data-
    // disjoint and side-effect free. The runtime executes each group's
    // members on concurrent workers — no scheduler, no locks, no speculation:
    // the proof is the whole mechanism, and it was free (build-time).
    if !par_groups.is_empty() && ver >= 3 {
        w.u8(par_groups.len() as u8);
        for g in par_groups {
            w.u8(g.len() as u8);
            for &fi in g {
                w.u16(fi);
            }
        }
    }

    // ---- v11: arrays schema table (flags bit 3) ----
    // One descriptor per array, in id order (globals first, then per-entity
    // in entity order). The runtime derives the flat-memory plane layout
    // from these: global arrays pack a 4B/cell region; entity arrays pack
    // max_live(type) rows of cap*4B. Sizes are compile-time constants, so
    // the loader can reject a game whose arrays exceed the build profile
    // LOUDLY (ERR_LIMIT_ARRAYS) instead of clipping silently.
    if !ctx.arrays.is_empty() && ver >= 3 {
        w.u8(ctx.arrays.len() as u8);
        for a in &ctx.arrays {
            w.u8(if a.ent == GLOBAL_ENT { 0 } else { 1 }); // kind
            w.u8(a.ent);
            w.u16(a.cap as u16);
        }
    }

    EmitResult {
        bytes: w.buf,
        code_uncompressed: uncompressed,
        code_packed: bw.bytes.len(),
        text_uncompressed: text_raw,
        text_packed,
    }
}
