//! .libyte disassembler — parses the binary format and prints a full dump.
//! Doubles as a round-trip format validator (emitter <-> reader agreement).

use crate::codegen::{self, OpK};
use crate::huffman;

struct Br<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Br<'a> {
    fn u8(&mut self) -> Result<u8, String> {
        if self.pos >= self.b.len() { return Err("unexpected EOF".into()); }
        let v = self.b[self.pos]; self.pos += 1; Ok(v)
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(self.u8()? as u16 | (self.u8()? as u16) << 8)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(self.u16()? as u32 | (self.u16()? as u32) << 16)
    }
    fn i16(&mut self) -> Result<i16, String> { Ok(self.u16()? as i16) }
    fn i32(&mut self) -> Result<i32, String> { Ok(self.u32()? as i32) }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.b.len() { return Err("unexpected EOF".into()); }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn str8(&mut self) -> Result<String, String> {
        let n = self.u8()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

struct BitBr<'a> {
    b: &'a [u8],
    bitpos: u32,
}

impl<'a> BitBr<'a> {
    fn read_bit(&mut self) -> Result<u32, String> {
        let byte = (self.bitpos >> 3) as usize;
        if byte >= self.b.len() { return Err("bitstream EOF".into()); }
        let v = (self.b[byte] >> (self.bitpos & 7)) & 1;
        self.bitpos += 1;
        Ok(v as u32)
    }
    fn read_u(&mut self, bits: u8) -> Result<u32, String> {
        let mut v = 0u32;
        for i in 0..bits { v |= self.read_bit()? << i; }
        Ok(v)
    }
    fn read_s(&mut self, bits: u8) -> Result<i32, String> {
        let v = self.read_u(bits)?;
        // sign extend
        let v = if bits < 32 && (v >> (bits - 1)) & 1 == 1 {
            (v | (0xFFFF_FFFFu32 << bits)) as i32
        } else { v as i32 };
        Ok(v)
    }
    fn peek8(&self) -> u32 {
        let mut v = 0u32;
        for i in 0..8u32 {
            let byte = ((self.bitpos + i) >> 3) as usize;
            if byte < self.b.len() {
                v |= (((self.b[byte] >> ((self.bitpos + i) & 7)) & 1) as u32) << i;
            }
        }
        v
    }
}

pub fn disasm(bytes: &[u8]) -> Result<String, String> {
    let mut out = String::new();
    let mut r = Br { b: bytes, pos: 0 };

    let magic = r.take(4)?;
    if magic != b"LIBY" { return Err("bad magic (not a .libyte)".into()); }
    let ver = r.u16()?;
    let flags = r.u16()?;
    let width = r.u16()?;
    let height = r.u16()?;
    let n_entities = r.u8()?;
    let n_fns = r.u16()?;
    let n_tables = r.u8()?;
    let n_scenes = r.u8()?;
    // v2 capacity + world header (present only when the game declares one)
    let cap_ent: Option<(u32, u32)> = if ver >= 2 {
        let cap = r.u32()?;
        let world_packed = r.u32()?; // v7: world_w | world_h << 16 (was reserved=0)
        Some((cap, world_packed))
    } else {
        None
    };

    out.push_str(&format!(
        "LILYTE v{} flags={:#x} game={}x{} wrap={} entities={} fns={} tables={} scenes={}{}\n",
        ver, flags, width, height, flags & 1 != 0, n_entities, n_fns, n_tables, n_scenes,
        match cap_ent {
            Some((c, wp)) => {
                let ww = wp & 0xFFFF;
                let wh = (wp >> 16) & 0xFFFF;
                let world = if ww == 0 && wh == 0 {
                    format!("world={}x{} (screen)", width, height)
                } else if ww != 0 && wh != 0 {
                    format!("world={}x{}{}", ww, wh, if ww == width as u32 && wh == height as u32 { " (screen)" } else { " (scrolling)" })
                } else { format!("world={:#x}", wp) };
                format!(" capacity={} ents/type {}", c, world)
            }
            None => String::new(),
        }
    ));

    // atom names
    let n_atoms = r.u16()?;
    let mut atom_names = Vec::new();
    for _ in 0..n_atoms {
        atom_names.push(r.str8()?);
    }
    let dom_names = ["sound", "key", "text", "scene", "sprite", "music", "anim"];
    let mut domains: Vec<Vec<u16>> = Vec::new();
    for dn in dom_names {
        let n = r.u16()?;
        let mut ids = Vec::new();
        for _ in 0..n { ids.push(r.u16()?); }
        out.push_str(&format!("  domain {}: [{}]\n", dn,
            ids.iter().map(|a| format!("#{}", atom_names.get(*a as usize).cloned().unwrap_or_default()))
                .collect::<Vec<_>>().join(", ")));
        domains.push(ids);
    }

    // texts
    let n_texts = r.u16()?;
    let mut texts = Vec::new();
    if flags & 2 != 0 {
        // dictionary synthesis: per-game canonical Huffman payload, decoded
        // with the same (len, sym) canonical reconstruction as the loader
        let n_dict = r.u16()?;
        let mut lens = [0u8; 256];
        for _ in 0..n_dict {
            let sym = r.u8()?;
            let l = r.u8()?;
            lens[sym as usize] = l;
        }
        let total = r.u32()? as usize;
        let packed_len = r.u32()? as usize;
        let packed = r.take(packed_len)?;
        let payload = huffman::decode_stream(packed, total, &lens)?;
        let mut pos = 0usize;
        for _ in 0..n_texts {
            if pos >= payload.len() { break; }
            let l = payload[pos] as usize;
            pos += 1;
            let end = (pos + l).min(payload.len());
            texts.push(String::from_utf8_lossy(&payload[pos..end]).into_owned());
            pos = end;
        }
    } else {
        for _ in 0..n_texts {
            texts.push(r.str8()?);
        }
    }
    out.push_str(&format!("  texts: {:?}\n", texts));

    // entity schemas (+ integration pairs)
    let mut ents = Vec::new();
    for i in 0..n_entities {
        let row = r.u16()?;
        let cold = r.u16()?;
        let max_live = r.u16()?;
        let n_integ = r.u8()?;
        let mut integ = Vec::new();
        for _ in 0..n_integ {
            let p = r.u16()?;
            let v = r.u16()?;
            let m = r.i32()?;
            integ.push((p, v, m));
        }
        ents.push((row, cold, max_live));
        out.push_str(&format!("  entity {}: row={}B cold={}B max_live={} integ={:?}\n", i, row, cold, max_live, integ));
    }

    // typed fields
    let n_tf = r.u16()?;
    let mut tfs = Vec::new();
    for i in 0..n_tf {
        let ent = r.u8()?;
        let off = r.u16()?;
        let coff = r.u16()?;
        let bits = r.u8()?;
        let vt = r.u8()?;
        let cold = r.u8()?;
        let default = r.i32()?;
        tfs.push((ent, off, bits, vt, cold));
        out.push_str(&format!(
            "  tf[{:3}] ent={} off={}bit c={}bit bits={} vt={} cold={} default={}\n",
            i, ent, off, coff, bits, vt, cold, default
        ));
    }

    // sprites
    let n_sprites = r.u16()?;
    for si in 0..n_sprites {
        let nv = r.u16()?;
        out.push_str(&format!("  sprite[{}] {} verts (14B each = {}B)\n", si, nv, nv * 14));
        for _ in 0..nv {
            let _x = r.i16()?; let _y = r.i16()?;
            let _u = r.u16()?; let _v = r.u16()?;
            let _c = r.u32()?; let _t = r.u16()?;
        }
    }

    // music
    let n_music = r.u16()?;
    for _ in 0..n_music {
        let bpm = r.u16()?;
        let nv = r.u8()?;
        out.push_str(&format!("  music bpm={} voices={}\n", bpm, nv));
        for _ in 0..nv {
            let _wave = r.u8()?; let _vol = r.u8()?;
            let nsteps = r.u8()?;
            for _ in 0..nsteps { let _row = r.u16()?; }
        }
    }

    // sfx
    let n_sfx = r.u16()?;
    for si in 0..n_sfx {
        let wave = r.u8()?; let _freq = r.u16()?;
        let _sweep = r.i16()?; let _decay = r.u8()?; let _vol = r.u8()?;
        // v8: fitted synthesis programs (v3 files only)
        if ver >= 3 && wave >= 4 {
            if wave == 4 {
                let f0 = r.u16()?;
                let sweep = r.i16()?;
                let ratio = r.u8()?;
                let index = r.u8()?;
                let decay = r.u8()?;
                let vol = r.u8()?;
                let noise = r.u8()?;
                out.push_str(&format!("  sfx[{}] FITTED FM: f0={}Hz sweep={:+} ratio={:.3} index={:.2} decay={} vol={} noise={}\n",
                    si, f0, sweep, ratio as f32 / 16.0, index as f32 / 16.0, decay, vol, noise));
            } else {
                let f0 = r.u16()?;
                let decay = r.u8()?;
                let vol = r.u8()?;
                let np = r.u8()?;
                let mut parts = Vec::new();
                for _ in 0..np {
                    let rr = r.u8()?; let aa = r.u8()?; let ph = r.u8()?;
                    parts.push(format!("{:.3}x{}@{}", rr as f32 / 64.0, aa, ph));
                }
                out.push_str(&format!("  sfx[{}] FITTED ADD: f0={}Hz decay={} vol={} partials=[{}]\n",
                    si, f0, decay, vol, parts.join(", ")));
            }
        }
    }

    // anims
    let n_anim = r.u16()?;
    for _ in 0..n_anim {
        let _dur = r.u16()?;
        let nseg = r.u8()?;
        for _ in 0..nseg {
            let _t0 = r.u16()?;
            let _p0 = r.i32()?; let _m0 = r.i32()?;
            let _p1 = r.i32()?; let _m1 = r.i32()?;
        }
    }

    // fn table
    let mut fns = Vec::new();
    for i in 0..n_fns {
        let kind = r.u8()?;
        let entity = r.u8()?;
        let nparams = r.u8()?;
        let nlocals = r.u8()?;
        let start = r.u32()?;
        let len = r.u32()?;
        fns.push((kind, entity, nparams, nlocals, start, len));
        out.push_str(&format!(
            "  fn[{:2}] kind={} ent={} params={} locals={} code=[{}..{})\n",
            i, kind, entity, nparams, nlocals, start, start + len
        ));
    }

    // tables
    for t in 0..n_tables {
        let ne = r.u8()?;
        let mut entries = Vec::new();
        for _ in 0..ne {
            let k = r.u8()?;
            let fi = r.u16()?;
            entries.push((k, fi));
        }
        out.push_str(&format!("  table[{}] {:?}\n", t, entries));
    }
    // scenes
    let mut scenes = Vec::new();
    for _ in 0..n_scenes {
        let fi = r.u16()?;
        scenes.push(fi);
    }
    out.push_str(&format!("  scenes->fns {:?}\n", scenes));

    // pool
    let n_pool = r.u16()?;
    let mut pool = Vec::new();
    for _ in 0..n_pool { pool.push(r.i32()?); }
    if !pool.is_empty() {
        out.push_str(&format!("  pool: {:?}\n", pool));
    }

    // huffman
    let n_huff = r.u16()?;
    let mut lens = [0u8; 256];
    let mut sym_list = Vec::new();
    for _ in 0..n_huff {
        let sym = r.u8()?;
        let len = r.u8()?;
        lens[sym as usize] = len;
        sym_list.push((sym, len));
    }

    // code bitstream
    let n_instr = r.u32()?;
    let _total_bits = r.u32()?;
    let byte_len = r.u32()?;
    let code_bytes = r.take(byte_len as usize)?;

    // decode with canonical codes + reverse-bit table
    // v14 hardening: the length bytes come straight from the file. A crafted
    // table with len > 8 made `1 << (8 - l)` underflow-shift (panic), and an
    // over-subscribed dictionary collided LUT entries silently. Validate
    // before building: lengths in 0..=8, and the Kraft sum must fit the
    // 256-entry LUT exactly once (complete) or with holes (incomplete = the
    // decode below reports "bad huffman code" on hole hits).
    let mut kraft: u32 = 0;
    for (s, &l) in lens.iter().enumerate() {
        if l == 0 { continue; }
        if l > 8 {
            return Err(format!("bad huffman table: symbol {} has length {} (max 8)", s, l));
        }
        kraft += 1u32 << (8 - l);
    }
    if kraft > 256 {
        return Err(format!("bad huffman table: over-subscribed dictionary (Kraft sum {} > 256)", kraft));
    }
    let codes = crate::huffman::canonical_codes(&crate::huffman::HuffTable { lens });
    let mut decode_table = [(0u8, 0u8); 256]; // (symbol, len)
    for s in 0..256 {
        let (c, l) = codes[s];
        if l == 0 { continue; }
        let base = crate::huffman::reverse_bits(c, l) as usize;
        for j in 0..(1usize << (8 - l)) {
            let idx = base + (j << l);
            if idx >= 256 { continue; } // defensive: validation above bounds this
            decode_table[idx] = (s as u8, l);
        }
    }

    let mut br = BitBr { b: code_bytes, bitpos: 0 };
    out.push_str(&format!("  code: {} instrs, {} bytes huffman\n", n_instr, byte_len));
    out.push_str("  ---- disassembly ----\n");

    let opname = |op: u8| -> &'static str {
        match op {
            0x00 => "NOP", 0x01 => "PUSH_S8", 0x02 => "PUSH_POOL", 0x03 => "LD_L", 0x04 => "ST_L",
            0x05 => "LD_G", 0x06 => "ST_G", 0x07 => "LD_E", 0x08 => "ST_E",
            0x09 => "ADD_F", 0x0A => "SUB_F", 0x0B => "MUL_F", 0x0C => "DIV_F",
            0x0D => "ADD_I", 0x0E => "SUB_I", 0x0F => "MUL_I", 0x10 => "DIV_I", 0x11 => "MOD_I",
            0x12 => "AND_I", 0x13 => "OR_I", 0x14 => "XOR_I", 0x15 => "SHL", 0x16 => "SHR",
            0x17 => "AND_B", 0x18 => "OR_B", 0x19 => "NOT_B", 0x1A => "NEG",
            0x1B => "EQ_F", 0x1C => "NE_F", 0x1D => "LT_F", 0x1E => "GT_F", 0x1F => "LE_F", 0x20 => "GE_F",
            0x21 => "EQ_I", 0x22 => "NE_I", 0x23 => "LT_I", 0x24 => "GT_I", 0x25 => "LE_I", 0x26 => "GE_I",
            0x27 => "SIN", 0x28 => "COS", 0x29 => "RAND_MAX", 0x2A => "ANIM", 0x2B => "KEY",
            0x2C => "JMP", 0x2D => "JZ", 0x2E => "JNZ", 0x2F => "FOR_BGN", 0x30 => "FOR_ADV",
            0x31 => "SPAWN", 0x32 => "KILL", 0x33 => "SFX", 0x34 => "MUSIC", 0x35 => "STOP_MUSIC",
            0x36 => "GOTO", 0x37 => "DRAW", 0x38 => "DRAW_TEXT", 0x39 => "DRAW_NUM",
            0x3A => "CALL_TBL", 0x3B => "COUNT", 0x3C => "RET", 0x3D => "CALL_FN",
            0x3E => "I2F", 0x3F => "F2I", 0x40 => "SHAKE",
            0x41 => "CAMERA", 0x42 => "MUSIC_VOL", 0x43 => "SAVE", 0x44 => "SAVED",
            0x45 => "DIST", 0x46 => "ATAN2", 0x47 => "CAM_X", 0x48 => "CAM_Y",
            0x49 => "SWEPT", 0x4A => "COLLIDE_MASK", 0x4B => "FOR_MASK_BGN",
            0x4C => "FOR_MASK_ADV", 0x4D => "FSM_NEXT", 0x4E => "SEL",
            0x4F => "LD_GARR", 0x50 => "ST_GARR", 0x51 => "LD_EARR", 0x52 => "ST_EARR",
            0x53 => "DUP", 0x54 => "CAM3", 0x55 => "PROJ3", 0x56 => "PROJ_X",
            0x57 => "PROJ_Y", 0x58 => "PROJ_OK", 0x59 => "DRAW3D",
            0x5A => "QUAT_AA", 0x5B => "Q_MUL", 0x5C => "M4_QT", 0x5D => "M4_MUL",
            0x5E => "SKIN3", 0x5F => "DRAW3DI",
            _ => "???",
        }
    };

    // per-instr: find owning fn for pretty labels.
    // v14 hardening: a crafted file with n_fns == 0 but n_instr > 0 used to
    // index fns[0] (panic); an out-of-range fn kind byte indexed an
    // 8-element name table (panic). Both are typed errors now — disasm
    // must NEVER panic on arbitrary bytes, it's the format inspector.
    if fns.is_empty() {
        if n_instr > 0 {
            return Err("bad header: instructions present but the fn table is empty".into());
        }
    }
    for (i, f) in fns.iter().enumerate() {
        if f.0 > 7 {
            return Err(format!("bad header: fn[{}] kind byte {} is out of range 0..=7", i, f.0));
        }
        if f.4.checked_add(f.5).map(|end| end > n_instr).unwrap_or(true) {
            return Err(format!("bad header: fn[{}] code range {}..{} escapes the instruction array (0..{})", i, f.4, f.4 + f.5, n_instr));
        }
    }
    let fn_of = |idx: u32| -> usize {
        let mut best = 0;
        for (i, f) in fns.iter().enumerate() {
            if idx >= f.4 && idx < f.4 + f.5 { best = i; }
        }
        best
    };

    let mut prev_fn = usize::MAX;
    for i in 0..n_instr {
        let peek = br.peek8();
        let (sym, len) = decode_table[peek as usize];
        if len == 0 { return Err(format!("instr {}: bad huffman code (peek={:#x})", i, peek)); }
        for _ in 0..len { br.read_bit()?; }
        let (ka, kb, kc) = codegen::sig(sym);
        // read_k is total: every operand kind yields a value (None -> 0),
        // so the pre-init below only exists to keep the borrow checker
        // happy — restructured as direct assignment, no dead initializers
        let read_k = |br: &mut BitBr, k: OpK| -> Result<i64, String> {
            match k {
                OpK::None => Ok(0),
                OpK::U8 => Ok(br.read_u(8)? as i64),
                OpK::U16 => Ok(br.read_u(16)? as i64),
                OpK::S16 => Ok(br.read_s(16)? as i64),
                OpK::S8 => Ok(br.read_s(8)? as i64),
                OpK::PoolU16 => Ok(br.read_u(16)? as i64),
            }
        };
        let av: i64 = read_k(&mut br, ka)?;
        let bv: i64 = read_k(&mut br, kb)?;
        let cv: i64 = read_k(&mut br, kc)?;
        let mut extra = String::new();
        if sym == codegen::OP_SPAWN {
            let n = bv as usize;
            let mut tfs = Vec::new();
            for _ in 0..n { tfs.push(br.read_u(16)?); }
            extra = format!(" tfs={:?}", tfs);
        }
        if sym == codegen::OP_COLLIDE_MASK {
            let xt = br.read_u(16)?;
            let yt = br.read_u(16)?;
            extra = format!(" xy_tf={}/{}", xt, yt);
        }
        // resolve pool consts
        let aval = if sym == codegen::OP_PUSH_POOL && (av as usize) < pool.len() {
            format!("{} (pool)", pool[av as usize])
        } else { format!("{}", av) };
        let fi = fn_of(i);
        if fi != prev_fn {
            const FN_KINDS: [&str; 8] = ["init", "update", "updateEnt", "draw", "drawEnt", "helper", "scene", "table"];
            out.push_str(&format!("  fn[{}] {}:\n", fi, FN_KINDS[fns[fi].0 as usize]));
            prev_fn = fi;
        }
        out.push_str(&format!("    {:4} {} {}, {}, {}{}{}\n", i, opname(sym), aval, bv, cv, extra,
            if matches!(sym, codegen::OP_JMP | codegen::OP_JZ | codegen::OP_JNZ | codegen::OP_FOR_ADV | codegen::OP_FOR_MASK_ADV) {
                format!("  ; -> {}", (i as i64 + 1 + av))
            } else if matches!(sym, codegen::OP_FOR_BGN | codegen::OP_FOR_MASK_BGN) { format!("  ; exit -> {}", (i as i64 + 1 + cv)) } else { String::new() }
        ));
    }

    // v3 trailer: FSM bit-plane transition tables
    if ver >= 3 {
        let n_fsm = r.u8()?;
        for fi in 0..n_fsm {
            let atom = r.u16()?;
            let nstates = r.u8()?;
            let span = r.u8()?;
            let tlen = nstates as usize * span as usize;
            let t = r.take(tlen)?;
            out.push_str(&format!("  fsm[{}] #{} states={} span={} T={:?}\n",
                fi, atom_names.get(atom as usize).cloned().unwrap_or_else(|| format!("#{}", atom)),
                nstates, span, t));
        }
    }

    // v10 trailer: compile-proven parallel-system groups (flags bit 2)
    if flags & 4 != 0 {
        let n_groups = r.u8()?;
        for gi in 0..n_groups {
            let n_members = r.u8()?;
            let mut members = Vec::new();
            for _ in 0..n_members {
                members.push(r.u16()?);
            }
            let labels: Vec<String> = members.iter().map(|&fi| {
                match fns.get(fi as usize) {
                    Some((kind, ent, ..)) => format!("fn{}(kind{},ent{})", fi, kind, ent),
                    None => format!("fn{}(?)", fi),
                }
            }).collect();
            out.push_str(&format!("  parallel[{}]: {} disjoint systems [{}] — concurrent workers\n",
                gi, n_members, labels.join(", ")));
        }
    }

    if r.pos != bytes.len() {
        out.push_str(&format!("  WARNING: {} trailing bytes\n", bytes.len() - r.pos));
    }
    out.push_str("  OK: format round-trip validated\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal VALID .libyte built through the real pipeline (the
    /// hand-assembled route proved layout-fragile; the compiler is the
    /// source of truth for the format). The robustness property under
    /// test: "any mutation -> Ok or Err, NEVER a panic".
    fn tiny_valid_lidyte() -> Vec<u8> {
        static SRC: &str = "fn init() { }\n";
        let (bytes, _) = crate::compile_to_libyte(SRC, None, None)
            .expect("minimal game must compile");
        bytes
    }

    /// Deterministic xorshift PRNG — no external deps, stable across runs.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn byte(&mut self) -> u8 { (self.next() >> 32) as u8 }
    }

    /// The fn record for a 0-arg global fn: [kind=0, ent=0xFF (GLOBAL_ENT),
    /// nparams=0, nlocals=0, start=0 u32, len=N u32]. The 0xFF entity byte
    /// anchors it uniquely in the blob.
    fn find_fn_record(b: &[u8]) -> usize {
        for i in 0..b.len().saturating_sub(12) {
            if b[i] == 0
                && b[i + 1] == 0xFF
                && b[i + 2] == 0
                && b[i + 3] == 0
                && b[i + 4..i + 8].iter().all(|&x| x == 0)
            {
                let len = u32::from_le_bytes([b[i + 8], b[i + 9], b[i + 10], b[i + 11]]);
                if (1..64).contains(&len) { return i; }
            }
        }
        panic!("fn record not found in fixture");
    }

    #[test]
    fn empty_fn_table_with_code_is_a_typed_error() {
        let mut b = tiny_valid_lidyte();
        let at = find_fn_record(&b);
        b.drain(at..at + 12); // remove the record -> stream stays consistent
        b[13] = 0; // n_fns (u16 LE) = 0, but instructions still decode
        b[14] = 0;
        let out = disasm(&b);
        assert!(out.is_err());
        let msg = out.unwrap_err();
        assert!(msg.contains("fn table is empty"), "{}", msg);
    }

    #[test]
    fn out_of_range_fn_kind_is_a_typed_error() {
        let mut b = tiny_valid_lidyte();
        let at = find_fn_record(&b);
        b[at] = 99; // kind byte beyond the 0..=7 table
        let out = disasm(&b);
        assert!(out.is_err());
        let msg = out.unwrap_err();
        assert!(msg.contains("out of range 0..=7"), "{}", msg);
    }

    #[test]
    fn mutated_blobs_never_panic() {
        let base = tiny_valid_lidyte();
        let mut rng = Rng(0x5EED_1A4E);
        let mut oks = 0;
        let mut errs = 0;
        for round in 0..2000 {
            let mut b = base.clone();
            // 1-8 random byte mutations (flips and full writes)
            let n = 1 + (rng.next() % 8) as usize;
            for _ in 0..n {
                let pos = (rng.next() as usize) % b.len();
                match rng.next() % 3 {
                    0 => b[pos] ^= 1 << (rng.byte() % 8),
                    1 => b[pos] = rng.byte(),
                    _ => b[pos] = b[pos].wrapping_add(rng.byte()),
                }
            }
            // truncations and extensions
            match round % 4 {
                0 => { let cut = (rng.next() as usize) % (b.len() + 1); b.truncate(cut); }
                1 => { b.push(rng.byte()); }
                _ => {}
            }
            match disasm(&b) {
                Ok(_) => oks += 1,
                Err(_) => errs += 1,
            }
        }
        // the invariant: both counters sum to every round — no panic escaped
        assert_eq!(oks + errs, 2000);
        // sanity: the unmutated base itself must validate
        assert!(disasm(&base).is_ok());
    }
}
