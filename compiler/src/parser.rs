//! LILA parser — recursive descent, Pratt expressions.
//! Keywords are matched by identifier TEXT (lexer is keyword-free).

use crate::ast::*;
use crate::lexer::Tok;

pub struct Parser {
    toks: Vec<(Tok, u32, u32)>,
    pos: usize,
    pub atom_names: Vec<String>,
}

impl Parser {
    pub fn new(toks: Vec<(Tok, u32, u32)>, atom_names: Vec<String>) -> Self {
        Parser { toks, pos: 0, atom_names }
    }

    fn peek(&self) -> &Tok { &self.toks[self.pos].0 }
    fn line(&self) -> u32 { self.toks[self.pos].1 }
    fn col(&self) -> u32 { self.toks[self.pos].2 }
    fn peek_at(&self, off: usize) -> &Tok { &self.toks[std::cmp::min(self.pos + off, self.toks.len() - 1)].0 }

    fn next(&mut self) -> Tok { let t = self.toks[self.pos].0.clone(); self.pos += 1; t }

    fn is_kw(&self, s: &str) -> bool {
        matches!(self.peek(), Tok::Ident(name) if name == s)
    }

    fn eat_kw(&mut self, s: &str) -> bool {
        if self.is_kw(s) { self.pos += 1; true } else { false }
    }

    fn err(&self, msg: &str) -> String {
        format!("parse error at line {}:{}: {} (found {:?})", self.line(), self.col(), msg, self.peek())
    }

    fn expect(&mut self, t: Tok) -> Result<(), String> {
        if *self.peek() == t { self.pos += 1; Ok(()) }
        else { Err(self.err(&format!("expected {:?}", t))) }
    }

    fn expect_ident(&mut self) -> Result<String, String> {
        match self.next() {
            Tok::Ident(s) => Ok(s),
            other => Err(format!("parse error at line {}:{}: expected identifier, found {:?}", self.line(), self.col(), other)),
        }
    }

    fn expect_atom(&mut self) -> Result<u16, String> {
        match self.next() {
            Tok::Atom(a) => Ok(a),
            other => Err(format!("parse error at line {}:{}: expected #atom, found {:?}", self.line(), self.col(), other)),
        }
    }

    fn accept_semi_or_comma(&mut self) {
        // newline-separated items: separators optional
        if matches!(self.peek(), Tok::Semi | Tok::Comma) { self.pos += 1; }
    }

    // ---------------- top level ----------------

    pub fn parse_program(&mut self) -> Result<Program, String> {
        let mut p = Program::default();
        while !matches!(self.peek(), Tok::Eof) {
            if self.eat_kw("game") { self.parse_game(&mut p)?; }
            else if self.eat_kw("sound") { self.parse_sound(&mut p)?; }
            else if self.eat_kw("key") { self.parse_key(&mut p)?; }
            else if self.eat_kw("text") { self.parse_text(&mut p)?; }
            else if self.eat_kw("scene") { self.parse_scene(&mut p)?; }
            else if self.eat_kw("sprite") { self.parse_sprite(&mut p)?; }
            else if self.eat_kw("music") { self.parse_music(&mut p)?; }
            else if self.eat_kw("sfx") && matches!(self.peek(), Tok::Atom(_)) { self.parse_sfx(&mut p)?; }
            else if self.eat_kw("anim") { self.parse_anim(&mut p)?; }
            else if self.eat_kw("entity") { self.parse_entity(&mut p)?; }
            else if self.eat_kw("particles") && matches!(self.peek(), Tok::Atom(_)) { self.parse_particles(&mut p)?; }
            else if self.eat_kw("global") { self.parse_global(&mut p)?; }
            else if self.eat_kw("table") { self.parse_table(&mut p)?; }
            else if self.eat_kw("fsm") && matches!(self.peek(), Tok::Atom(_)) { p.fsms.push(self.parse_fsm()?); }
            else if self.eat_kw("sdf") && matches!(self.peek(), Tok::Atom(_)) { p.sdfs.push(self.parse_sdf()?); }
            else if self.eat_kw("fsm") || self.eat_kw("sdf") {
                return Err(self.err("fsm/sdf need a #name"));
            }
            else if self.eat_kw("fn") { p.fns.push(self.parse_fn()?); }
            else { return Err(self.err("expected a top-level declaration")); }
        }
        Ok(p)
    }

    fn parse_game(&mut self, p: &mut Program) -> Result<(), String> {
        let mut g = GameDecl { title: String::new(), width: 512, height: 512, wrap: true,
                               world_w: None, world_h: None, capacity: None, optimize: false };
        self.expect(Tok::LBrace)?;
        while !matches!(self.peek(), Tok::RBrace) {
            let k = self.expect_ident()?;
            if k == "capacity" && matches!(self.peek(), Tok::LBrace) {
                // capacity { entities: N } — how many live entities per
                // type this game is built for (drives the engine build
                // profile; default 512). Brace form: no colon.
                self.next(); // consume LBrace
                while !matches!(self.peek(), Tok::RBrace) {
                    let ck = self.expect_ident()?;
                    self.expect(Tok::Colon)?;
                    match ck.as_str() {
                        "entities" => {
                            let v = self.parse_int_pt()? as u32;
                            g.capacity = Some(v);
                        }
                        other => return Err(format!("line {}:{}: unknown capacity field '{}'", self.line(), self.col(), other)),
                    }
                    self.accept_semi_or_comma();
                }
                self.expect(Tok::RBrace)?;
                self.accept_semi_or_comma();
                continue;
            }
            self.expect(Tok::Colon)?;
            match k.as_str() {
                "title" => {
                    g.title = match self.next() {
                        Tok::Str(s) => s,
                        other => return Err(format!("line {}:{}: title needs a string, found {:?}", self.line(), self.col(), other)),
                    };
                }
                "width" => g.width = self.parse_int_pt()? as u32,
                "height" => g.height = self.parse_int_pt()? as u32,
                "wrap" => g.wrap = self.expect_bool()?,
                // v8 OPTIMIZER: opt into the full compile-time pipeline
                // (bit-width synthesis + cache-line SoA repacking). Default
                // false: legacy games keep byte-identical .libyte output.
                "optimize" => g.optimize = self.expect_bool()?,
                // v7: simulation bounds — a world larger than the screen is a
                // scrolling world driven by camera(x, y). Defaults: world == screen.
                "world_w" => {
                    let v = self.parse_int_pt()? as u32;
                    g.world_w = Some(v);
                }
                "world_h" => {
                    let v = self.parse_int_pt()? as u32;
                    g.world_h = Some(v);
                }
                other => return Err(format!("line {}:{}: unknown game field '{}'", self.line(), self.col(), other)),
            }
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        p.game = Some(g);
        Ok(())
    }

    fn parse_sound(&mut self, p: &mut Program) -> Result<(), String> {
        self.expect(Tok::LBrace)?;
        while !matches!(self.peek(), Tok::RBrace) {
            p.sound_atoms.push(self.expect_atom()?);
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        Ok(())
    }

    fn parse_key(&mut self, p: &mut Program) -> Result<(), String> {
        self.expect(Tok::LBrace)?;
        while !matches!(self.peek(), Tok::RBrace) {
            p.key_atoms.push(self.expect_atom()?);
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        Ok(())
    }

    fn parse_text(&mut self, p: &mut Program) -> Result<(), String> {
        self.expect(Tok::LBrace)?;
        while !matches!(self.peek(), Tok::RBrace) {
            let a = self.expect_atom()?;
            self.expect(Tok::Colon)?;
            let s = match self.next() {
                Tok::Str(s) => s,
                other => return Err(format!("line {}:{}: text atom needs a string, found {:?}", self.line(), self.col(), other)),
            };
            p.text_atoms.push((a, s));
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        Ok(())
    }

    fn parse_scene(&mut self, p: &mut Program) -> Result<(), String> {
        self.expect(Tok::LBrace)?;
        while !matches!(self.peek(), Tok::RBrace) {
            let a = self.expect_atom()?;
            self.expect(Tok::Colon)?;
            let f = self.expect_ident()?;
            p.scene_atoms.push((a, f));
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        Ok(())
    }

    fn parse_int_pt(&mut self) -> Result<i32, String> {
        let mut sign = 1i64;
        if matches!(self.peek(), Tok::Minus) { self.pos += 1; sign = -1; }
        match self.next() {
            Tok::Int(v) => Ok((sign * v as i64) as i32),
            other => Err(format!("line {}:{}: expected integer coordinate, found {:?}", self.line(), self.col(), other)),
        }
    }

    fn parse_sprite(&mut self, p: &mut Program) -> Result<(), String> {
        let atom = self.expect_atom()?;
        self.expect(Tok::LParen)?;
        // hit radius: syntactic (documented grammar `sprite #x (12)`) but
        // unused by the engine — collision radii are data fields now.
        // Consumed and validated, never stored.
        let _radius = match self.next() {
            Tok::Int(v) => v,
            other => return Err(format!("line {}:{}: sprite hit radius expected, found {:?}", self.line(), self.col(), other)),
        };
        self.expect(Tok::RParen)?;
        self.expect(Tok::LBrace)?;
        let mut polys = Vec::new();
        while !matches!(self.peek(), Tok::RBrace) {
            if !self.eat_kw("poly") { return Err(self.err("expected 'poly' in sprite body")); }
            self.expect(Tok::LBracket)?;
            let mut pts = Vec::new();
            while !matches!(self.peek(), Tok::RBracket) {
                let x = self.parse_int_pt()?;
                self.expect(Tok::Comma)?;
                let y = self.parse_int_pt()?;
                pts.push((x, y));
                if matches!(self.peek(), Tok::Comma) { self.pos += 1; }
            }
            self.expect(Tok::RBracket)?;
            if pts.len() < 3 { return Err(self.err("poly needs at least 3 points")); }
            let mut poly = PolyDecl { pts, fill: 0xFFFFFFFF, grad: None, add: false, pat: Pat::Flat };
            loop {
                if self.eat_kw("fill") {
                    self.expect(Tok::Colon)?;
                    poly.fill = self.expect_hex()?;
                } else if self.eat_kw("grad") {
                    self.expect(Tok::Colon)?;
                    poly.grad = Some(self.expect_hex()?);
                } else if self.eat_kw("add") {
                    self.expect(Tok::Colon)?;
                    poly.add = self.expect_bool()?;
                } else if self.eat_kw("pat") {
                    self.expect(Tok::Colon)?;
                    let w = self.expect_ident()?;
                    poly.pat = match w.as_str() {
                        "flat" => Pat::Flat,
                        "noise" => Pat::Noise,
                        other => return Err(format!("unknown pattern '{}'", other)),
                    };
                } else { break; }
            }
            polys.push(poly);
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        p.sprites.push(SpriteDecl { atom, polys });
        Ok(())
    }

    fn expect_hex(&mut self) -> Result<u32, String> {
        match self.next() {
            Tok::Int(v) => Ok(v),
            other => Err(format!("line {}:{}: expected hex color literal, found {:?}", self.line(), self.col(), other)),
        }
    }

    fn expect_bool(&mut self) -> Result<bool, String> {
        match self.next() {
            Tok::Ident(s) if s == "true" => Ok(true),
            Tok::Ident(s) if s == "false" => Ok(false),
            other => Err(format!("line {}:{}: expected true/false, found {:?}", self.line(), self.col(), other)),
        }
    }

    fn parse_note_step(&mut self) -> Result<Step, String> {
        match self.next() {
            Tok::Dot => Ok(Step::Silence),
            Tok::Minus => Ok(Step::Hold),
            Tok::Ident(s) if s == "x" => Ok(Step::Hit),
            Tok::Ident(s) => {
                // note like C4, E2, B8
                let bytes = s.as_bytes();
                if bytes.len() < 2 || bytes.len() > 2 {
                    return Err(format!("line {}:{}: bad note '{}' (use e.g. C4)", self.line(), self.col(), s));
                }
                let semi = match bytes[0] {
                    b'C' => 0, b'D' => 2, b'E' => 4, b'F' => 5,
                    b'G' => 7, b'A' => 9, b'B' => 11,
                    _ => return Err(format!("line {}:{}: bad note letter in '{}'", self.line(), self.col(), s)),
                };
                let oct = (bytes[1] as char).to_digit(10).unwrap_or(0);
                if !(1..=8).contains(&oct) {
                    return Err(format!("line {}:{}: bad octave in '{}' (1..8)", self.line(), self.col(), s));
                }
                Ok(Step::Note((((oct - 1) * 12) + semi) as u8))
            }
            other => Err(format!("line {}:{}: bad music step, found {:?}", self.line(), self.col(), other)),
        }
    }

    fn parse_music(&mut self, p: &mut Program) -> Result<(), String> {
        let atom = self.expect_atom()?;
        let mut bpm = 120u32;
        let mut voices = Vec::new();
        self.expect(Tok::LBrace)?;
        while !matches!(self.peek(), Tok::RBrace) {
            let kw = self.expect_ident()?;
            self.expect(Tok::Colon)?;
            match kw.as_str() {
                "bpm" => {
                    bpm = match self.next() {
                        Tok::Int(v) => v,
                        other => return Err(format!("line {}:{}: bpm needs int, found {:?}", self.line(), self.col(), other)),
                    };
                }
                "square" | "tri" | "noise" | "saw" => {
                    let wave = match kw.as_str() {
                        "square" => Wave::Square, "tri" => Wave::Tri,
                        "noise" => Wave::Noise, _ => Wave::Saw,
                    };
                    let mut vol = 12u8;
                    self.expect(Tok::LBracket)?;
                    let mut steps = Vec::new();
                    while !matches!(self.peek(), Tok::RBracket) {
                        steps.push(self.parse_note_step()?);
                    }
                    // format contract: the runtime stores 16 steps per voice
                    // (40B voice slot) — reject loudly at parse time
                    if steps.len() > 16 {
                        return Err(format!("line {}:{}: music voice '{}' has {} steps (max 16 per voice)",
                            self.line(), self.col(), kw, steps.len()));
                    }
                    self.expect(Tok::RBracket)?;
                    // optional trailing vol: N before separator
                    if self.eat_kw("vol") {
                        self.expect(Tok::Colon)?;
                        vol = match self.next() {
                            Tok::Int(v) if v <= 15 => v as u8,
                            _ => return Err(format!("line {}:{}: vol must be 0..15", self.line(), self.col())),
                        };
                    }
                    voices.push(VoiceDecl { wave, steps, vol });
                }
                other => return Err(format!("line {}:{}: unknown music field '{}'", self.line(), self.col(), other)),
            }
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        p.music.push(MusicDecl { atom, bpm, voices });
        Ok(())
    }

    fn parse_sfx(&mut self, p: &mut Program) -> Result<(), String> {
        let atom = self.expect_atom()?;
        let mut d = SfxDecl { atom, wave: Wave::Square, freq: 440, sweep: 0, decay: 20, vol: 10, from: None };
        self.expect(Tok::LBrace)?;
        while !matches!(self.peek(), Tok::RBrace) {
            let k = self.expect_ident()?;
            self.expect(Tok::Colon)?;
            match k.as_str() {
                "wave" => {
                    let w = self.expect_ident()?;
                    d.wave = match w.as_str() {
                        "square" => Wave::Square, "tri" => Wave::Tri,
                        "noise" => Wave::Noise, "saw" => Wave::Saw,
                        other => return Err(format!("line {}:{}: unknown wave '{}'", self.line(), self.col(), other)),
                    };
                }
                "freq" => d.freq = self.parse_int_pt()?,
                "sweep" => d.sweep = self.parse_int_pt()?,
                "decay" => d.decay = self.parse_int_pt()?,
                "vol" => {
                    let v = self.parse_int_pt()?;
                    if !(0..=15).contains(&v) { return Err(format!("line {}:{}: sfx vol must be 0..15", self.line(), self.col())); }
                    d.vol = v as u32;
                }
                // v8 OPTIMIZER subsystem 4: build-time WAV synthesis source.
                // The path is read ONCE at compile time; the .libyte carries
                // only the fitted parameters (no strings, no samples).
                "from" => {
                    let s = match self.next() {
                        Tok::Str(s) => s,
                        other => return Err(format!("line {}:{}: sfx from: needs a path string, found {:?}", self.line(), self.col(), other)),
                    };
                    d.from = Some(s);
                }
                other => return Err(format!("line {}:{}: unknown sfx field '{}'", self.line(), self.col(), other)),
            }
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        p.sfx.push(d);
        Ok(())
    }

    fn parse_anim(&mut self, p: &mut Program) -> Result<(), String> {
        let atom = self.expect_atom()?;
        self.expect(Tok::LBrace)?;
        if !self.eat_kw("keys") { return Err(self.err("anim needs 'keys:'")); }
        self.expect(Tok::Colon)?;
        self.expect(Tok::LBracket)?;
        let mut keys = Vec::new();
        while !matches!(self.peek(), Tok::RBracket) {
            self.expect(Tok::LParen)?;
            let t = match self.next() {
                Tok::Int(v) => v,
                other => return Err(format!("line {}:{}: anim key time expected, found {:?}", self.line(), self.col(), other)),
            };
            self.expect(Tok::Comma)?;
            let mut sign = 1i64;
            if matches!(self.peek(), Tok::Minus) { self.pos += 1; sign = -1; }
            let val = match self.next() {
                Tok::Int(v) => sign * v as i64,
                Tok::Fix(f) => f as i64,
                other => return Err(format!("line {}:{}: anim key value expected, found {:?}", self.line(), self.col(), other)),
            };
            self.expect(Tok::RParen)?;
            keys.push((t, val as i32));
            if matches!(self.peek(), Tok::Comma) { self.pos += 1; }
        }
        self.expect(Tok::RBracket)?;
        self.expect(Tok::RBrace)?;
        if keys.is_empty() { return Err(self.err("anim needs at least one key")); }
        // monotone t check
        for w in keys.windows(2) {
            if w[1].0 <= w[0].0 { return Err(self.err("anim key times must be strictly increasing")); }
        }
        p.anims.push(AnimDecl { atom, keys });
        Ok(())
    }

    fn parse_field_type(&mut self) -> Result<Ty, String> {
        let name = self.expect_ident()?;
        match name.as_str() {
            "fixed" => Ok(Ty::Fixed),
            "bool" => Ok(Ty::Bool),
            "ang" => Ok(Ty::Ang),
            "u" => Ok(Ty::UInt(0)),  // 0 = infer marker
            "s" => Ok(Ty::SInt(0)),
            other => {
                if let Some(digits) = other.strip_prefix('u') {
                    let w: u8 = digits.parse().map_err(|_| format!("bad width in '{}'", other))?;
                    if !(1..=16).contains(&w) { return Err(format!("u{} width out of range 1..16", w)); }
                    return Ok(Ty::UInt(w));
                }
                if let Some(digits) = other.strip_prefix('s') {
                    let w: u8 = digits.parse().map_err(|_| format!("bad width in '{}'", other))?;
                    if !(1..=16).contains(&w) { return Err(format!("s{} width out of range 1..16", w)); }
                    return Ok(Ty::SInt(w));
                }
                Ok(Ty::Entity(name)) // entity-typed helper param — checker validates
            }
        }
    }

    fn parse_field(&mut self) -> Result<FieldDecl, String> {
        let cold = self.eat_kw("cold");
        let name = self.expect_ident()?;
        self.expect(Tok::Colon)?;
        let ty = self.parse_field_type()?;
        let mut f = FieldDecl { name, ty, default_int: None, default_fix: None, default_bool: None, cold };
        if matches!(self.peek(), Tok::Eq) {
            self.pos += 1;
            match self.next() {
                Tok::Int(v) => f.default_int = Some(v as i64),
                Tok::Fix(v) => f.default_fix = Some(v),
                Tok::Ident(s) if s == "true" => f.default_bool = Some(true),
                Tok::Ident(s) if s == "false" => f.default_bool = Some(false),
                Tok::Minus => match self.next() {
                    Tok::Int(v) => f.default_int = Some(-(v as i64)),
                    Tok::Fix(v) => f.default_fix = Some(-v),
                    other => return Err(format!("line {}:{}: bad default value, found {:?}", self.line(), self.col(), other)),
                },
                other => return Err(format!("line {}:{}: bad default value, found {:?}", self.line(), self.col(), other)),
            }
        }
        self.accept_semi_or_comma();
        Ok(f)
    }

    fn parse_entity(&mut self, p: &mut Program) -> Result<(), String> {
        let name = self.expect_ident()?;
        // v10 PREFABS: `entity FastDrone from Drone { ... }` — the parent is
        // resolved (and the fields merged) by synth::expand after the whole
        // file is parsed, so declaration order is free like everything else.
        let from = if self.is_kw("from") {
            self.pos += 1;
            Some(self.expect_ident()?)
        } else {
            None
        };
        self.expect(Tok::LBrace)?;
        let mut fields = Vec::new();
        let mut arrs = Vec::new();
        let mut fsm: Option<String> = None;
        while !matches!(self.peek(), Tok::RBrace) {
            // v8 OPTIMIZER: `fsm: #name` binds a state machine to this entity.
            // Disambiguated from a field literally named `fsm` by the #atom.
            if self.is_kw("fsm") && matches!(self.peek_at(1), Tok::Colon) && matches!(self.peek_at(2), Tok::Atom(_)) {
                self.pos += 2;
                let a = self.expect_atom()?;
                fsm = Some(self.atom_names.get(a as usize).cloned()
                    .ok_or_else(|| format!("line {}:{}: bad fsm atom", self.line(), self.col()))?);
                self.accept_semi_or_comma();
                continue;
            }
            // v11 ARRAYS: `arr fixed[16] trail;` inside an entity block = a
            // per-instance array (its own plane, not a bit-packed field).
            if self.is_kw("arr") {
                arrs.push(self.parse_arr()?);
                continue;
            }
            fields.push(self.parse_field()?);
        }
        self.expect(Tok::RBrace)?;
        p.entities.push(EntityDecl { name, fields, fsm, from, max_live_override: None, arrs });
        Ok(())
    }

    /// v11 ARRAYS: `arr <ty>[<cap>] <name>;` — element type, compile-time
    /// capacity, no default (zero-initialized). Used in global + entity blocks.
    /// v12: array arguments accept a plain global name or an `e.name`
    /// dotted form; the CHECKER rejects anything that isn't a global array
    /// with a precise message (rig storage is shared, not per-instance).
    fn parse_arr_ref(&mut self) -> Result<String, String> {
        let base = self.expect_ident()?;
        if matches!(self.peek(), Tok::Dot) {
            self.pos += 1;
            let arr = self.expect_ident()?;
            return Ok(format!("{}.{}", base, arr));
        }
        Ok(base)
    }

    fn parse_arr(&mut self) -> Result<ArrDecl, String> {
        let (line, col) = (self.line(), self.col());
        self.pos += 1; // consume 'arr'
        let elem = self.parse_field_type()?;
        self.expect(Tok::LBracket)?;
        let cap = match self.next() {
            Tok::Int(v) if v >= 1 && v <= 16384 => v,
            other => return Err(format!("line {}:{}: array capacity must be an integer literal 1..=16384, found {:?}", line, col, other)),
        };
        self.expect(Tok::RBracket)?;
        let name = self.expect_ident()?;
        self.accept_semi_or_comma();
        Ok(ArrDecl { name, elem, cap, line })
    }

    /// v10 PARTICLES: `particles #boomfx (192) { sprite: #p  life: 22
    /// speed: 1.8  drag: 0.92 ... }` — parsed into a ParticleDecl and lowered
    /// to an entity + emit/update/draw fns by synth::expand.
    fn parse_particles(&mut self, p: &mut Program) -> Result<(), String> {
        let atom = self.expect_atom()?;
        self.expect(Tok::LParen)?;
        let cap = self.parse_int_pt()? as u32;
        self.expect(Tok::RParen)?;
        self.expect(Tok::LBrace)?;
        let mut sprite: Option<u16> = None;
        let (mut life, mut speed_fix, mut jitter_q8) = (24u32, 384i32, 128u32);
        let (mut drag_q8, mut grav_fix, mut burst) = (256u32, 0i32, 8u32);
        let (mut color, mut fade) = (0xFFFFFFu32, true);
        while !matches!(self.peek(), Tok::RBrace) {
            let k = self.expect_ident()?;
            self.expect(Tok::Colon)?;
            match k.as_str() {
                "sprite" => sprite = Some(self.expect_atom()?),
                "life" => life = self.parse_int_pt()? as u32,
                "speed" => match self.next() {
                    Tok::Fix(v) => speed_fix = v,
                    Tok::Int(v) => speed_fix = (v as i32) * 256,
                    other => return Err(format!("line {}:{}: bad speed, found {:?}", self.line(), self.col(), other)),
                },
                "jitter" => match self.next() {
                    // Tok::Fix carries raw Q8 (0.6 -> 154): clamp to 0..256
                    Tok::Fix(v) => jitter_q8 = v.clamp(0, 256) as u32,
                    other => return Err(format!("line {}:{}: bad jitter, found {:?}", self.line(), self.col(), other)),
                },
                "drag" => match self.next() {
                    // Tok::Fix carries raw Q8 (0.93 -> 238): clamp to 0..256
                    Tok::Fix(v) => drag_q8 = v.clamp(0, 256) as u32,
                    other => return Err(format!("line {}:{}: bad drag, found {:?}", self.line(), self.col(), other)),
                },
                "grav" => match self.next() {
                    Tok::Fix(v) => grav_fix = v,
                    Tok::Int(v) => grav_fix = (v as i32) * 256,
                    Tok::Minus => match self.next() {
                        Tok::Fix(v) => grav_fix = -v,
                        Tok::Int(v) => grav_fix = -(v as i32) * 256,
                        other => return Err(format!("line {}:{}: bad grav, found {:?}", self.line(), self.col(), other)),
                    },
                    other => return Err(format!("line {}:{}: bad grav, found {:?}", self.line(), self.col(), other)),
                },
                "burst" => burst = self.parse_int_pt()? as u32,
                "color" => match self.next() {
                    Tok::Int(v) => color = v & 0xFFFFFF,
                    other => return Err(format!("line {}:{}: bad color, found {:?}", self.line(), self.col(), other)),
                },
                "fade" => match self.next() {
                    Tok::Ident(s) if s == "true" => fade = true,
                    Tok::Ident(s) if s == "false" => fade = false,
                    other => return Err(format!("line {}:{}: bad fade, found {:?}", self.line(), self.col(), other)),
                },
                other => return Err(format!("line {}:{}: unknown particles field '{}' (sprite life speed jitter drag grav burst color fade)", self.line(), self.col(), other)),
            }
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        let sprite = sprite.ok_or_else(|| {
            format!("line {}:{}: particles #{} needs a `sprite: #name`", self.line(), self.col(),
                self.atom_names.get(atom as usize).map(|s| s.as_str()).unwrap_or("?"))
        })?;
        if cap < 16 || cap > 65535 {
            return Err(format!("line {}:{}: particles capacity {} out of range 16..=65535", self.line(), self.col(), cap));
        }
        if life == 0 || life > 255 { return Err(format!("line {}:{}: particles life {} out of range 1..=255", self.line(), self.col(), life)); }
        if burst == 0 || burst > 64 { return Err(format!("line {}:{}: particles burst {} out of range 1..=64", self.line(), self.col(), burst)); }
        p.particles.push(crate::ast::ParticleDecl {
            atom, cap, sprite, life, speed_fix, jitter_q8, drag_q8, grav_fix, burst, color, fade,
        });
        Ok(())
    }

    fn parse_global(&mut self, p: &mut Program) -> Result<(), String> {
        self.expect(Tok::LBrace)?;
        while !matches!(self.peek(), Tok::RBrace) {
            // v11 ARRAYS: `arr fixed[64] stars;` — game-wide arrays (meshes,
            // paths, grids) living in their own flat-memory region.
            if self.is_kw("arr") {
                p.garrs.push(self.parse_arr()?);
                continue;
            }
            p.globals.push(self.parse_field()?);
        }
        self.expect(Tok::RBrace)?;
        Ok(())
    }

    fn parse_table(&mut self, p: &mut Program) -> Result<(), String> {
        let name = self.expect_ident()?;
        self.expect(Tok::LBrace)?;
        let mut entries = Vec::new();
        while !matches!(self.peek(), Tok::RBrace) {
            let key = match self.next() {
                Tok::Int(v) if v <= 255 => v as u8,
                other => return Err(format!("line {}:{}: table key must be 0..255, found {:?}", self.line(), self.col(), other)),
            };
            self.expect(Tok::Colon)?;
            let f = self.expect_ident()?;
            entries.push((key, f));
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        p.tables.push(TableDecl { name, entries });
        Ok(())
    }

    fn parse_fn(&mut self) -> Result<FnDecl, String> {
        // Capture the fn's own line BEFORE parsing the body. `parse_block()`
        // advances `pos` past the closing brace, so reading line() afterwards
        // reported the line *after* the function (its closing brace or EOF) —
        // and since statements carry no position of their own, every
        // checker/codegen error (`f.line`) pointed there instead of at the fn.
        let line = self.line();
        let name = self.expect_ident()?;
        self.expect(Tok::LParen)?;
        let mut params = Vec::new();
        while !matches!(self.peek(), Tok::RParen) {
            let pn = self.expect_ident()?;
            self.expect(Tok::Colon)?;
            let ty = self.parse_field_type()?;
            params.push((pn, ty));
            if matches!(self.peek(), Tok::Comma) { self.pos += 1; }
        }
        self.expect(Tok::RParen)?;
        let body = self.parse_block()?;
        Ok(FnDecl { name, params, body, line })
    }

    // ---------------- v8 OPTIMIZER declarations ----------------

    /// fsm #name {
    ///     STATE { guard -> TARGET, guard -> TARGET }
    ///     ...
    /// }
    /// Guards are expressions over the owning entity's fields (implicit self),
    /// globals, and key() — the checker validates them in entity context.
    fn parse_fsm(&mut self) -> Result<FsmDecl, String> {
        let atom = self.expect_atom()?;
        // the fsm NAME is the #atom name (same convention as sdf blocks —
        // `fsm #guard { ... }`); no separate identifier is required.
        let name = self.atom_names.get(atom as usize).cloned()
            .ok_or_else(|| format!("line {}:{}: unknown fsm atom", self.line(), self.col()))?;
        self.expect(Tok::LBrace)?;
        let mut states: Vec<(String, Vec<(Expr, String)>)> = Vec::new();
        while !matches!(self.peek(), Tok::RBrace) {
            let sname = self.expect_ident()?;
            self.expect(Tok::LBrace)?;
            let mut guards = Vec::new();
            while !matches!(self.peek(), Tok::RBrace) {
                let pred = self.parse_expr()?;
                self.expect(Tok::Arrow)?;
                let target = self.expect_ident()?;
                guards.push((pred, target));
                self.accept_semi_or_comma();
            }
            self.expect(Tok::RBrace)?;
            states.push((sname, guards));
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        let line = self.line();
        Ok(FsmDecl { atom, name, states, line })
    }

    /// sdf #name { ... } — shape programs (see opt/sdf.rs for the pass docs).
    fn parse_sdf(&mut self) -> Result<crate::opt::sdf::SdfDecl, String> {
        use crate::opt::sdf::{SdfDecl, Shape, V3};
        let atom = self.expect_atom()?;
        let (line, col) = (self.line(), self.col());
        let name = self.atom_names.get(atom as usize).cloned()
            .ok_or_else(|| format!("line {}:{}: unknown sdf atom", line, col))?;
        self.expect(Tok::LBrace)?;
        let mut kids: Vec<Shape> = Vec::new(); // top-level implicit union
        let mut color = 0x33AAFFu32;
        let mut glow = 0.5f32;
        let mut eye = V3::new(0.0, 1.2, -3.4);
        let mut look = V3::zero();
        let mut hud = false;
        while !matches!(self.peek(), Tok::RBrace) {
            let kw = self.expect_ident()?;
            match kw.as_str() {
                "sphere" => {
                    self.expect(Tok::LBrace)?;
                    let (mut r, mut c) = (1.0f32, V3::zero());
                    self.sdf_params(|s, key| match key {
                        "r" => { r = s.sdf_f32()?; Ok(()) }
                        "at" => { c = s.sdf_v3()?; Ok(()) }
                        other => Err(format!("unknown sphere field '{}'", other)),
                    })?;
                    kids.push(Shape::Sphere { c, r });
                }
                "box" => {
                    self.expect(Tok::LBrace)?;
                    let (mut half, mut c, mut round) = (V3::new(0.5, 0.5, 0.5), V3::zero(), 0.0f32);
                    self.sdf_params(|s, key| match key {
                        "r" => { half = s.sdf_v3()?; Ok(()) }
                        "at" => { c = s.sdf_v3()?; Ok(()) }
                        "round" => { round = s.sdf_f32()?; Ok(()) }
                        other => Err(format!("unknown box field '{}'", other)),
                    })?;
                    kids.push(Shape::Box { c, half, round });
                }
                "torus" => {
                    self.expect(Tok::LBrace)?;
                    let (mut major, mut minor, mut c) = (0.7f32, 0.12f32, V3::zero());
                    self.sdf_params(|s, key| match key {
                        "r" => { major = s.sdf_f32()?; Ok(()) }
                        "t" => { minor = s.sdf_f32()?; Ok(()) }
                        "at" => { c = s.sdf_v3()?; Ok(()) }
                        other => Err(format!("unknown torus field '{}'", other)),
                    })?;
                    kids.push(Shape::Torus { c, major, minor });
                }
                "capsule" => {
                    self.expect(Tok::LBrace)?;
                    let (mut a, mut b, mut r) = (V3::zero(), V3::new(0.0, 1.0, 0.0), 0.25f32);
                    self.sdf_params(|s, key| match key {
                        "a" => { a = s.sdf_v3()?; Ok(()) }
                        "b" => { b = s.sdf_v3()?; Ok(()) }
                        "r" => { r = s.sdf_f32()?; Ok(()) }
                        other => Err(format!("unknown capsule field '{}'", other)),
                    })?;
                    kids.push(Shape::Capsule { a, b, r });
                }
                "plane" => {
                    self.expect(Tok::LBrace)?;
                    let mut y = -1.0f32;
                    self.sdf_params(|s, key| match key {
                        "y" => { y = s.sdf_f32()?; Ok(()) }
                        other => Err(format!("unknown plane field '{}'", other)),
                    })?;
                    kids.push(Shape::Plane { y });
                }
                "union" | "smooth_union" | "sub" | "isect" | "smooth_sub" | "twist" | "repeat" => {
                    let inner = self.parse_sdf_op(&kw)?;
                    kids.push(inner);
                }
                "color" => { self.expect(Tok::Colon)?; color = self.expect_hex()?; }
                "glow" => { self.expect(Tok::Colon)?; glow = self.sdf_f32()?; }
                "hud" => { self.expect(Tok::Colon)?; hud = self.expect_bool()?; }
                "eye" => { self.expect(Tok::Colon)?; eye = self.sdf_v3()?; }
                "look" => { self.expect(Tok::Colon)?; look = self.sdf_v3()?; }
                "smooth" => {
                    // top-level smoothness: rewrites the implicit union into a
                    // smooth union with the declared k
                    self.expect(Tok::Colon)?;
                    let k = self.sdf_f32()?;
                    let merged = std::mem::take(&mut kids);
                    kids.push(Shape::SmoothUnion { k, kids: merged });
                }
                other => return Err(format!("line {}:{}: unknown sdf field '{}'", self.line(), self.col(), other)),
            }
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        let root: Shape = match kids.len() {
            0 => return Err(format!("line {}:{}: sdf {}: no shapes", line, col, name)),
            1 => kids.pop().unwrap(),
            _ => Shape::Union(kids),
        };
        Ok(SdfDecl { name, root, color, glow, eye, look, hud })
    }

    /// Parse an operator block: union { ... } / smooth_union { k: .. ... } /
    /// twist { k: .. <shape> } / repeat { x: .. <shape> } etc.
    fn parse_sdf_op(&mut self, kw: &str) -> Result<crate::opt::sdf::Shape, String> {
        use crate::opt::sdf::{Shape, V3};
        self.expect(Tok::LBrace)?;
        let mut kids: Vec<Shape> = Vec::new();
        let mut k = 0.3f32;
        let mut cell = V3::zero();
        while !matches!(self.peek(), Tok::RBrace) {
            if self.is_kw("k") && matches!(self.peek_at(1), Tok::Colon) {
                self.pos += 2;
                k = self.sdf_f32()?;
                self.accept_semi_or_comma();
                continue;
            }
            if self.is_kw("x") && matches!(self.peek_at(1), Tok::Colon)
                || self.is_kw("y") && matches!(self.peek_at(1), Tok::Colon)
                || self.is_kw("z") && matches!(self.peek_at(1), Tok::Colon) {
                let axis = match self.next() {
                    Tok::Ident(a) => a,
                    _ => unreachable!(),
                };
                self.expect(Tok::Colon)?;
                let v = self.sdf_f32()?;
                match axis.as_str() {
                    "x" => cell.x = v,
                    "y" => cell.y = v,
                    _ => cell.z = v,
                }
                self.accept_semi_or_comma();
                continue;
            }
            let inner_kw = self.expect_ident()?;
            let shape = match inner_kw.as_str() {
                "sphere" => {
                    self.expect(Tok::LBrace)?;
                    let (mut r, mut c) = (1.0f32, V3::zero());
                    self.sdf_params(|s, key| match key {
                        "r" => { r = s.sdf_f32()?; Ok(()) }
                        "at" => { c = s.sdf_v3()?; Ok(()) }
                        other => Err(format!("unknown sphere field '{}'", other)),
                    })?;
                    Shape::Sphere { c, r }
                }
                "box" => {
                    self.expect(Tok::LBrace)?;
                    let (mut half, mut c, mut round) = (V3::new(0.5, 0.5, 0.5), V3::zero(), 0.0f32);
                    self.sdf_params(|s, key| match key {
                        "r" => { half = s.sdf_v3()?; Ok(()) }
                        "at" => { c = s.sdf_v3()?; Ok(()) }
                        "round" => { round = s.sdf_f32()?; Ok(()) }
                        other => Err(format!("unknown box field '{}'", other)),
                    })?;
                    Shape::Box { c, half, round }
                }
                "torus" => {
                    self.expect(Tok::LBrace)?;
                    let (mut major, mut minor, mut c) = (0.7f32, 0.12f32, V3::zero());
                    self.sdf_params(|s, key| match key {
                        "r" => { major = s.sdf_f32()?; Ok(()) }
                        "t" => { minor = s.sdf_f32()?; Ok(()) }
                        "at" => { c = s.sdf_v3()?; Ok(()) }
                        other => Err(format!("unknown torus field '{}'", other)),
                    })?;
                    Shape::Torus { c, major, minor }
                }
                "capsule" => {
                    self.expect(Tok::LBrace)?;
                    let (mut a, mut b, mut r) = (V3::zero(), V3::new(0.0, 1.0, 0.0), 0.25f32);
                    self.sdf_params(|s, key| match key {
                        "a" => { a = s.sdf_v3()?; Ok(()) }
                        "b" => { b = s.sdf_v3()?; Ok(()) }
                        "r" => { r = s.sdf_f32()?; Ok(()) }
                        other => Err(format!("unknown capsule field '{}'", other)),
                    })?;
                    Shape::Capsule { a, b, r }
                }
                "plane" => {
                    self.expect(Tok::LBrace)?;
                    let mut y = -1.0f32;
                    self.sdf_params(|s, key| match key {
                        "y" => { y = s.sdf_f32()?; Ok(()) }
                        other => Err(format!("unknown plane field '{}'", other)),
                    })?;
                    Shape::Plane { y }
                }
                "union" | "smooth_union" | "sub" | "isect" | "smooth_sub" | "twist" | "repeat" => {
                    self.parse_sdf_op(&inner_kw)?
                }
                other => return Err(format!("line {}:{}: unknown sdf shape '{}'", self.line(), self.col(), other)),
            };
            kids.push(shape);
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        let shape = match kw {
            "union" => Shape::Union(kids),
            "smooth_union" => Shape::SmoothUnion { k, kids },
            "sub" => if kids.len() == 2 {
                let b = kids.pop().unwrap();
                let a = kids.pop().unwrap();
                Shape::Sub(Box::new(a), Box::new(b))
            } else { return Err(format!("line {}:{}: sub needs exactly 2 shapes", self.line(), self.col())) },
            "isect" => if kids.len() == 2 {
                let b = kids.pop().unwrap();
                let a = kids.pop().unwrap();
                Shape::Isect(Box::new(a), Box::new(b))
            } else { return Err(format!("line {}:{}: isect needs exactly 2 shapes", self.line(), self.col())) },
            "smooth_sub" => if kids.len() == 2 {
                let b = kids.pop().unwrap();
                let a = kids.pop().unwrap();
                Shape::SmoothSub { k, a: Box::new(a), b: Box::new(b) }
            } else { return Err(format!("line {}:{}: smooth_sub needs exactly 2 shapes", self.line(), self.col())) },
            "twist" => if kids.len() == 1 {
                Shape::Twist { k, kid: Box::new(kids.pop().unwrap()) }
            } else { return Err(format!("line {}:{}: twist needs exactly 1 shape", self.line(), self.col())) },
            "repeat" => if kids.len() == 1 {
                Shape::Repeat { cell, kid: Box::new(kids.pop().unwrap()) }
            } else { return Err(format!("line {}:{}: repeat needs exactly 1 shape", self.line(), self.col())) },
            other => return Err(format!("line {}:{}: unknown sdf op '{}'", self.line(), self.col(), other)),
        };
        Ok(shape)
    }

    /// key: value / key: (a, b, c) pairs inside a shape block.
    fn sdf_params(&mut self, mut f: impl FnMut(&mut Self, &str) -> Result<(), String>) -> Result<(), String> {
        while !matches!(self.peek(), Tok::RBrace) {
            let key = self.expect_ident()?;
            self.expect(Tok::Colon)?;
            f(self, &key)?;
            self.accept_semi_or_comma();
        }
        self.expect(Tok::RBrace)?;
        Ok(())
    }

    /// Float literal in sdf context (Int or Fix tokens, optional minus).
    fn sdf_f32(&mut self) -> Result<f32, String> {
        let mut sign = 1.0f32;
        if matches!(self.peek(), Tok::Minus) {
            self.pos += 1;
            sign = -1.0;
        }
        match self.next() {
            Tok::Int(v) => Ok(sign * v as f32),
            Tok::Fix(v) => Ok(sign * (v as f32 / 256.0)),
            other => Err(format!("line {}:{}: expected number, found {:?}", self.line(), self.col(), other)),
        }
    }

    /// (x, y, z) tuple of numbers.
    fn sdf_v3(&mut self) -> Result<crate::opt::sdf::V3, String> {
        self.expect(Tok::LParen)?;
        let x = self.sdf_f32()?;
        self.expect(Tok::Comma)?;
        let y = self.sdf_f32()?;
        self.expect(Tok::Comma)?;
        let z = self.sdf_f32()?;
        self.expect(Tok::RParen)?;
        Ok(crate::opt::sdf::V3::new(x, y, z))
    }

    // ---------------- statements ----------------

    fn parse_block(&mut self) -> Result<Vec<Stmt>, String> {
        self.expect(Tok::LBrace)?;
        let mut out = Vec::new();
        while !matches!(self.peek(), Tok::RBrace) {
            out.push(self.parse_stmt()?);
        }
        self.expect(Tok::RBrace)?;
        Ok(out)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        if self.eat_kw("let") {
            let name = self.expect_ident()?;
            self.expect(Tok::Eq)?;
            let e = self.parse_expr()?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Let(name, e));
        }
        if self.eat_kw("if") { return self.parse_if(); }
        if self.eat_kw("while") {
            let cond = self.parse_expr()?;
            let body = self.parse_block()?;
            return Ok(Stmt::While(cond, body));
        }
        if self.eat_kw("for") {
            self.expect(Tok::LParen)?;
            let var = self.expect_ident()?;
            self.expect(Tok::Colon)?;
            let ent = self.expect_ident()?;
            self.expect(Tok::RParen)?;
            let body = self.parse_block()?;
            return Ok(Stmt::For(var, ent, body));
        }
        if self.eat_kw("spawn") {
            let ent = self.expect_ident()?;
            let mut inits = Vec::new();
            if matches!(self.peek(), Tok::LBrace) {
                self.pos += 1;
                while !matches!(self.peek(), Tok::RBrace) {
                    let f = self.expect_ident()?;
                    self.expect(Tok::Colon)?;
                    let e = self.parse_expr()?;
                    inits.push((f, e));
                    if matches!(self.peek(), Tok::Comma) { self.pos += 1; }
                }
                self.expect(Tok::RBrace)?;
            }
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Spawn(ent, inits));
        }
        if self.eat_kw("kill") {
            let e = self.parse_expr()?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Kill(e));
        }
        if self.eat_kw("return") {
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Return);
        }
        // v8 OPTIMIZER subsystem 5: fsm_step(#name); — the whole AI tick in
        // one statement; expanded branchlessly by opt::fsm before codegen.
        if self.is_kw("fsm_step") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let a = self.expect_atom()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::FsmStep(a));
        }
        // v10 PARTICLES: emit(#name, x, y); — sugar for a call to the
        // synthesized `emit_<name>` helper. Lowered HERE (parser level) so
        // the checker/codegen only ever see an ordinary helper call.
        if self.is_kw("emit") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let a = self.expect_atom()?;
            let mut args = Vec::new();
            while matches!(self.peek(), Tok::Comma) {
                self.pos += 1;
                args.push(self.parse_expr()?);
            }
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            let name = self.atom_names.get(a as usize).cloned()
                .ok_or_else(|| format!("line {}:{}: bad emit atom", self.line(), self.col()))?;
            return Ok(Stmt::CallStmt(format!("emit_{}", name), args));
        }
        if self.eat_kw("call") {
            let name = self.expect_ident()?;
            if matches!(self.peek(), Tok::LBracket) {
                // call table[key](ent);
                self.pos += 1;
                let key = self.parse_expr()?;
                self.expect(Tok::RBracket)?;
                self.expect(Tok::LParen)?;
                let ent = self.expect_ident()?;
                self.expect(Tok::RParen)?;
                self.expect(Tok::Semi)?;
                return Ok(Stmt::CallTable(name, Box::new(key), ent));
            }
            self.expect(Tok::LParen)?;
            let mut args = Vec::new();
            while !matches!(self.peek(), Tok::RParen) {
                args.push(self.parse_expr()?);
                if matches!(self.peek(), Tok::Comma) { self.pos += 1; }
            }
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::CallStmt(name, args));
        }
        // atom-keyword statements
        if self.is_kw("sfx") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let a = self.expect_atom()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Sfx(a));
        }
        if self.is_kw("music") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let a = self.expect_atom()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Music(a));
        }
        if self.is_kw("stop_music") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::StopMusic);
        }
        if self.is_kw("shake") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let (line, col) = (self.line(), self.col());
            let amp = match self.next() {
                Tok::Int(v) => v,
                other => return Err(format!("line {}:{}: shake amplitude must be an integer literal 0..255, found {:?}", line, col, other)),
            };
            if amp > 255 {
                return Err(format!("line {}:{}: shake amplitude out of range 0..255: {}", line, col, amp));
            }
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Shake(amp as u32));
        }
        if self.is_kw("cam3") && matches!(self.peek_at(1), Tok::LParen) {
            // v11 3D: cam3(x, y, z, yaw, pitch) — position the 3D camera.
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let x = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let y = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let z = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let yaw = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let pitch = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Cam3(x, y, z, yaw, pitch));
        }
        if self.is_kw("draw3d") && matches!(self.peek_at(1), Tok::LParen) {
            // v11 3D: draw3d(mesh, n, x, y, z, yaw, rgba) — the engine-side
            // mesh pipeline over a global vertex array (see ast::Stmt::Draw3d).
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let arr = self.expect_ident()?;
            self.expect(Tok::Comma)?;
            let n = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let x = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let y = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let z = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let yaw = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let rgba = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Draw3d { arr, n, x, y, z, yaw, rgba });
        }
        if self.is_kw("draw3di") && matches!(self.peek_at(1), Tok::LParen) {
            // v12: draw3di(verts, nv, idx, ni, x, y, z, yaw, rgba) — the
            // INDEXED mesh pass over a shared vertex buffer + index buffer.
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let verts = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let nv = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let idx = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let ni = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let x = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let y = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let z = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let yaw = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let rgba = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Draw3DI { verts, nv, idx, ni, x, y, z, yaw, rgba });
        }
        // v12 ARTICULATION: quat_aa / qmul / m4qt / m4mul / skinv — mat4 &
        // quat builtins over global arrays (see ast::Stmt for the layouts).
        // All array arguments are plain names resolved by the checker; every
        // other argument is a full expression.
        if self.is_kw("quat_aa") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let q = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let off = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let ax = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let ay = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let az = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let ang = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::QuatAA { q, off, ax, ay, az, ang });
        }
        if self.is_kw("qmul") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let d = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let doff = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let a = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let aoff = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let b = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let boff = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::QMul { d, doff, a, aoff, b, boff });
        }
        if self.is_kw("m4qt") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let m = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let moff = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let q = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let qoff = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let tx = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let ty = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let tz = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::M4QT { m, moff, q, qoff, tx, ty, tz });
        }
        if self.is_kw("m4mul") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let d = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let doff = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let a = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let aoff = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let b = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let boff = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::M4Mul { d, doff, a, aoff, b, boff });
        }
        if self.is_kw("skinv") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let v = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let voff = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let m = self.parse_arr_ref()?;
            self.expect(Tok::Comma)?;
            let moff = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let x = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let y = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let z = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::SkinV { v, voff, m, moff, x, y, z });
        }
        if self.is_kw("camera") && matches!(self.peek_at(1), Tok::LParen) {
            // v7: camera(x, y) — aim the viewport center at a world point.
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let x = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let y = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Camera(x, y));
        }
        if self.is_kw("music_vol") && matches!(self.peek_at(1), Tok::LParen) {
            // v7: music_vol(n) — master music volume 0..16 (16 = full).
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let (line, col) = (self.line(), self.col());
            let vol = match self.next() {
                Tok::Int(v) => v,
                other => return Err(format!("line {}:{}: music_vol must be an integer literal 0..16, found {:?}", line, col, other)),
            };
            if vol > 16 {
                return Err(format!("line {}:{}: music_vol out of range 0..16: {}", line, col, vol));
            }
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::MusicVol(vol as u32));
        }
        if self.is_kw("save") && matches!(self.peek_at(1), Tok::LParen) {
            // v7: save(ix, value) — persist a raw 32-bit value to SRAM slot ix.
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let ix = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let v = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Save(ix, v));
        }
        if self.is_kw("goto") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let a = self.expect_atom()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Goto(a));
        }
        if self.is_kw("draw") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let spr = self.expect_atom()?;
            self.expect(Tok::Comma)?;
            let x = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let y = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let rot = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let scale = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let rgba = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::Draw(spr, x, y, rot, scale, rgba));
        }
        if self.is_kw("draw_text") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let a = self.expect_atom()?;
            self.expect(Tok::Comma)?;
            let x = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let y = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let rgba = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::DrawText(a, x, y, rgba));
        }
        if self.is_kw("draw_num") && matches!(self.peek_at(1), Tok::LParen) {
            self.pos += 1;
            self.expect(Tok::LParen)?;
            let v = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let x = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let y = self.parse_expr()?;
            self.expect(Tok::Comma)?;
            let rgba = self.parse_expr()?;
            self.expect(Tok::RParen)?;
            self.expect(Tok::Semi)?;
            return Ok(Stmt::DrawNum(v, x, y, rgba));
        }
        // assignment: lvalue (=|+=|-=|*=) expr ;
        let target = self.parse_lvalue()?;
        let op = match self.next() {
            Tok::Eq => AssignOp::Set,
            Tok::PlusEq => AssignOp::Add,
            Tok::MinusEq => AssignOp::Sub,
            Tok::StarEq => AssignOp::Mul,
            other => return Err(format!("line {}:{}: expected assignment operator, found {:?} (bare expressions are not statements)", self.line(), self.col(), other)),
        };
        let val = self.parse_expr()?;
        self.expect(Tok::Semi)?;
        Ok(Stmt::Assign(target, op, val))
    }

    fn parse_lvalue(&mut self) -> Result<Expr, String> {
        let name = self.expect_ident()?;
        let mut e = Expr::Ident(name);
        self.parse_postfix(&mut e)?;
        Ok(e)
    }

    /// Shared postfix chain: `.field` and v11 `[index]` (array element).
    /// Used by both lvalues (`a[i] = v`, `e.trail[i] += v`) and expressions.
    fn parse_postfix(&mut self, e: &mut Expr) -> Result<(), String> {
        loop {
            if matches!(self.peek(), Tok::Dot) {
                self.pos += 1;
                let f = self.expect_ident()?;
                *e = Expr::Field(Box::new(e.clone()), f);
            } else if matches!(self.peek(), Tok::LBracket) {
                self.pos += 1;
                let ix = self.parse_expr()?;
                self.expect(Tok::RBracket)?;
                *e = Expr::Index(Box::new(e.clone()), Box::new(ix));
            } else {
                return Ok(());
            }
        }
    }

    fn parse_if(&mut self) -> Result<Stmt, String> {
        let cond = self.parse_expr()?;
        let then_b = self.parse_block()?;
        let mut else_b = Vec::new();
        if self.eat_kw("else") {
            if self.is_kw("if") {
                self.pos += 1;
                else_b.push(self.parse_if()?);
            } else {
                else_b = self.parse_block()?;
            }
        }
        Ok(Stmt::If(cond, then_b, else_b))
    }

    // ---------------- expressions (Pratt) ----------------

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_bin(1)
    }

    fn tok_prec(&self) -> Option<(u8, BinOp)> {
        let op = match self.peek() {
            Tok::OrOr => BinOp::LOr,
            Tok::AndAnd => BinOp::LAnd,
            Tok::EqEq => BinOp::Eq, Tok::NotEq => BinOp::Ne,
            Tok::Lt => BinOp::Lt, Tok::Gt => BinOp::Gt,
            Tok::Le => BinOp::Le, Tok::Ge => BinOp::Ge,
            Tok::Amp => BinOp::And, Tok::Pipe => BinOp::Or, Tok::Caret => BinOp::Xor,
            Tok::Shl => BinOp::Shl, Tok::Shr => BinOp::Shr,
            Tok::Plus => BinOp::Add, Tok::Minus => BinOp::Sub,
            Tok::Star => BinOp::Mul, Tok::Slash => BinOp::Div, Tok::Percent => BinOp::Mod,
            _ => return None,
        };
        let prec = match op {
            BinOp::LOr => 1,
            BinOp::LAnd => 2,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => 3,
            BinOp::And | BinOp::Or | BinOp::Xor => 4,
            BinOp::Shl | BinOp::Shr => 5,
            BinOp::Add | BinOp::Sub => 6,
            BinOp::Mul | BinOp::Div | BinOp::Mod => 7,
        };
        Some((prec, op))
    }

    fn parse_bin(&mut self, min_prec: u8) -> Result<Expr, String> {
        let mut lhs = self.parse_unary()?;
        loop {
            if let Some((prec, op)) = self.tok_prec() {
                if prec >= min_prec {
                    self.pos += 1;
                    let rhs = self.parse_bin(prec + 1)?;
                    lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
                    continue;
                }
            }
            break;
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        if matches!(self.peek(), Tok::Minus) {
            self.pos += 1;
            let e = self.parse_unary()?;
            // constant-fold negative literals
            return Ok(match e {
                Expr::IntLit(v) => Expr::IntLit((-(v as i64)) as u32),
                Expr::FixLit(v) => Expr::FixLit(-v),
                other => Expr::Unary(UnOp::Neg, Box::new(other)),
            });
        }
        if matches!(self.peek(), Tok::Not) {
            self.pos += 1;
            let e = self.parse_unary()?;
            return Ok(match e {
                Expr::BoolLit(b) => Expr::BoolLit(!b),
                other => Expr::Unary(UnOp::Not, Box::new(other)),
            });
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        // Capture the position BEFORE consuming: `next()` advances `self.pos`,
        // so calling line()/col() inside the match arm reports the FOLLOWING
        // token's position (e.g. `px = ;` pointed at the next line's `}`
        // instead of at the `;`).
        let (tok_line, tok_col) = (self.line(), self.col());
        let mut e = match self.next() {
            Tok::Int(v) => Expr::IntLit(v),
            Tok::Fix(v) => Expr::FixLit(v),
            Tok::Atom(a) => Expr::Atom(a),
            Tok::LParen => {
                let inner = self.parse_expr()?;
                self.expect(Tok::RParen)?;
                inner
            }
            Tok::Ident(name) => {
                if name == "true" { Expr::BoolLit(true) }
                else if name == "false" { Expr::BoolLit(false) }
                else if matches!(self.peek(), Tok::LParen) {
                    self.pos += 1;
                    let mut args = Vec::new();
                    while !matches!(self.peek(), Tok::RParen) {
                        args.push(self.parse_expr()?);
                        if matches!(self.peek(), Tok::Comma) { self.pos += 1; }
                    }
                    self.expect(Tok::RParen)?;
                    Expr::Call(name, args)
                } else {
                    Expr::Ident(name)
                }
            }
            other => return Err(format!("line {}:{}: unexpected token in expression: {:?}", tok_line, tok_col, other)),
        };
        // postfix field access + v11 array indexing
        self.parse_postfix(&mut e)?;
        Ok(e)
    }
}
