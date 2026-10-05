//! LILA semantic checker.
//! - context-sensitive #Atom resolution into per-domain u16 tables
//! - Q24.8 / int / bool / ang / entity typing with implicit int→fixed widening
//! - bit-width inference for `u`/`s` fields (min 4 bits)
//! - typed-field registry (bit offsets in dense rows / cold rows / globals)
//! - sprite baking: polygon → triangle fan → 14-byte quantized vertices
//! - anim compression: RDP simplify + Catmull-Rom → Hermite control points

use crate::ast::*;
use std::collections::HashMap;

mod assets;
mod expr;
mod layout;
mod stmt;

// internal wiring (the split modules call each other through here)
pub(crate) use layout::{bits_for_uint, build_fields, wrap_entity_fields};
pub(crate) use assets::{bake_sprite, compress_anim};
pub(crate) use stmt::check_block;
use expr::expr_type;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VT { Fixed, Int, Bool, Ang, Ent(usize) }

pub const GLOBAL_ENT: u8 = 0xFF;

#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub name: String,
    pub vt: VT,        // Fixed | Int | Bool | Ang (never Ent)
    pub signed: bool,  // sN fields: sign-extended loads (bits|0x80 on wire)
    pub bits: u8,
    pub cold: bool,
    pub off_bits: u16, // bit offset within dense row (or cold row, or globals block)
    pub c_off_bits: u16,
    pub default: i32,  // raw value bits
}

#[derive(Debug, Clone)]
pub struct EntityInfo {
    pub name: String,
    pub fields: Vec<FieldInfo>,
    pub row_bytes: u16,     // dense row size
    pub cold_bytes: u16,    // cold row size
    pub max_live: u32,
}

/// v11 ARRAYS: a validated fixed-capacity array. `ent` = GLOBAL_ENT for
/// game-wide arrays, else the owning entity type. Elements are RAW 32-bit
/// cells (the VM value model) in a dedicated flat-memory plane — arrays
/// never touch the bit-packed field layout or the SoA repack.
#[derive(Debug, Clone)]
pub struct ArrInfo {
    pub name: String,
    pub elem: VT,
    pub cap: u32,
    pub ent: u8,
}

#[derive(Debug, Clone)]
pub struct TypedField {
    pub ent: u8,       // entity index or GLOBAL_ENT
    pub field: u16,    // index into entity fields (or globals)
    pub off_bits: u16, // dense offset (globals use this too)
    pub signed: bool,
    pub bits: u8,
    pub vt: VT,
    pub cold: bool,
    pub c_off_bits: u16,
}

#[derive(Debug, Clone, Copy)]
pub struct V16 {
    pub x: i16, pub y: i16, // i16 pixel-space (sprite-local)
    pub u: u16, pub v: u16, // u16 texture/procedural coords
    pub rgba: u32,
    pub tok: u16, // bit0 additive, bits1-2 pattern, bits3+ reserved
}

#[derive(Debug, Clone)]
pub struct SpriteInfo {
    pub atom: u16,
    pub verts: Vec<V16>, // triangle soup
}

#[derive(Debug, Clone)]
pub struct VoiceInfo {
    pub wave: u8, // 0 square 1 tri 2 noise 3 saw
    pub vol: u8,
    pub rows: Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct MusicInfo {
    pub atom: u16,
    pub bpm: u32,
    pub voices: Vec<VoiceInfo>,
}

#[derive(Debug, Clone)]
pub struct SfxInfo {
    pub atom: u16,
    pub wave: u8,
    pub freq: i32,
    pub sweep: i32,
    pub decay: i32,
    pub vol: u32,
    /// v8 OPTIMIZER subsystem 4: build-time fitted synthesis program.
    /// kind: 0..3 legacy waves (square/tri/noise/saw), 4 = FM, 5 = additive.
    /// `fitted` is None for legacy programs (identical v1..v2 wire format).
    pub kind: u8,
    pub fitted: Option<crate::opt::audio::FittedModel>,
}

/// v8 OPTIMIZER subsystem 5: a validated, compiled-into-tables state machine.
#[derive(Debug, Clone)]
pub struct FsmInfo {
    pub atom: u16,
    pub name: String,
    pub entity: usize,   // owning entity type index
    pub states: Vec<String>,
    /// per state: (predicate expr over the owning entity's fields, target state index)
    pub guards: Vec<Vec<(Expr, usize)>>,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct AnimSeg {
    pub t0: u16,
    pub p0: i32, pub m0: i32,
    pub p1: i32, pub m1: i32,
}

#[derive(Debug, Clone)]
pub struct AnimInfo {
    pub atom: u16,
    pub duration: u16,
    pub segs: Vec<AnimSeg>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FnKind { Init, Update0, UpdateEnt, Draw0, DrawEnt, Helper, Scene, Table }

#[derive(Debug, Clone)]
pub struct FnInfo {
    pub name: String,
    pub params: Vec<(String, VT)>,
    pub body: Vec<Stmt>,
    pub kind: FnKind,
    pub entity: Option<usize>, // for Ent param
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct TableInfo {
    pub name: String,
    pub entries: Vec<(u8, u16)>, // key -> fn index
}

pub struct Ctx {
    pub atom_names: Vec<String>,
    pub game: GameDecl,
    pub sound: Vec<u16>,       // atom ids in order
    pub keys: Vec<u16>,
    pub texts: Vec<(u16, String)>,
    pub scenes: Vec<(u16, u16)>, // atom id -> fn index
    pub sprites: Vec<SpriteInfo>,
    pub music: Vec<MusicInfo>,
    pub sfx: Vec<SfxInfo>,
    pub anims: Vec<AnimInfo>,
    pub entities: Vec<EntityInfo>,
    pub globals: Vec<FieldInfo>,
    pub globals_bits: u32,
    pub arrays: Vec<ArrInfo>,
    pub typed_fields: Vec<TypedField>,
    pub fns: Vec<FnInfo>,
    pub tables: Vec<TableInfo>,
    pub fsms: Vec<FsmInfo>,
    // index maps (reborrowed by codegen via CtxRefs::from_ctx)
    pub ent_index: HashMap<String, usize>,
    pub global_index: HashMap<String, usize>,
    pub tf_index: HashMap<(u8, String), usize>,
    pub arr_index: HashMap<(u8, String), usize>,
    pub fn_index: HashMap<String, usize>,
    pub scene_atom_ids: Vec<u16>,
    pub sprite_atoms: Vec<u16>,
    pub music_atoms: Vec<u16>,
    pub anim_atoms: Vec<u16>,
}

pub(crate) struct FnEnv {
    pub(crate) locals: Vec<(String, VT)>,
}

pub fn check(prog: &Program, atom_names: Vec<String>, fitted_audio: &std::collections::HashMap<u16, crate::opt::audio::FittedSfx>) -> Result<Ctx, String> {
    let game = prog.game.clone().unwrap_or(GameDecl {
        title: String::new(), width: 512, height: 512, wrap: true,
        world_w: None, world_h: None, capacity: None, optimize: false,
    });
    // v7 effective simulation bounds: world defaults to the screen size
    // (v1 behavior — byte-compatible); a bigger world is a scrolling world
    // driven by camera(x, y).
    let world_w = game.world_w.unwrap_or(game.width);
    let world_h = game.world_h.unwrap_or(game.height);
    for (dim, what) in [(world_w, "world_w"), (world_h, "world_h")] {
        if !(16..=32768).contains(&dim) {
            return Err(format!("game: {} must be 16..=32768 (got {})", what, dim));
        }
    }
    if game.wrap && (!world_w.is_power_of_two() || !world_h.is_power_of_two()) {
        return Err("game: wrap=true requires power-of-two world dims (world_w/world_h, default = width/height; bitwise AND wrap)".into());
    }
    if prog.entities.len() > 64 {
        return Err(format!("{} entity types declared (max 64 — build profiles raise max_types up to 64; see scripts/build_engine.sh)",
            prog.entities.len()));
    }
    // v14 format guards: screen dims ride the header as u16 and feed every
    // projection path; anim segs ride a u8 count AND the runtime reserves
    // MAX_SEGS=32 slots per curve (ANIM_STRIDE). Both were silently
    // truncatable before — loud now, mirroring the runtime's own contract.
    if !(16..=32768).contains(&game.width) || !(16..=32768).contains(&game.height) {
        return Err(format!(
            "game: width/height must be 16..=32768 (got {}x{}) — the .libyte header stores u16 dims",
            game.width, game.height));
    }
    // v2 capacity: per-type live-entity slots the game is built for.
    // Structural bounds: entity slot index is a 20-bit handle field and the
    // emitted schema stores max_live as u16. The engine build profile is the
    // real authority — a runtime with fewer slots rejects the file loudly
    // (ERR_LIMIT_ENTITIES) instead of clamping.
    let cap_ent: u32 = game.capacity.unwrap_or(512);
    if !(16..=65535).contains(&cap_ent) {
        return Err(format!(
            "game.capacity.entities: {} out of range 16..=65535 (engine default is 512; \
             build a bigger profile with scripts/build_engine.sh)", cap_ent));
    }

    let atom_name = |a: u16| -> String {
        atom_names.get(a as usize).cloned().unwrap_or_else(|| format!("#{}", a))
    };

    // ---- v8 OPTIMIZER subsystem 5: fsm declarations (structure only; the
    // entity binding + guard validation happen after the entities build) ----
    let mut fsms: Vec<FsmInfo> = Vec::new();
    {
        let mut fsm_names: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for fd in &prog.fsms {
            let name = atom_name(fd.atom);
            if fd.states.len() < 2 || fd.states.len() > 16 {
                return Err(format!("line {}: fsm {}: needs 2..=16 states (got {})", fd.line, name, fd.states.len()));
            }
            if !fsm_names.insert(fd.name.as_str()) {
                return Err(format!("line {}: duplicate fsm '{}'", fd.line, fd.name));
            }
            let mut state_seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for (sname, _) in &fd.states {
                if !state_seen.insert(sname.as_str()) {
                    return Err(format!("line {}: fsm {}: duplicate state '{}'", fd.line, name, sname));
                }
            }
            for (_, gs) in &fd.states {
                if gs.len() > 4 {
                    return Err(format!("line {}: fsm {}: max 4 guards per state (bit-plane width)", fd.line, name));
                }
            }
            fsms.push(FsmInfo {
                atom: fd.atom, name: fd.name.clone(), entity: usize::MAX,
                states: fd.states.iter().map(|(s, _)| s.clone()).collect(),
                guards: Vec::new(), line: fd.line,
            });
        }
    }

    // ---- domains ----
    let sound = prog.sound_atoms.clone();
    let keys = prog.key_atoms.clone();
    if keys.len() > 32 {
        return Err("key {}: max 32 key atoms (4-byte input bitmask)".into());
    }
    let texts = prog.text_atoms.clone();
    let scene_atom_ids: Vec<u16> = prog.scene_atoms.iter().map(|(a, _)| *a).collect();

    // dup check within domains
    let dup = |v: &[u16], dom: &str| -> Option<String> {
        let mut seen = std::collections::HashSet::new();
        for a in v {
            if !seen.insert(a) {
                return Some(format!("duplicate atom #{} in {} domain", atom_name(*a), dom));
            }
        }
        None
    };
    if let Some(e) = dup(&sound, "sound") { return Err(e); }
    if let Some(e) = dup(&keys, "key") { return Err(e); }

    // ---- entities & globals: widths, defaults, offsets ----
    // cam_x / cam_y are engine registers (readable fixed expressions) —
    // user declarations of those names are rejected to keep them unambiguous.
    for g in &prog.globals {
        if g.name == "cam_x" || g.name == "cam_y" {
            return Err(format!("global '{}' is reserved (engine camera registers; read via cam_x/cam_y expressions)", g.name));
        }
    }
    for fd in &prog.fns {
        for (pn, _) in &fd.params {
            if pn == "cam_x" || pn == "cam_y" {
                return Err(format!("line {}: param '{}' is reserved (engine camera registers)", fd.line, pn));
            }
        }
    }
    let mut entities = Vec::new();
    for (ei, ed) in prog.entities.iter().enumerate() {
        let mut fields = ed.fields.clone();
        // v8 OPTIMIZER subsystem 5: an entity driving an fsm gets its state
        // bitfield synthesized into the schema — minimum width for the state
        // count (5 states -> u3), packed into the SoA row like any field.
        if let Some(fname) = &ed.fsm {
            let fi = fsms.iter().position(|f| &f.name == fname)
                .ok_or_else(|| format!("entity {}: unknown fsm '{}'", ed.name, fname))?;
            if fsms[fi].entity != usize::MAX {
                return Err(format!(
                    "fsm '{}': already drives entity {} (one fsm owns one entity type)",
                    fsms[fi].name, prog.entities[fsms[fi].entity].name));
            }
            fsms[fi].entity = ei;
            let nstates = fsms[fi].states.len();
            let bits = bits_for_uint((nstates - 1) as u64).max(1);
            fields.push(FieldDecl {
                name: "fsm_state".into(), ty: Ty::UInt(bits),
                default_int: Some(0), default_fix: None, default_bool: None, cold: false,
            });
        }
        // v10: per-type capacity override (particles pools declare their own
        // slot count; everything else uses game.capacity).
        let (ent, _) = build_fields(&fields, None, &ed.name, cap_ent, ed.max_live_override)?;
        entities.push(ent);
    }
    let (globals_ent, globals_bits) = build_fields(&prog.globals, None, "global", 0, None)?;
    // v13 guard: the globals block is the 128-byte register-free gap
    // 0x80..0x100 in page 0 (GLOBALS_BYTES in core.zig). Before this check
    // existed, a game with >1024 bits of globals silently overran the
    // shake/cam3/proj registers and corrupted the engine (found by LILA
    // DRIVE: ~1500 bits of globals zeroed the shake decay into an overflow
    // panic). Fail the build with the numbers instead.
    // Mirrors runtime/src/core.zig GLOBALS_BYTES (128) — the Rust side has
    // no view of the Zig constants, so the two MUST be kept in sync; the
    // zig test_map suite pins the layout from the engine side.
    const GLOBALS_BYTES_SYNC: u32 = 128;
    if globals_bits > GLOBALS_BYTES_SYNC * 8 {
        return Err(format!(
            "globals block overflow: the game needs {} bits ({} bytes) but the engine reserves {} bytes ({} bits) at page-0 0x80..0x100 — move bulk data (colors, tables, scratch) into arrays",
            globals_bits,
            globals_bits / 8,
            GLOBALS_BYTES_SYNC,
            GLOBALS_BYTES_SYNC * 8
        ));
    }
    let globals = globals_ent.fields;

    // ---- v11 ARRAYS registry ----
    // Element type checked + capacity validated; ids assigned in declaration
    // order (global arrays first, then per-entity in entity order). Name
    // uniqueness is per scope and never collides with a field name.
    fn arr_elem_vt(ty: &Ty, name: &str, line: u32) -> Result<VT, String> {
        match ty {
            Ty::Fixed => Ok(VT::Fixed),
            Ty::UInt(_) | Ty::SInt(_) => Ok(VT::Int),
            Ty::Bool => Ok(VT::Bool),
            Ty::Ang => Ok(VT::Ang),
            Ty::Entity(_) => Err(format!(
                "line {}: array '{}': entity elements are forbidden (handle-free design)",
                line, name
            )),
        }
    }
    let mut arrays: Vec<ArrInfo> = Vec::new();
    for ad in &prog.garrs {
        let vt = arr_elem_vt(&ad.elem, &ad.name, ad.line)?;
        arrays.push(ArrInfo { name: ad.name.clone(), elem: vt, cap: ad.cap, ent: GLOBAL_ENT });
    }
    for (ei, ed) in prog.entities.iter().enumerate() {
        for ad in &ed.arrs {
            let vt = arr_elem_vt(&ad.elem, &ad.name, ad.line)?;
            arrays.push(ArrInfo { name: ad.name.clone(), elem: vt, cap: ad.cap, ent: ei as u8 });
        }
    }
    {
        let mut seen = std::collections::HashSet::new();
        for a in &arrays {
            if !seen.insert((a.ent, a.name.clone())) {
                return Err(format!("line {}: duplicate array '{}' in the same scope", a.cap, a.name));
            }
        }
        // an array must not shadow a field name in its own scope (the tf
        // index is keyed by (ent, name) too — ambiguity would be unsound)
        for a in &arrays {
            if a.ent == GLOBAL_ENT {
                if globals.iter().any(|g| g.name == a.name) {
                    return Err(format!("global: array '{}' collides with a global field name", a.name));
                }
            } else {
                let e = &entities[a.ent as usize];
                if e.fields.iter().any(|f| f.name == a.name) {
                    return Err(format!("entity {}: array '{}' collides with a field name", e.name, a.name));
                }
            }
        }
    }
    let arr_index: HashMap<(u8, String), usize> = {
        let mut m = HashMap::new();
        for (i, a) in arrays.iter().enumerate() {
            m.insert((a.ent, a.name.clone()), i);
        }
        m
    };

    // per-entity field name uniqueness
    for e in &entities {
        let mut seen = std::collections::HashSet::new();
        for f in &e.fields {
            if !seen.insert(f.name.clone()) {
                return Err(format!("entity {}: duplicate field '{}'", e.name, f.name));
            }
        }
    }
    {
        let mut seen = std::collections::HashSet::new();
        for f in &globals {
            if !seen.insert(f.name.clone()) {
                return Err(format!("global: duplicate field '{}'", f.name));
            }
        }
    }

    // ---- typed-field registry ----
    let mut typed_fields = Vec::new();
    for (ei, e) in entities.iter().enumerate() {
        for (fi, f) in e.fields.iter().enumerate() {
            typed_fields.push(TypedField {
                ent: ei as u8,
                field: fi as u16,
                off_bits: f.off_bits,
                signed: f.signed,
                bits: f.bits,
                vt: f.vt,
                cold: f.cold,
                c_off_bits: f.c_off_bits,
            });
        }
    }
    for (fi, f) in globals.iter().enumerate() {
        typed_fields.push(TypedField {
            ent: GLOBAL_ENT,
            field: fi as u16,
            off_bits: f.off_bits,
            signed: f.signed,
            bits: f.bits,
            vt: f.vt,
            cold: false, // globals always dense
            c_off_bits: 0,
        });
    }
    let tf_index: HashMap<(u8, String), usize> = {
        let mut m = HashMap::new();
        for (i, tf) in typed_fields.iter().enumerate() {
            let fname = if tf.ent == GLOBAL_ENT {
                globals[tf.field as usize].name.clone()
            } else {
                entities[tf.ent as usize].fields[tf.field as usize].name.clone()
            };
            m.insert((tf.ent, fname), i);
        }
        m
    };
    let global_index: HashMap<String, usize> = {
        let mut m = HashMap::new();
        for (i, g) in globals.iter().enumerate() {
            m.insert(g.name.clone(), i);
        }
        m
    };

    // ---- validate fsm guards against the owning entity's context ----
    // Guards are written against the entity's fields BARE (implicit self):
    // `hp <= 0`, `key(#fire)`. For validation they are re-written onto a
    // synthetic param; opt::fsm rewrites them onto the REAL param name of
    // whichever fn executes fsm_step at expansion time.
    let ent_index: HashMap<String, usize> = prog.entities.iter()
        .enumerate().map(|(i, e)| (e.name.clone(), i)).collect();
    // Guard predicates are plain field/key/dist expressions — fsm_step cannot
    // appear inside one, so validation passes an EMPTY fsm list (also avoids
    // aliasing fsms while it is being mutated below).
    let no_fsms: Vec<FsmInfo> = Vec::new();
    for f in fsms.iter_mut() {
        if f.entity == usize::MAX {
            return Err(format!(
                "fsm '{}': no entity drives it (bind with `fsm: {}` in an entity block)",
                f.name, f.name));
        }
        let ei = f.entity;
        let fd = prog.fsms.iter().find(|d| d.atom == f.atom).unwrap();
        let dummy_fn = FnInfo {
            name: format!("<fsm:{}>", f.name), params: vec![("__ent".into(), VT::Ent(ei))],
            body: Vec::new(), kind: FnKind::UpdateEnt, entity: Some(ei), line: f.line,
        };
        let env = vec![("__ent".to_string(), VT::Ent(ei))];
        let refs = CtxRefs {
            atom_names: &atom_names, entities: &entities, globals: &globals, global_index: &global_index,
            ent_index: &ent_index, fn_index: &HashMap::new(), fns: &Vec::new(),
            tables: &Vec::new(), sound: &sound, keys: &keys, texts: &texts,
            scene_atom_ids: &scene_atom_ids, sprites: &Vec::new(),
            music_atoms: &Vec::new(), anim_atoms: &Vec::new(),
            fsms: &no_fsms,
            arrays: &arrays, arr_index: &arr_index,
        };
        let mut guards: Vec<Vec<(Expr, usize)>> = Vec::new();
        for (sname, gs) in &fd.states {
            let mut out = Vec::new();
            for (pred, target) in gs {
                let ti = f.states.iter().position(|s| s == target)
                    .ok_or_else(|| format!(
                        "line {}: fsm {}: state '{}' transitions to unknown state '{}'",
                        fd.line, f.name, sname, target))?;
                let rewritten = wrap_entity_fields(pred, ei, &entities);
                let vt = expr_type(&rewritten, &env, &dummy_fn, &refs).map_err(|e| {
                    format!("fsm {}: guard in state '{}': {}", f.name, sname, e)
                })?;
                if vt != VT::Bool {
                    return Err(format!("fsm {}: guard in state '{}' must be bool, got {:?}", f.name, sname, vt));
                }
                out.push((pred.clone(), ti));
            }
            guards.push(out);
        }
        f.guards = guards;
    }

    // ---- fn kinds ----
    // v14 wire guard: the fn count rides a u16 — 65,536+ fns would wrap
    // to zero and desync the whole file. Profile limits (default 128, up
    // to 512) are the loader's business; this is the format's hard floor.
    if prog.fns.len() > 65535 {
        return Err(format!("{} fns declared (max 65535 — u16 count on the wire)", prog.fns.len()));
    }
    let mut referenced: HashMap<String, FnKind> = HashMap::new();
    for (_, fname) in &prog.scene_atoms { referenced.insert(fname.clone(), FnKind::Scene); }
    for t in &prog.tables {
        for (_, fname) in &t.entries { referenced.insert(fname.clone(), FnKind::Table); }
    }

    let mut fns = Vec::new();
    for fd in &prog.fns {
        let mut params: Vec<(String, VT)> = Vec::new();
        for (pn, pt) in &fd.params {
            let vt = match pt {
                Ty::Fixed => VT::Fixed,
                Ty::Bool => VT::Bool,
                Ty::Ang => VT::Ang,
                Ty::UInt(_) | Ty::SInt(_) => VT::Int,
                Ty::Entity(name) => {
                    match ent_index.get(name) {
                        Some(i) => VT::Ent(*i),
                        None => return Err(format!("line {}: unknown entity type '{}' for param '{}'", fd.line, name, pn)),
                    }
                }
            };
            params.push((pn.clone(), vt));
        }
        if fd.params.len() > 3 {
            return Err(format!("line {}: fn '{}' has too many params (max 3)", fd.line, fd.name));
        }
        let kind = if let Some(k) = referenced.get(&fd.name) {
            *k
        } else {
            match fd.name.as_str() {
                "init" | "start" => {
                    if !params.is_empty() {
                        return Err(format!("line {}: init() takes no params", fd.line));
                    }
                    FnKind::Init
                }
                "update" => {
                    if params.is_empty() { FnKind::Update0 }
                    else if params.len() == 1 && matches!(params[0].1, VT::Ent(_)) { FnKind::UpdateEnt }
                    else { return Err(format!("line {}: update() must be () or (e: Entity)", fd.line)); }
                }
                "draw" => {
                    if params.is_empty() { FnKind::Draw0 }
                    else if params.len() == 1 && matches!(params[0].1, VT::Ent(_)) { FnKind::DrawEnt }
                    else { return Err(format!("line {}: draw() must be () or (e: Entity)", fd.line)); }
                }
                _ => FnKind::Helper,
            }
        };
        // param shape validation per kind
        match kind {
            FnKind::Scene => if !params.is_empty() {
                return Err(format!("line {}: scene fn '{}' must take no params", fd.line, fd.name));
            },
            FnKind::Table => {
                if params.len() != 1 || !matches!(params[0].1, VT::Ent(_)) {
                    return Err(format!("line {}: table fn '{}' must take one entity param", fd.line, fd.name));
                }
            }
            _ => {}
        }
        let entity = params.iter().find_map(|(_, vt)| match vt {
            VT::Ent(i) => Some(*i),
            _ => None,
        });
        // helper max 2 params; entity fns exactly 1
        if kind == FnKind::Helper && params.len() > 2 {
            return Err(format!("line {}: helper '{}' max 2 params", fd.line, fd.name));
        }
        if matches!(kind, FnKind::UpdateEnt | FnKind::DrawEnt) && params.len() != 1 {
            return Err(format!("line {}: entity fn '{}' takes exactly one entity param", fd.line, fd.name));
        }
        // table fns of one table must share entity type
        fns.push(FnInfo { name: fd.name.clone(), params, body: fd.body.clone(), kind, entity, line: fd.line });
    }

    // fn uniqueness: keyed by (name, entity param). Entity fns (`update(e: T)`,
    // `draw(e: T)`) may overload the same name across entity types.
    {
        let mut seen = std::collections::HashSet::new();
        for f in &fns {
            if !seen.insert((f.name.clone(), f.entity)) {
                return Err(format!("line {}: duplicate fn '{}' for the same param type", f.line, f.name));
            }
        }
    }
    // fn_index for REFERENCES (call/scene/table): only unambiguous names.
    // Overloaded names (e.g. multiple `update(e: T)`) are not referenceable.
    let fn_index: HashMap<String, usize> = {
        let mut m = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for f in &fns { *counts.entry(f.name.clone()).or_insert(0) += 1; }
        for (i, f) in fns.iter().enumerate() {
            if counts[&f.name] == 1 {
                m.insert(f.name.clone(), i);
            }
        }
        m
    };

    // ---- tables ----
    let mut tables = Vec::new();
    for td in &prog.tables {
        // format guard: the entry count rides a u8 on the wire
        if td.entries.len() > 255 {
            return Err(format!("table {}: {} entries (max 255 — u8 count on the wire)",
                td.name, td.entries.len()));
        }
        let mut entries = Vec::new();
        let mut ent_ty: Option<usize> = None;
        for (key, fname) in &td.entries {
            let fi = *fn_index.get(fname)
                .ok_or_else(|| format!("table {}: unknown fn '{}'", td.name, fname))?;
            if fns[fi].kind != FnKind::Table {
                return Err(format!("table {}: fn '{}' is not a table fn (needs one entity param)", td.name, fname));
            }
            let et = fns[fi].entity
                .ok_or_else(|| format!("table {}: fn '{}' lost entity param", td.name, fname))?;
            if let Some(prev) = ent_ty {
                if prev != et {
                    return Err(format!("table {}: fns must share one entity type", td.name));
                }
            } else {
                ent_ty = Some(et);
            }
            entries.push((*key, fi as u16));
        }
        entries.sort_by_key(|(k, _)| *k);
        tables.push(TableInfo { name: td.name.clone(), entries });
    }

    // ---- scenes ----
    let mut scenes = Vec::new();
    for (atom, fname) in &prog.scene_atoms {
        let fi = *fn_index.get(fname)
            .ok_or_else(|| format!("scene {}: unknown fn '{}'", atom_name(*atom), fname))?;
        if fns[fi].kind != FnKind::Scene {
            return Err(format!("scene {}: fn '{}' must take no params", atom_name(*atom), fname));
        }
        scenes.push((*atom, fi as u16));
    }

    // ---- sprites (bake) ----
    let mut sprites = Vec::new();
    if prog.sprites.len() > 65535 {
        return Err(format!("{} sprites declared (max 65535 — u16 count on the wire)", prog.sprites.len()));
    }
    for sd in &prog.sprites {
        sprites.push(bake_sprite(sd)?);
    }

    // ---- music ----
    let mut music = Vec::new();
    for md in &prog.music {
        // runtime contract: 4 voice kinds (square/tri/noise/saw) and the
        // 40B voice slot fits 16 steps — both mirrored here, loud
        if md.voices.len() > 4 {
            return Err(format!("music #{}: {} voices (max 4: square/tri/noise/saw)",
                md.atom, md.voices.len()));
        }
        let mut voices = Vec::new();
        for v in &md.voices {
            let wave = match v.wave { Wave::Square => 0, Wave::Tri => 1, Wave::Noise => 2, Wave::Saw => 3 };
            let mut rows = Vec::new();
            for s in &v.steps {
                let r = match s {
                    Step::Note(n) => ((*n as u16) << 4) | v.vol as u16,
                    Step::Hold => 0xFFE0 | v.vol as u16,
                    Step::Silence => 0xFFF0 | v.vol as u16,
                    Step::Hit => (0u16 << 4) | v.vol as u16,
                };
                rows.push(r);
            }
            voices.push(VoiceInfo { wave, vol: v.vol, rows });
        }
        if !(30..=400).contains(&md.bpm) {
            return Err(format!("music {}: bpm {} out of range 30..400", atom_name(md.atom), md.bpm));
        }
        music.push(MusicInfo { atom: md.atom, bpm: md.bpm, voices });
    }

    // ---- sfx ----
    let mut sfx = Vec::new();
    for sd in &prog.sfx {
        if !sound.contains(&sd.atom) {
            return Err(format!("sfx #{}: atom not declared in sound {{}} block", atom_name(sd.atom)));
        }
        // v8 OPTIMIZER subsystem 4: a build-time-fitted program replaces the
        // legacy wave params (main.rs ran the WAV analysis BEFORE checking;
        // the runtime consumes only the baked numbers).
        if let Some(fit) = fitted_audio.get(&sd.atom) {
            let kind = match &fit.model {
                crate::opt::audio::FittedModel::Fm(_) => 4u8,
                crate::opt::audio::FittedModel::Add(_) => 5u8,
            };
            sfx.push(SfxInfo {
                atom: sd.atom, wave: kind, kind, freq: 0, sweep: 0, decay: 1, vol: 0,
                fitted: Some(fit.model.clone()),
            });
            continue;
        }
        let wave = match sd.wave { Wave::Square => 0, Wave::Tri => 1, Wave::Noise => 2, Wave::Saw => 3 };
        sfx.push(SfxInfo {
            atom: sd.atom, wave, kind: wave, freq: sd.freq, sweep: sd.sweep,
            decay: sd.decay.max(1), vol: sd.vol, fitted: None,
        });
    }

    // ---- anims ----
    let mut anims = Vec::new();
    for ad in &prog.anims {
        let info = compress_anim(ad)?;
        // runtime sync: ANIM_STRIDE reserves MAX_SEGS=32 segments per curve
        // (mirrors GLOBALS_BYTES_SYNC below — keep the two in lockstep)
        if info.segs.len() > 32 {
            return Err(format!("anim #{}: RDP compression left {} segments (max 32 — the engine's ANIM_STRIDE reserves 32; add keys or simplify the curve)",
                ad.atom, info.segs.len()));
        }
        anims.push(info);
    }

    // ---- statement/expression checking per fn ----
    for f in &fns {
        let mut env = FnEnv { locals: f.params.clone() };
        check_block(&f.body, &mut env, f, &CtxRefs {
            atom_names: &atom_names, entities: &entities, globals: &globals, global_index: &global_index, ent_index: &ent_index,
            fn_index: &fn_index, fns: &fns, tables: &tables, sound: &sound, keys: &keys,
            texts: &texts, scene_atom_ids: &scene_atom_ids,
            sprites: &sprites, music_atoms: &music.iter().map(|m| m.atom).collect::<Vec<_>>(),
            anim_atoms: &anims.iter().map(|a| a.atom).collect::<Vec<_>>(), fsms: &fsms,
            arrays: &arrays, arr_index: &arr_index,
        })?;
    }

    let sprite_atoms: Vec<u16> = sprites.iter().map(|s| s.atom).collect();
    let music_atoms: Vec<u16> = music.iter().map(|m| m.atom).collect();
    let anim_atoms: Vec<u16> = anims.iter().map(|a| a.atom).collect();
    Ok(Ctx {
        atom_names, game, sound, keys, texts, scenes, sprites, music, sfx, anims,
        entities, globals, globals_bits, arrays, typed_fields, fns, tables, fsms,
        ent_index, global_index, tf_index, arr_index, fn_index,
        scene_atom_ids, sprite_atoms, music_atoms, anim_atoms,
    })
}

impl Ctx {
    /// Borrow view for codegen: typing rules without re-computation.
    pub fn refs(&self) -> CtxRefs<'_> {
        CtxRefs {
            atom_names: &self.atom_names,
            entities: &self.entities,
            globals: &self.globals,
            global_index: &self.global_index,
            ent_index: &self.ent_index,
            fn_index: &self.fn_index,
            fns: &self.fns,
            tables: &self.tables,
            sound: &self.sound,
            keys: &self.keys,
            texts: &self.texts,
            scene_atom_ids: &self.scene_atom_ids,
            sprites: &self.sprites,
            music_atoms: &self.music_atoms,
            anim_atoms: &self.anim_atoms,
            fsms: &self.fsms,
            arrays: &self.arrays,
            arr_index: &self.arr_index,
        }
    }
}


// ---------------- statement/expression checking ----------------

pub struct CtxRefs<'a> {
    atom_names: &'a Vec<String>,
    entities: &'a Vec<EntityInfo>,
    globals: &'a Vec<FieldInfo>,
    global_index: &'a HashMap<String, usize>,
    ent_index: &'a HashMap<String, usize>,
    fn_index: &'a HashMap<String, usize>,
    fns: &'a Vec<FnInfo>,
    tables: &'a Vec<TableInfo>,
    sound: &'a Vec<u16>,
    keys: &'a Vec<u16>,
    texts: &'a Vec<(u16, String)>,
    scene_atom_ids: &'a Vec<u16>,
    sprites: &'a Vec<SpriteInfo>,
    music_atoms: &'a Vec<u16>,
    anim_atoms: &'a Vec<u16>,
    fsms: &'a Vec<FsmInfo>,
    /// v11 ARRAYS: registry + (scope, name) -> id. Shared by expression
    /// typing, lvalue typing, alen() and draw3d validation.
    pub arrays: &'a Vec<ArrInfo>,
    pub arr_index: &'a HashMap<(u8, String), usize>,
}

impl<'a> CtxRefs<'a> {
    /// Resolve an array base expression to its registry id. `name[i]` names a
    /// global array; `e.name[i]` names an array of e's entity type. Anything
    /// else (expressions, unknown names, wrong scope) is a hard error.
    pub fn resolve_arr(&self, base: &Expr, env: &[(String, VT)]) -> Result<usize, String> {
        match base {
            Expr::Ident(name) => self.arr_index
                .get(&(GLOBAL_ENT, name.clone()))
                .copied()
                .ok_or_else(|| format!("unknown global array '{}' (entity arrays need the handle prefix, e.g. e.{})", name, name)),
            Expr::Field(inner, fname) => {
                let v = match &**inner {
                    Expr::Ident(v) => v,
                    _ => return Err("array base must be a variable or variable.field".into()),
                };
                let ei = match env.iter().find(|(n, _)| n == v).map(|(_, vt)| *vt) {
                    Some(VT::Ent(ei)) => ei,
                    _ => return Err(format!("'{}' is not an entity variable", v)),
                };
                self.arr_index.get(&(ei as u8, fname.clone())).copied().ok_or_else(|| {
                    format!("entity {} has no array '{}'", self.entities[ei].name, fname)
                })
            }
            _ => Err("array base must be a name or entity handle field".into()),
        }
    }

    /// Compile-time bounds PROOF for literal indices — the Lila flavor of
    /// safety: out-of-range literals are build errors, dynamic indices wrap.
    fn check_index(&self, arr_id: usize, ix: &Expr, env: &[(String, VT)], f: &FnInfo) -> Result<VT, String> {
        let a = &self.arrays[arr_id];
        let ivt = expr_type(ix, env, f, self)?;
        if ivt != VT::Int {
            return Err(format!("line {}: array '{}' index must be int, got {:?}", f.line, a.name, ivt));
        }
        if let Expr::IntLit(v) = ix {
            if (*v as u64) >= a.cap as u64 {
                return Err(format!(
                    "line {}: array '{}' index {} out of bounds 0..{} (proven at compile time)",
                    f.line, a.name, v, a.cap
                ));
            }
        }
        Ok(a.elem)
    }
}

impl<'a> CtxRefs<'a> {
    /// Public typing entry (used by codegen).
    pub fn expr_type(&self, e: &Expr, env: &[(String, VT)], f: &FnInfo) -> Result<VT, String> {
        // Optimizer intrinsics (INTR_BIT01 / INTR_FSM_NEXT / INTR_COLLIDE_MASK)
        // are injected by the passes AFTER checking; codegen still types their
        // SURROUNDING expressions through here, so they carry their proven
        // result type (all three yield Int at the VM level).
        if matches!(e, Expr::Intrin(..)) {
            return Ok(VT::Int);
        }
        expr_type(e, env, f, self)
    }
    fn aname(&self, a: u16) -> String {
        self.atom_names.get(a as usize).cloned().unwrap_or_else(|| format!("#{}", a))
    }
    fn domain_pos(&self, dom: &[u16], a: u16, dom_name: &str) -> Result<u16, String> {
        dom.iter().position(|&x| x == a)
            .map(|p| p as u16)
            .ok_or_else(|| format!("#{} is not declared in the {} domain", self.aname(a), dom_name))
    }
}

