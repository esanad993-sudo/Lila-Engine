//! LILA AST — raw parse output (pre-type-check).

#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    Fixed,
    UInt(u8),  // bit width 1..=16 (u1 = bool-ish but distinct)
    SInt(u8),
    Bool,
    Ang,
    Entity(String),
}

#[derive(Debug, Clone)]
pub struct GameDecl {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub wrap: bool,
    /// v7: simulation bounds (world > screen = scrolling world with camera).
    /// None = world == screen (v1 behavior, byte-compatible).
    pub world_w: Option<u32>,
    pub world_h: Option<u32>,
    /// v2: game-declared capacity — entity slots per type the game needs.
    /// None = engine default (512). Written into a v2 .libyte header; the
    /// runtime rejects the file (ERR_LIMIT_ENTITIES) if its build profile
    /// is smaller, so capacity mismatches are loud, never silent.
    pub capacity: Option<u32>,
    /// v8 OPTIMIZER mandate: `game { optimize: true }` opts this game into
    /// the full compile-time pipeline (bit-width synthesis + cache-line SoA
    /// repacking). Off by default: legacy games keep byte-identical output.
    pub optimize: bool,
}

#[derive(Debug, Clone)]
pub struct FieldDecl {
    pub name: String,
    pub ty: Ty,
    pub default_int: Option<i64>, // raw literal (fixed already Q24.8 when fixed)
    pub default_fix: Option<i32>,
    pub default_bool: Option<bool>,
    pub cold: bool,
}

/// v11 ARRAYS (TArray/List<T> analog) — fixed-capacity, zero-alloc, i32-cell
/// aggregates declared with `arr <ty>[<cap>] <name>;` inside `global { }`
/// (game-wide arrays: meshes, waypoint paths, grids) or an entity block
/// (per-instance arrays: trails, inventories). Capacity is a compile-time
/// constant: the checker PROVES literal indices in range, the runtime wraps
/// dynamic indices (mask if power-of-two cap, else modulo) — the same
/// "no trap paths" contract as div-by-zero and spawn budgets.
#[derive(Debug, Clone)]
pub struct ArrDecl {
    pub name: String,
    pub elem: Ty, // scalar element type (Entity arrays are forbidden)
    pub cap: u32, // element count, 1..=16384
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct EntityDecl {
    pub name: String,
    pub fields: Vec<FieldDecl>,
    /// v8 OPTIMIZER: fsm this entity drives. The checker synthesizes a
    /// minimum-width `fsm_state` bitfield into the schema (e.g. 5 states ->
    /// u3) and opt::fsm bakes the transition table.
    pub fsm: Option<String>,
    /// v10 PREFABS (Unity prefab analog): `entity FastDrone from Drone {
    /// ... }` inherits every parent field at compile time; the child's own
    /// blocks override defaults (same name = same type, new value) and may
    /// append fields. Resolved by synth::expand before checking — the
    /// checker/codegen/runtime never see inheritance at all.
    pub from: Option<String>,
    /// v10 PARTICLES: per-entity slot capacity override (the `(192)` in
    /// `particles #boomfx (192) { ... }`). The checker uses it instead of
    /// game.capacity for this type's max_live.
    pub max_live_override: Option<u32>,
    /// v11 ARRAYS: per-instance arrays (`arr fixed[16] trail;` inside the
    /// entity block). Stored in a dedicated per-type plane OUTSIDE the 32B
    /// dense rows — arrays never touch the bit-packed field layout.
    pub arrs: Vec<ArrDecl>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pat { Flat, Noise }

#[derive(Debug, Clone)]
pub struct PolyDecl {
    pub pts: Vec<(i32, i32)>,
    pub fill: u32,
    pub grad: Option<u32>,
    pub add: bool,
    pub pat: Pat,
}

#[derive(Debug, Clone)]
pub struct SpriteDecl {
    pub atom: u16,
    pub polys: Vec<PolyDecl>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Wave { Square, Tri, Noise, Saw }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Step {
    Note(u8), // semitone index 0..=95 (0 = C1)
    Hold,
    Silence,
    Hit, // noise hit marker 'x'
}

#[derive(Debug, Clone)]
pub struct VoiceDecl {
    pub wave: Wave,
    pub steps: Vec<Step>,
    pub vol: u8,
}

#[derive(Debug, Clone)]
pub struct MusicDecl {
    pub atom: u16,
    pub bpm: u32,
    pub voices: Vec<VoiceDecl>,
}

#[derive(Debug, Clone)]
pub struct SfxDecl {
    pub atom: u16,
    pub wave: Wave,
    pub freq: i32,
    pub sweep: i32,
    pub decay: i32,
    pub vol: u32,
    /// v8 OPTIMIZER: build-time audio-synthesis source. When present, the
    /// compiler READS the .wav AT BUILD TIME (dev-time only — the player's
    /// runtime never sees a file), fits an O(1) f(t) model to the PCM and
    /// bakes the fitted parameters into the .libyte. The string is an asset
    /// PATH, not a runtime string — the no-runtime-strings rule is intact:
    /// nothing here survives into the bytecode except numbers.
    pub from: Option<String>,
}

/// v10 PARTICLES — Niagara/VFX-Graph analog as a COMPILE-TIME emitter.
/// A `particles #boomfx (192) { ... }` block is not a runtime subsystem:
/// synth::expand lowers it into an ordinary entity type + `emit_X` helper +
/// `update(X)` + `draw(X)` fn trio, and the rest of the pipeline (SoA
/// packing, spawn budgets, integration, draw pass) treats them like
/// hand-written code. Zero runtime bytes, zero new opcodes.
#[derive(Debug, Clone)]
pub struct ParticleDecl {
    pub atom: u16,
    /// live-particle slot capacity
    pub cap: u32,
    /// sprite drawn per particle (atom in the sprite domain)
    pub sprite: u16,
    /// particle lifetime in frames (ttl counts down from here)
    pub life: u32,
    /// initial speed, Q24.8 px/frame
    pub speed_fix: i32,
    /// speed jitter 0..1 in Q8 (0 = constant speed)
    pub jitter_q8: u32,
    /// velocity retention per frame in Q8 (256 = no drag)
    pub drag_q8: u32,
    /// downward acceleration, Q24.8 px/frame^2
    pub grav_fix: i32,
    /// particles spawned per emit()
    pub burst: u32,
    /// base RGB tint (alpha computed from ttl when fade)
    pub color: u32,
    /// scale + alpha fade linearly over the lifetime
    pub fade: bool,
}

/// v8 OPTIMIZER SUBSYSTEM 5 — AI state machines.
/// Parsed verbatim; the compiler (opt::fsm) enumerates states, compiles every
/// guard to a predicate program, and synthesizes the bit-plane transition
/// table T[state][events] that the runtime consults with ONE shift+mask.
#[derive(Debug, Clone)]
pub struct FsmDecl {
    pub atom: u16,
    pub name: String,
    /// (state name, guard list) — guard = (predicate expr, target state name)
    pub states: Vec<(String, Vec<(Expr, String)>)>,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct AnimDecl {
    pub atom: u16,
    pub keys: Vec<(u32, i32)>, // (t frames, value Q24.8)
}

#[derive(Debug, Clone)]
pub struct TableDecl {
    pub name: String,
    pub entries: Vec<(u8, String)>,
}

/// v8 OPTIMIZER intrinsic — emitted ONLY by compiler passes (never parsed).
/// It is the unchecked-intrinsics channel between the metaprogramming passes
/// and codegen: the pass has already proven operand types, so codegen uses
/// the fixed result type below instead of re-deriving it.
/// The checker rejects these if one ever leaks into user code (defensive).
pub const INTR_BIT01: u16 = 1;      // (bool) -> int 0/1        [no-op at VM level]
pub const INTR_FSM_NEXT: u16 = 2;   // (em, state) -> next state via bit-plane table
pub const INTR_COLL_MASK: u16 = 3;  // (cx, cy) -> u32 live-slot bitmask (Galois lanes)
pub const INTR_SEL: u16 = 4;        // (cond, then_val, else_val) -> branchless cond-move

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp { Neg, Not }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    And, Or, Xor, Shl, Shr,
    Eq, Ne, Lt, Gt, Le, Ge,
    LAnd, LOr,
}

#[derive(Debug, Clone)]
pub enum Expr {
    IntLit(u32),
    FixLit(i32),
    BoolLit(bool),
    Atom(u16),
    Ident(String),
    Field(Box<Expr>, String),
    /// v11 ARRAYS: `name[i]` (global array) or `e.name[i]` (entity array).
    /// The checker resolves the base to an array id and enforces the element
    /// type; literal indices are bounds-PROVEN at compile time.
    Index(Box<Expr>, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>), // builtins: key sin cos ang fixed int rand anim count swept_hit
    /// optimizer-only intrinsic (see INTR_* consts). aux = opcode context
    /// (fsm table index / entity type), consts carry folded immediates.
    Intrin(u16, u16, Vec<Expr>, Vec<i32>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AssignOp { Set, Add, Sub, Mul }

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(String, Expr),
    Assign(Expr, AssignOp, Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>), // else-if chains nest If in the else vec
    While(Expr, Vec<Stmt>),
    For(String, String, Vec<Stmt>), // (var, entity type, body)
    Spawn(String, Vec<(String, Expr)>),
    Kill(Expr),
    Sfx(u16),
    Music(u16),
    StopMusic,
    Shake(u32),                               // screen-shake amplitude (px, literal)
    /// v11 3D: `cam3(x, y, z, yaw, pitch);` — position the 3D camera (fixed
    /// xyz + ang yaw/pitch). Registers live in page 0 of flat memory, so
    /// snapshots/rewind/rollback capture them with the standard 2MB memcpy.
    Cam3(Expr, Expr, Expr, Expr, Expr),
    /// v11 3D: `draw3d(mesh, n, x, y, z, yaw, rgba);` — software mesh pass.
    /// `mesh` names a GLOBAL array holding packed local-space vertices
    /// (3 fixed per vertex). The ENGINE transforms (yaw+translate), views,
    /// perspective-projects, backface-culls, near-clips, depth-sorts and
    /// rasterizes — the Burst pattern: Lila orchestrates, native code crunches.
    Draw3d {
        arr: String,
        n: Expr,
        x: Expr,
        y: Expr,
        z: Expr,
        yaw: Expr,
        rgba: Expr,
    },
    /// v12 ARTICULATION — mat4/quat builtins ON TOP OF ARRAYS. A mat4 is 16
    /// consecutive fixed cells (column-major, translation in cells 12..14),
    /// a quat is 4 consecutive cells (x, y, z, w) of a GLOBAL array. Each
    /// statement is ONE native VM opcode: the game orchestrates the rig,
    /// the engine crunches the math (the Burst pattern that keeps Lila
    /// games lightweight without a floating-point unit anywhere).
    ///   quat_aa(q, off, ax, ay, az, ang) — q[off..off+3] = axis-angle quat
    ///   qmul(d, doff, a, aoff, b, boff)  — d = a*b (Hamilton)
    ///   m4qt(m, moff, q, qoff, tx, ty, tz) — M = T(t) * R(q) in one op
    ///   m4mul(d, doff, a, aoff, b, boff) — d = a*b (d may alias a or b)
    ///   skinv(v, voff, m, moff, x, y, z) — v[voff..+2] = M * (x,y,z,1)
    QuatAA {
        q: String,
        off: Expr,
        ax: Expr,
        ay: Expr,
        az: Expr,
        ang: Expr,
    },
    QMul {
        d: String,
        doff: Expr,
        a: String,
        aoff: Expr,
        b: String,
        boff: Expr,
    },
    M4QT {
        m: String,
        moff: Expr,
        q: String,
        qoff: Expr,
        tx: Expr,
        ty: Expr,
        tz: Expr,
    },
    M4Mul {
        d: String,
        doff: Expr,
        a: String,
        aoff: Expr,
        b: String,
        boff: Expr,
    },
    SkinV {
        v: String,
        voff: Expr,
        m: String,
        moff: Expr,
        x: Expr,
        y: Expr,
        z: Expr,
    },
    /// v12 INDEXED MESH PASS: `draw3di(verts, nv, idx, ni, x, y, z, yaw,
    /// rgba);` — verts is a global array of 3 fixed per vertex (shared
    /// vertex buffer, animated by skinv), idx a global array of vertex
    /// indices (3 per tri). The engine transforms/views/projects/culls/
    /// sorts/rasterizes exactly like draw3d, but triangles address the
    /// shared vertices — the GPU vertex/index-buffer vocabulary, and the
    /// shape every articulated/skinned model is submitted in.
    Draw3DI {
        verts: String,
        nv: Expr,
        idx: String,
        ni: Expr,
        x: Expr,
        y: Expr,
        z: Expr,
        yaw: Expr,
        rgba: Expr,
    },
    Camera(Expr, Expr),                       // v7: aim the viewport at a world point (fixed, fixed)
    MusicVol(u32),                            // v7: master music volume 0..16 (literal)
    Save(Expr, Expr),                         // v7: save(ix, value) -> SRAM slot 0..63
    Goto(u16),
    Draw(u16, Expr, Expr, Expr, Expr, Expr), // sprite, x, y, rot, scale, rgba
    DrawText(u16, Expr, Expr, Expr),        // text atom, x, y, rgba
    DrawNum(Expr, Expr, Expr, Expr),        // value, x, y, rgba
    CallStmt(String, Vec<Expr>),
    CallTable(String, Box<Expr>, String), // table name, key expr, entity var name
    Return,
    /// v8 OPTIMIZER: `fsm_step(#name);` — parsed form. The checker validates
    /// placement (entity fn, entity drives that fsm); opt::fsm REWRITES it
    /// into the branchless event-mask + FSM_NEXT form before codegen.
    FsmStep(u16),
    /// optimizer-only: iterate the live slots named by a Galois lane mask.
    /// (var, entity type, mask local name, body). Never parsed; produced by
    /// opt::physics when it compiles a collision loop to immediate bitmasks.
    ForMask(String, String, String, Vec<Stmt>),
}

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<(String, Ty)>,
    pub body: Vec<Stmt>,
    pub line: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Program {
    pub game: Option<GameDecl>,
    pub sound_atoms: Vec<u16>,
    pub key_atoms: Vec<u16>,
    pub text_atoms: Vec<(u16, String)>,
    pub scene_atoms: Vec<(u16, String)>, // (atom, fn name)
    pub sprites: Vec<SpriteDecl>,
    pub music: Vec<MusicDecl>,
    pub sfx: Vec<SfxDecl>,
    pub anims: Vec<AnimDecl>,
    pub entities: Vec<EntityDecl>,
    pub globals: Vec<FieldDecl>,
    pub tables: Vec<TableDecl>,
    pub fsms: Vec<FsmDecl>,
    /// v10 PARTICLES: declarative emitters (see ParticleDecl). Lowered to
    /// entities + fns by synth::expand before the checker runs.
    pub particles: Vec<ParticleDecl>,
    /// v8 OPTIMIZER SUBSYSTEM 2 — SDF shape programs parsed from `sdf` blocks.
    /// Dev-time assets: `lilac build` ignores them (the .libyte never carries
    /// them), `lilac shader` symbolically compiles them to WGSL.
    pub sdfs: Vec<crate::opt::sdf::SdfDecl>,
    pub fns: Vec<FnDecl>,
    /// v11 ARRAYS: game-wide arrays from `global { arr ... }`.
    pub garrs: Vec<ArrDecl>,
}
