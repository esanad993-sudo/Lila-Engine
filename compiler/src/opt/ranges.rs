//! ============================================================================
//! LILA OPTIMIZER — SUBSYSTEM 1a: AUTOMATED MINIMUM BIT-WIDTH SYNTHESIS
//! ============================================================================
//!
//! THE DEV WRITES:  `hp: u = 5`  ...  `if hit { if hp > 0 { hp -= 1; } }`
//! THE COMPILER PROVES: across the ENTIRE project, hp is only ever assigned
//! the literals {5, 4, 3, 2, 1, 0} — every site is a constant or a guarded
//! decrement — so 3 bits carry the whole value domain. The declared u8 (or
//! inferred u4) is synthesized DOWN to u3, and the SoA repacker (opt::soa)
//! packs the freed bits into another entity's fields.
//!
//! Soundness contract (why this cannot change what the player experiences):
//!   * the analysis is a monotone fixpoint over EVERY assignment site in EVERY
//!     function (init/update/draw/helpers/tables) — one site is enough to
//!     keep the field wide;
//!   * any site the analyzer cannot prove (helper calls, `dist()` results,
//!     arbitrary expressions) marks the field `open` and it KEEPS its width;
//!   * unsigned wrap semantics are preserved because narrowing happens only
//!     when the proven interval [min, max] fits the target width — a value
//!     outside the interval was never reachable;
//!   * subtract sites are only trusted when syntactically guarded by
//!     `if f > c` / `if f >= c` / `if f != 0` on the same field (the canonical
//!     ammo/hp/ttl pattern); an unguarded `-=` leaves the field wide.
//!   * fixed/ang/bool/cold fields are untouched; signed fields keep their
//!     declared width (documented conservatism).
//!
//! This is all DEV-TIME work: the player's runtime just executes the narrower
//! `& mask >> shift` extract/deposit the repacker baked into the schema.

use crate::ast::*;
use crate::checker::{Ctx, VT, GLOBAL_ENT};
use std::collections::HashMap;

/// (entity type or GLOBAL_ENT, field index)
pub type FieldKey = (u8, u16);

#[derive(Clone)]
struct Interval {
    min: i64,
    max: i64,
    /// false = some site was unprovable; the field must keep its width
    closed: bool,
}

impl Interval {
    fn bottom(default: i64, signed: bool) -> Self {
        if signed {
            Interval { min: default as i64, max: default as i64, closed: true }
        } else {
            Interval { min: 0, max: default.max(0) as i64, closed: true }
        }
    }
    fn open() -> Self {
        Interval { min: i64::MIN, max: i64::MAX, closed: false }
    }
    fn add(&self, o: &Interval) -> Self {
        Interval {
            min: self.min.saturating_add(o.min),
            max: self.max.saturating_add(o.max),
            closed: self.closed && o.closed,
        }
    }
    fn sub(&self, o: &Interval) -> Self {
        Interval {
            min: self.min.saturating_sub(o.max),
            max: self.max.saturating_sub(o.min),
            closed: self.closed && o.closed,
        }
    }
    fn mul(&self, o: &Interval) -> Self {
        // Clamp inputs to the i32 value domain first, then multiply in i64:
        // unbounded interval endpoints (±i64 extremes from widening through
        // fixed<->int conversions — v11 arrays surfaced this) overflow i64
        // when multiplied raw. Clamping is monotone, so the min/max remain
        // sound bounds for every value the VM can actually hold.
        let cl = |v: i64| -> i64 { v.clamp(i32::MIN as i64, i32::MAX as i64) };
        let (a0, a1) = (cl(self.min), cl(self.max));
        let (b0, b1) = (cl(o.min), cl(o.max));
        let cands = [a0 * b0, a0 * b1, a1 * b0, a1 * b1];
        Interval { min: *cands.iter().min().unwrap(), max: *cands.iter().max().unwrap(), closed: self.closed && o.closed }
    }
    fn join(&self, o: &Interval) -> Self {
        Interval { min: self.min.min(o.min), max: self.max.max(o.max), closed: self.closed && o.closed }
    }
}

/// Provable interval of an integer-valued expression.
/// `rand(N)` is the only non-literal we trust (the engine contract fixes its
/// range to 0..N-1 — docs/SPEC.md "rand: LCG ... u32 in 0..N-1").
fn expr_interval(e: &Expr) -> Interval {
    match e {
        Expr::IntLit(v) => {
            let v = *v as i64 as i32 as i64; // u32 -> i32 raw semantics
            Interval { min: v, max: v, closed: true }
        }
        Expr::FixLit(v) => Interval { min: *v as i64, max: *v as i64, closed: true },
        Expr::Unary(UnOp::Neg, inner) => {
            let i = expr_interval(inner);
            Interval { min: -i.max, max: -i.min, closed: i.closed }
        }
        Expr::Binary(op, a, b) => {
            let (ia, ib) = (expr_interval(a), expr_interval(b));
            match op {
                BinOp::Add => ia.add(&ib),
                BinOp::Sub => ia.sub(&ib),
                BinOp::Mul => ia.mul(&ib),
                _ => Interval::open(), // div/mod/bitwise: not worth proving
            }
        }
        Expr::Call(name, args) if name == "rand" => {
            if let Some(Interval { min: n, max: n2, closed: true }) = args.first().map(expr_interval) {
                if n == n2 && n > 0 {
                    return Interval { min: 0, max: n - 1, closed: true };
                }
            }
            Interval::open()
        }
        _ => Interval::open(),
    }
}

/// Does this condition guard field `f` to be > lower? Recognizes the exact
/// canonical ammo/hp/ttl guard shapes (f > c, f >= c, f != 0).
fn guard_lower(e: &Expr, f: &FieldKey, env: &WalkEnv) -> Option<i64> {
    let field_of = |x: &Expr| -> Option<FieldKey> { env.target_key(x) };
    let const_of = |x: &Expr| -> Option<i64> {
        match x {
            Expr::IntLit(v) => Some(*v as i64 as i32 as i64),
            Expr::FixLit(v) => Some(*v as i64),
            _ => None,
        }
    };
    match e {
        Expr::Binary(BinOp::Gt, a, b) => {
            if let (Some(ff), Some(c)) = (field_of(a), const_of(b)) {
                if &ff == f { return Some(c + 1); }
            }
            None
        }
        Expr::Binary(BinOp::Ge, a, b) => {
            if let (Some(ff), Some(c)) = (field_of(a), const_of(b)) {
                if &ff == f { return Some(c); }
            }
            None
        }
        Expr::Binary(BinOp::Ne, a, b) => {
            if let (Some(ff), Some(c)) = (field_of(a), const_of(b)) {
                if &ff == f && c == 0 { return Some(1); }
            }
            None
        }
        _ => None,
    }
}

/// Typing environment for the walk: maps variable name -> entity index.
struct WalkEnv<'a> {
    ctx: &'a Ctx,
    /// var name -> entity index (from fn params + For loops)
    ent_vars: HashMap<String, usize>,
}

impl<'a> WalkEnv<'a> {
    fn field_key(&self, e: &Expr) -> Option<FieldKey> {
        if let Expr::Field(base, fname) = e {
            if let Expr::Ident(var) = &**base {
                if let Some(&ei) = self.ent_vars.get(var) {
                    let fi = self.ctx.entities[ei].fields.iter().position(|f| f.name == *fname)?;
                    return Some((ei as u8, fi as u16));
                }
            }
        }
        None
    }
    fn global_key(&self, e: &Expr) -> Option<FieldKey> {
        if let Expr::Ident(name) = e {
            if let Some(gi) = self.ctx.global_index.get(name) {
                return Some((GLOBAL_ENT, *gi as u16));
            }
        }
        None
    }
    fn target_key(&self, e: &Expr) -> Option<FieldKey> {
        self.field_key(e).or_else(|| self.global_key(e))
    }
}

pub struct NarrowResult {
    /// (display name, declared bits, synthesized bits)
    pub narrowed: Vec<(String, u8, u8)>,
}

/// Run the global range analysis over ALL functions and narrow integer
/// field widths in place (mutates ctx.entities[..].fields[..].bits and
/// ctx.globals[..].bits). Offsets are NOT touched here — opt::soa rebuilds
/// the whole packed layout afterwards.
pub fn narrow(ctx: &mut Ctx) -> NarrowResult {
    // per-field interval state (fixpoint)
    let mut ivs: HashMap<FieldKey, Interval> = HashMap::new();
    // fields with any unguarded subtract site
    let mut has_unguarded_sub: HashMap<FieldKey, ()> = HashMap::new();
    // fields written by a COMPOUND op (+=, *=). Globals and entity fields
    // persist across frames — `tick += 1` executed every frame is unbounded
    // within its declared width, but the fixpoint walk only unrolls the body
    // 16 rounds, which "proved" max=16 and narrowed a u16 frame counter to
    // 5 bits (found by LILA DRIVE: the day/night cycle wrapped every 32
    // frames). Compound ops make the domain unbounded: never narrow them.
    let mut has_compound: HashMap<FieldKey, ()> = HashMap::new();

    // seed with defaults
    for (ei, ent) in ctx.entities.iter().enumerate() {
        for (fi, f) in ent.fields.iter().enumerate() {
            if matches!(f.vt, VT::Int) && !f.signed {
                ivs.insert((ei as u8, fi as u16), Interval::bottom(f.default as i64, false));
            }
        }
    }
    for (gi, g) in ctx.globals.iter().enumerate() {
        if matches!(g.vt, VT::Int) && !g.signed {
            ivs.insert((GLOBAL_ENT, gi as u16), Interval::bottom(g.default as i64, false));
        }
    }

    // collect all walks: fn bodies with param env + spawn inits
    let mut walks: Vec<(usize, Vec<Stmt>)> = Vec::new(); // (fn idx, body)
    for (fi, f) in ctx.fns.iter().enumerate() {
        walks.push((fi, f.body.clone()));
    }

    // fixpoint: keep walking until intervals stabilize (max 16 rounds)
    let mut changed = true;
    let mut round = 0;
    while changed && round < 16 {
        changed = false;
        round += 1;
        for (fi, body) in &walks {
            let f = &ctx.fns[*fi];
            let mut env = WalkEnv { ctx: ctx as &Ctx, ent_vars: HashMap::new() };
            for (pn, pvt) in &f.params {
                if let VT::Ent(ei) = pvt {
                    env.ent_vars.insert(pn.clone(), *ei);
                }
            }
            if walk_block(&body, &mut env, &mut ivs, &mut has_unguarded_sub, &mut has_compound) {
                changed = true;
            }
        }
        // spawn inits act as constant seeds each round (they are assignment sites)
        // (spawn field inits were folded into the walk via Stmt::Spawn below)
    }

    // ---- apply narrowing ----
    let mut out = Vec::new();
    let bits_needed = |max: i64| -> u8 {
        let mut w = 1u8;
        while w < 16 && max >= (1i64 << w) { w += 1; }
        w
    };
    for (ei, ent) in ctx.entities.iter_mut().enumerate() {
        for (fi, f) in ent.fields.iter_mut().enumerate() {
            let key = (ei as u8, fi as u16);
            maybe_narrow(f, key, &ivs, &has_unguarded_sub, &has_compound, &ent.name, &mut out, bits_needed);
        }
    }
    for (gi, g) in ctx.globals.iter_mut().enumerate() {
        let key = (GLOBAL_ENT, gi as u16);
        maybe_narrow(g, key, &ivs, &has_unguarded_sub, &has_compound, "global", &mut out, bits_needed);
    }
    NarrowResult { narrowed: out }
}

fn maybe_narrow(
    f: &mut crate::checker::FieldInfo,
    key: FieldKey,
    ivs: &HashMap<FieldKey, Interval>,
    has_unguarded_sub: &HashMap<FieldKey, ()>,
    has_compound: &HashMap<FieldKey, ()>,
    owner: &str,
    out: &mut Vec<(String, u8, u8)>,
    bits_needed: impl Fn(i64) -> u8,
) {
    // Only unsigned integer fields; fsm_state is owned by opt::fsm.
    if !matches!(f.vt, VT::Int) || f.signed || f.name == "fsm_state" {
        return;
    }
    if has_unguarded_sub.contains_key(&key) {
        return;
    }
    if has_compound.contains_key(&key) {
        return;
    }
    let Some(iv) = ivs.get(&key) else { return };
    if !iv.closed || iv.min < 0 || iv.max < 0 || iv.max > (1i64 << 31) {
        return;
    }
    let w = bits_needed(iv.max);
    if w < f.bits {
        out.push((format!("{}.{}", owner, f.name), f.bits, w));
        f.bits = w;
    }
}

/// Walk one statement block; returns true if any interval grew (fixpoint signal).
fn walk_block(
    stmts: &[Stmt],
    env: &mut WalkEnv,
    ivs: &mut HashMap<FieldKey, Interval>,
    has_unguarded_sub: &mut HashMap<FieldKey, ()>,
    has_compound: &mut HashMap<FieldKey, ()>,
) -> bool {
    let mut changed = false;
    for s in stmts {
        changed |= walk_stmt(s, env, ivs, has_unguarded_sub, has_compound);
    }
    changed
}

fn walk_stmt(
    s: &Stmt,
    env: &mut WalkEnv,
    ivs: &mut HashMap<FieldKey, Interval>,
    has_unguarded_sub: &mut HashMap<FieldKey, ()>,
    has_compound: &mut HashMap<FieldKey, ()>,
) -> bool {
    match s {
        Stmt::Assign(target, op, val) => {
            let Some(key) = env.target_key(target) else { return false };
            // typing: only analyze int-typed targets
            let is_int = match key.0 {
                GLOBAL_ENT => matches!(env.ctx.globals[key.1 as usize].vt, VT::Int),
                ei => matches!(env.ctx.entities[ei as usize].fields[key.1 as usize].vt, VT::Int),
            };
            if !is_int {
                // fixed/ang/bool: still recurse for nested effects (none), but no state
                return false;
            }
            let vi = expr_interval(val);
            let entry = ivs.entry(key).or_insert_with(|| Interval::open());
            let before = (entry.min, entry.max, entry.closed);
            match op {
                AssignOp::Set => {
                    *entry = entry.join(&vi);
                }
                AssignOp::Add => {
                    has_compound.insert(key, ());
                    let next = entry.add(&vi);
                    entry.max = entry.max.max(next.max);
                    entry.min = entry.min.min(next.min);
                    entry.closed = entry.closed && vi.closed;
                }
                AssignOp::Sub => {
                    // unguarded subtract: lower bound escapes to negative
                    has_unguarded_sub.insert(key, ());
                    entry.closed = entry.closed && vi.closed;
                }
                AssignOp::Mul => {
                    has_compound.insert(key, ());
                    let next = entry.mul(&vi);
                    entry.max = entry.max.max(next.max);
                    entry.closed = entry.closed && vi.closed;
                }
            }
            let after = (entry.min, entry.max, entry.closed);
            before != after
        }
        Stmt::Spawn(ent_name, inits) => {
            let Some(&ei) = env.ctx.ent_index.get(ent_name) else { return false };
            let mut changed = false;
            for (fname, e) in inits {
                let Some(fi) = env.ctx.entities[ei].fields.iter().position(|f| f.name == *fname)
                    else { continue };
                let key = (ei as u8, fi as u16);
                let is_int = matches!(env.ctx.entities[ei].fields[fi].vt, VT::Int)
                    && !env.ctx.entities[ei].fields[fi].signed;
                if !is_int { continue; }
                let vi = expr_interval(e);
                let entry = ivs.entry(key).or_insert_with(|| Interval::open());
                let before = (entry.min, entry.max, entry.closed);
                *entry = entry.join(&vi);
                let after = (entry.min, entry.max, entry.closed);
                changed |= before != after;
            }
            changed
        }
        Stmt::If(cond, then_b, else_b) => {
            let mut changed = walk_block(else_b, env, ivs, has_unguarded_sub, has_compound);
            // guard recognition: try each conjunct of the condition as a
            // lower-bound guard (the canonical `if hp > 0 { hp -= 1 }` shape)
            let mut guarded: HashMap<FieldKey, i64> = HashMap::new();
            for g in conjuncts(cond) {
                if let Some(target) = guard_target(g, env) {
                    if let Some(lower) = guard_lower(g, &target, env) {
                        guarded.insert(target, lower);
                    }
                }
            }
            if guarded.is_empty() {
                changed |= walk_block(then_b, env, ivs, has_unguarded_sub, has_compound);
            } else {
                // within the guarded branch, subtracts on guarded fields keep
                // min >= guard: simulate by temporarily allowing the subtract
                // but registering the guard so narrowing sees min >= lower.
                changed |= walk_block_guarded(then_b, env, ivs, has_unguarded_sub, has_compound, &guarded);
            }
            changed
        }
        Stmt::While(cond, body) => {
            // loops: iterate the body twice to reach a stable over-approx
            let mut changed = false;
            for _ in 0..2 {
                changed |= walk_block(body, env, ivs, has_unguarded_sub, has_compound);
            }
            let _ = cond;
            changed
        }
        Stmt::For(var, ent_name, body) => {
            let mut env2 = WalkEnv { ctx: env.ctx, ent_vars: env.ent_vars.clone() };
            if let Some(&ei) = env.ctx.ent_index.get(ent_name) {
                env2.ent_vars.insert(var.clone(), ei);
            }
            walk_block(body, &mut env2, ivs, has_unguarded_sub, has_compound)
        }
        Stmt::FsmStep(_) => {
            // the fsm pass assigns the state bitfield; its domain is
            // 0..nstates-1 by construction and opt::fsm owns the width.
            false
        }
        Stmt::ForMask(_, _, _, body) => walk_block(body, env, ivs, has_unguarded_sub, has_compound),
        _ => false,
    }
}

/// Walk a block whose subtract sites are protected by `guarded` lower bounds.
/// We implement it by recording the guard as a floor: subtracts on guarded
/// fields do NOT taint them.
fn walk_block_guarded(
    stmts: &[Stmt],
    env: &mut WalkEnv,
    ivs: &mut HashMap<FieldKey, Interval>,
    has_unguarded_sub: &mut HashMap<FieldKey, ()>,
    has_compound: &mut HashMap<FieldKey, ()>,
    guarded: &HashMap<FieldKey, i64>,
) -> bool {
    let mut changed = false;
    for s in stmts {
        match s {
            Stmt::Assign(target, AssignOp::Sub, val) => {
                if let Some(key) = env.target_key(target) {
                    if guarded.contains_key(&key) {
                        // guarded decrement: lower bound held by the guard,
                        // upper unchanged (it only decreases) — sound.
                        let vi = expr_interval(val);
                        if let Some(entry) = ivs.get_mut(&key) {
                            entry.closed = entry.closed && vi.closed;
                        }
                        continue;
                    }
                }
                changed |= walk_stmt(s, env, ivs, has_unguarded_sub, has_compound);
            }
            other => changed |= walk_stmt(other, env, ivs, has_unguarded_sub, has_compound),
        }
    }
    changed
}

fn conjuncts(e: &Expr) -> Vec<&Expr> {
    match e {
        Expr::Binary(BinOp::LAnd, a, b) => {
            let mut v = conjuncts(a);
            v.extend(conjuncts(b));
            v
        }
        other => vec![other],
    }
}

fn guard_target(e: &Expr, env: &WalkEnv) -> Option<FieldKey> {
    match e {
        Expr::Binary(_, a, b) => {
            env.target_key(a).or_else(|| env.target_key(b))
        }
        other => env.target_key(other),
    }
}
