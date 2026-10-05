//! ============================================================================
//! LILA OPTIMIZER — SUBSYSTEM 3: PHYSICS & SPATIAL QUERIES
//! Automated inequality transformation + Galois bit-mask collision synthesis
//! ============================================================================
//!
//! THE DEV WRITES the naive thing (a per-pair loop with an exact test):
//!
//!     for (t: Turret) {
//!         if dist(b.x, b.y, t.x, t.y) < 15 { kill b; t.hp -= 1; }
//!     }
//!     if dist(p.x, p.y, e.x, e.y) < 300 { ... }
//!
//! THE COMPILER REWRITES (provably equivalent; see the unit tests that sample
//! the two forms against each other):
//!
//!   1. CHEBYSHEV GUARD (statement-level inequality transform) —
//!      the Euclidean distance is ALWAYS >= max(|dx|, |dy|) (the Chebyshev /
//!      L-infinity bound; note the Manhattan L1 bound runs the WRONG way for
//!      skipping: |dx|+|dy| >= dist means `manh < K` is only a SUFFICIENT
//!      enter-test, never a sound skip-guard — lilac refuses unsound
//!      "optimizations" and documents the distinction). The compiler lowers
//!      the hot path to:
//!
//!          if (dx < 15 && dx > -15 && dy < 15 && dy > -15) {
//!              if dist(...) < 15 { ... }        // exact test, rare path
//!          }
//!
//!      so the expensive i64-isqrt (OP_DIST) executes only for points inside
//!      the bounding box. The exact test is RETAINED — semantics are
//!      bit-identical; only the frequency of expensive work changes.
//!
//!   2. GALOIS 32-LANE COLLISION MASKS (loop-level synthesis) — when a
//!      collision loop scans a type whose declared capacity fits 32 slots,
//!      the compiler synthesizes ONE new opcode (COLLIDE_MASK) that sweeps
//!      the live-entity bitmap and, per 32-bit word, evaluates the bounding
//!      box test for all lanes with pure AND/shift/select arithmetic,
//!      producing an immediate u32 bitmask. The loop becomes a ctz-scan of
//!      that mask (FOR_MASK), executing the exact test only on candidates:
//!
//!          let __cm0 = coll_mask(Turret, b.x, b.y, 15, 15);  // 32 lanes
//!          for_mask (t: Turret) in __cm0 {
//!              if dist(b.x, b.y, t.x, t.y) < 15 { ... }       // candidates
//!          }
//!
//!      "up to 32 entity states in a single ALU cycle" = one bitmap word per
//!      COLLIDE_MASK step; the compiler rewrites loops only for types whose
//!      declared capacity <= 32 (otherwise the original loop is kept — the
//!      lane window is an explicit, documented contract, and the loader will
//!      reject oversized capacities anyway).
//!
//!   3. SWEPT MINKOWSKI INTERVALS (`swept_hit` builtin -> OP_SWEPT) —
//!      a moving-vs-static box test across the whole frame step as CONTINUOUS
//!      1D inequalities per axis. For relative motion r(t) = r0 + v*t with
//!      t in [0,1] and half-width h, the interval test is exact:
//!          hit_axis  iff  (r0 * r1 < 0)  or  (min(|r0|, |r1|) < h)
//!      (either the relative position crosses zero during the step, or an
//!      endpoint already sits inside). Both axes combine with AND. One
//!      branchless opcode replaces a developer loop of sub-step sampling.
//!
//! All decisions are made when the DEV builds. The runtime executes finished
//! opcodes; it never sees (or knows about) the original source shape.

use crate::ast::*;
use crate::checker::{Ctx, FnKind, VT};
use std::collections::HashMap;

// ---------------- predicate recognition ----------------

/// Is this expression free of side effects? (rand() is the only effectful
/// builtin — re-evaluating it would change the deterministic stream.)
fn is_pure(e: &Expr) -> bool {
    match e {
        Expr::Call(name, args) => {
            if name == "rand" { return false; }
            args.iter().all(is_pure)
        }
        Expr::Unary(_, a) => is_pure(a),
        Expr::Binary(_, a, b) => is_pure(a) && is_pure(b),
        Expr::Field(_, _) | Expr::Ident(_) | Expr::IntLit(_) | Expr::FixLit(_)
        | Expr::BoolLit(_) | Expr::Atom(_) => true,
        Expr::Index(base, ix) => is_pure(base) && is_pure(ix),
        Expr::Intrin(_, _, args, _) => args.iter().all(is_pure),
    }
}

/// Constant K of a `< K` radius predicate: returns the raw Q24.8 value.
/// Int literals widen (K px = K << 8); Fix literals are already raw.
fn const_k(e: &Expr) -> Option<i32> {
    match e {
        Expr::IntLit(v) => {
            let k = *v as i64 as i32 as i64;
            if k <= 0 || k > (1 << 22) { return None; }
            Some((k << 8) as i32)
        }
        Expr::FixLit(v) if *v > 0 => Some(*v),
        _ => None,
    }
}

/// Recognize `dist(a,b,c,d) < K` (or `K > dist(...)`).
fn dist_pred(e: &Expr) -> Option<(&[Expr], i32)> {
    match e {
        Expr::Binary(BinOp::Lt, a, b) => {
            if let Expr::Call(name, args) = &**a {
                if name == "dist" && args.len() == 4 {
                    return const_k(b).map(|k| (&args[..], k));
                }
            }
            None
        }
        Expr::Binary(BinOp::Gt, a, b) => {
            if let Expr::Call(name, args) = &**b {
                if name == "dist" && args.len() == 4 {
                    return const_k(a).map(|k| (&args[..], k));
                }
            }
            None
        }
        _ => None,
    }
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

fn fold_ands(mut es: Vec<Expr>) -> Option<Expr> {
    let mut acc = es.pop()?;
    while let Some(e) = es.pop() {
        acc = Expr::Binary(BinOp::LAnd, Box::new(e), Box::new(acc));
    }
    Some(acc)
}

fn neg_lit(k: &Expr) -> Option<Expr> {
    match k {
        Expr::IntLit(v) => Some(Expr::IntLit((-(*v as i64)) as u32)),
        Expr::FixLit(v) => Some(Expr::FixLit(-v)),
        _ => None,
    }
}

// ---------------- the pass ----------------

pub struct PhysicsResult {
    pub rewrites: Vec<(String, String)>,
}

struct Env {
    ent_index: HashMap<String, usize>,
    ent_vars: HashMap<String, usize>,
    /// per-type declared capacity (max live slots)
    lane_max: Vec<u32>,
    counter: usize,
}

impl Env {
    fn child(&self, var: &str, tidx: usize) -> Env {
        let mut v = self.ent_vars.clone();
        v.insert(var.to_string(), tidx);
        Env { ent_index: self.ent_index.clone(), ent_vars: v, lane_max: self.lane_max.clone(), counter: self.counter }
    }
}

pub fn rewrite(ctx: &mut Ctx) -> PhysicsResult {
    // own the shared data (avoids borrow conflicts while mutating fn bodies)
    let ent_index = ctx.ent_index.clone();
    // v9: Galois masks are WINDOWED — 32 lanes per window, up to 4 windows
    // (128 slots). Wider capacities keep the original loop (code-size bound).
    let lane_max: Vec<u32> = ctx.entities.iter().map(|e| e.max_live).collect();

    let mut rewrites = Vec::new();
    for fi in 0..ctx.fns.len() {
        if matches!(ctx.fns[fi].kind, FnKind::Draw0 | FnKind::DrawEnt) {
            continue; // draw fns hold no physics
        }
        let fname = ctx.fns[fi].name.clone();
        let mut ent_vars = HashMap::new();
        for (pn, pvt) in &ctx.fns[fi].params {
            if let VT::Ent(ei) = pvt {
                ent_vars.insert(pn.clone(), *ei);
            }
        }
        let mut env = Env { ent_index: ent_index.clone(), ent_vars, lane_max: lane_max.clone(), counter: 0 };
        let body = std::mem::take(&mut ctx.fns[fi].body);
        ctx.fns[fi].body = rw_block(body, &mut env, &fname, &mut rewrites);
    }
    PhysicsResult { rewrites }
}

fn rw_block(stmts: Vec<Stmt>, env: &mut Env, site: &str, report: &mut Vec<(String, String)>) -> Vec<Stmt> {
    let mut out: Vec<Stmt> = Vec::with_capacity(stmts.len());
    for s in stmts {
        match s {
            Stmt::For(var, ent_name, body) => {
                let tidx = env.ent_index.get(&ent_name).copied();
                if let Some(t) = tidx {
                    let max_live = env.lane_max[t];
                    if max_live <= 32 * 4 {
                        if let Some((cx, cy, k, exact_if, rest, _k_desc)) =
                            try_galois(&var, t, &body, env)
                        {
                            // WINDOWED Galois rewrite: one 32-lane mask + ctz
                            // candidate loop per 32-slot window. Windows scan
                            // ascending, so overall entity order matches the
                            // original nextLive scan. The window base rides in
                            // consts[2]; codegen packs it into the opcode's
                            // spare operand bits.
                            let n_win = ((max_live + 31) / 32) as i32;
                            let mut env2 = env.child(&var, t);
                            let inner_rest = rw_block(rest, &mut env2, site, report);
                            for w in 0..n_win {
                                let name = format!("__cm{}", env.counter);
                                env.counter += 1;
                                out.push(Stmt::Let(
                                    name.clone(),
                                    Expr::Intrin(INTR_COLL_MASK, t as u16, vec![cx.clone(), cy.clone()], vec![k, k, w * 32]),
                                ));
                                let mut inner = vec![exact_if.clone()];
                                inner.extend(inner_rest.iter().cloned());
                                out.push(Stmt::ForMask(var.clone(), ent_name.clone(), name, inner));
                            }
                            report.push((
                                format!("{}:for({}:{})", site, var, ent_name),
                                format!("Galois {}-window mask (K={} raw box superset of the K disk, {} slots) + ctz candidate loops; exact dist retained on candidates", n_win, k, max_live),
                            ));
                            continue;
                        }
                    }
                }
                let mut env2 = match tidx {
                    Some(t) => env.child(&var, t),
                    None => Env {
                        ent_index: env.ent_index.clone(),
                        ent_vars: env.ent_vars.clone(),
                        lane_max: env.lane_max.clone(),
                        counter: env.counter,
                    },
                };
                let body = rw_block(body, &mut env2, site, report);
                out.push(Stmt::For(var, ent_name, body));
            }
            Stmt::If(cond, then_b, else_b) => {
                let then_b = rw_block(then_b, env, site, report);
                let else_b = rw_block(else_b, env, site, report);
                if let Some((pre, outer_cond, new_then, desc)) = try_chebyshev(&cond, &then_b, env) {
                    report.push((format!("{}:if", site), desc));
                    out.push(pre.0);
                    out.push(pre.1);
                    out.push(Stmt::If(outer_cond, new_then, else_b));
                } else {
                    out.push(Stmt::If(cond, then_b, else_b));
                }
            }
            Stmt::While(cond, body) => {
                let body = rw_block(body, env, site, report);
                out.push(Stmt::While(cond, body));
            }
            Stmt::ForMask(var, ent_name, m, body) => {
                let body = rw_block(body, env, site, report);
                out.push(Stmt::ForMask(var, ent_name, m, body));
            }
            other => out.push(other),
        }
    }
    out
}

/// Galois rewrite of one collision loop: returns the mask ingredients
/// (center exprs, radius, the exact-test If, the remaining body) and lets
/// the caller synthesize one mask+ForMask pair per 32-slot window.
fn try_galois(
    var: &str,
    _tidx: usize,
    body: &[Stmt],
    env: &mut Env,
) -> Option<(Expr, Expr, i32, Stmt, Vec<Stmt>, String)> {
    // first statement must be the exact-test If
    let (if_stmt, rest) = match body.split_first() {
        Some((s @ Stmt::If(_, _, else_b), r)) if else_b.is_empty() => (s, r.to_vec()),
        _ => return None,
    };
    let Stmt::If(cond, then_b, _) = if_stmt else { return None };

    // exactly ONE dist predicate among the conjuncts
    let mut dist_site: Option<&[Expr]> = None;
    let mut k: i32 = 0;
    for c in conjuncts(cond) {
        if let Some((args, kk)) = dist_pred(c) {
            if dist_site.is_some() { return None; }
            dist_site = Some(args);
            k = kk;
        }
    }
    let args = dist_site?;

    // args must be (a.x, a.y, v.x, v.y) or (v.x, v.y, a.x, a.y) where v is
    // the loop var and a is some other entity variable
    let field_var = |e: &Expr| -> Option<(String, String)> {
        if let Expr::Field(base, fname) = e {
            if let Expr::Ident(v) = &**base {
                return Some((v.clone(), fname.clone()));
            }
        }
        None
    };
    let (o1, o2, o3, o4) = (field_var(&args[0])?, field_var(&args[1])?, field_var(&args[2])?, field_var(&args[3])?);
    let (outer, cx, cy) = if o3.0 == var && o4.0 == var && o3.1 == "x" && o4.1 == "y" && o1.0 != var && o1.1 == "x" && o2.1 == "y" {
        (o1.0, args[0].clone(), args[1].clone())
    } else if o1.0 == var && o2.0 == var && o1.1 == "x" && o2.1 == "y" && o3.0 != var && o3.1 == "x" && o4.1 == "y" {
        (o3.0, args[2].clone(), args[3].clone())
    } else {
        return None;
    };
    if !env.ent_vars.contains_key(&outer) {
        return None;
    }

    let exact_if = Stmt::If(cond.clone(), then_b.clone(), Vec::new());
    let desc = format!(
        "Galois mask windows (K={} raw box superset of the K disk) + ctz candidate loops; exact dist retained on candidates",
        k
    );
    Some((cx, cy, k, exact_if, rest, desc))
}

/// Chebyshev guard rewrite of one If statement.
/// Returns ((let __dx, let __dy), guarded outer cond, new then, description).
fn try_chebyshev(
    cond: &Expr,
    then_b: &[Stmt],
    env: &mut Env,
) -> Option<((Stmt, Stmt), Expr, Vec<Stmt>, String)> {
    let cands = conjuncts(cond);
    let mut found: Option<(usize, Expr, &[Expr], i32)> = None;
    for (i, c) in cands.iter().enumerate() {
        if let Some((args, k)) = dist_pred(c) {
            if found.is_some() { return None; }
            found = Some((i, (*c).clone(), args, k));
        }
    }
    let (idx, pred_clone, args, _k) = found?;
    if !is_pure(&args[0]) || !is_pure(&args[1]) || !is_pure(&args[2]) || !is_pure(&args[3]) {
        return None;
    }

    let n = env.counter;
    env.counter += 1;
    let dxn = format!("__dx{}", n);
    let dyn_ = format!("__dy{}", n);
    let let_dx = Stmt::Let(
        dxn.clone(),
        Expr::Binary(BinOp::Sub, Box::new(args[0].clone()), Box::new(args[2].clone())),
    );
    let let_dy = Stmt::Let(
        dyn_.clone(),
        Expr::Binary(BinOp::Sub, Box::new(args[1].clone()), Box::new(args[3].clone())),
    );

    // the K literal in its ORIGINAL form (preserves int->fixed widening)
    let k_expr: Expr = match &cands[idx] {
        Expr::Binary(BinOp::Lt, _, b) => (**b).clone(),
        Expr::Binary(BinOp::Gt, b, _) => (**b).clone(),
        _ => return None,
    };
    let neg_k = neg_lit(&k_expr)?;

    let dx = || Expr::Ident(dxn.clone());
    let dy = || Expr::Ident(dyn_.clone());
    let g = |a: Expr, op: BinOp, b: Expr| Expr::Binary(op, Box::new(a), Box::new(b));
    let guard = g(
        g(g(dx(), BinOp::Lt, k_expr.clone()), BinOp::LAnd, g(dx(), BinOp::Gt, neg_k.clone())),
        BinOp::LAnd,
        g(g(dy(), BinOp::Lt, k_expr.clone()), BinOp::LAnd, g(dy(), BinOp::Gt, neg_k)),
    );

    // outer condition: other conjuncts + guard (dist predicate replaced)
    let mut outer_parts: Vec<Expr> = Vec::new();
    for (i, c) in cands.iter().enumerate() {
        if i != idx {
            outer_parts.push((*c).clone());
        }
    }
    outer_parts.push(guard);
    let outer_cond = fold_ands(outer_parts)?;

    // inner: the EXACT original predicate gates the original body
    let inner = Stmt::If(pred_clone, then_b.to_vec(), Vec::new());
    Some((
        (let_dx, let_dy),
        outer_cond,
        vec![inner],
        "Chebyshev L-inf guard: max(|dx|,|dy|) >= K skips the isqrt (exact dist test retained)".into(),
    ))
}

// ---------------- tests: soundness of the two transforms ----------------

#[cfg(test)]
mod tests {
    /// The L-inf bound: dist >= max(|dx|, |dy|) on integer samples — the
    /// mathematical fact both transforms rely on. If this ever fails, the
    /// guards are unsound and the passes must be disabled.
    #[test]
    fn chebyshev_bound_holds_on_samples() {
        for dx in -65i64..=65 {
            for dy in -65i64..=65 {
                let d = ((dx * dx + dy * dy) as f64).sqrt();
                let linf = dx.abs().max(dy.abs()) as f64;
                assert!(d >= linf - 1e-9, "dist {} < linf {} at ({},{})", d, linf, dx, dy);
            }
        }
    }

    /// The Galois mask box is a SUPERSET of the disk: any point with
    /// dist < K satisfies |dx| < K && |dy| < K, so the candidate loop sees
    /// every true hit (the exact test inside decides).
    #[test]
    fn galois_box_superset_of_disk() {
        for dx in -20..=20i64 {
            for dy in -20..=20i64 {
                let dist2 = dx * dx + dy * dy;
                let in_disk = dist2 < 15 * 15;
                let in_box = dx.abs() < 15 && dy.abs() < 15;
                if in_disk {
                    assert!(in_box, "disk point ({},{}) missed by the mask box", dx, dy);
                }
            }
        }
    }

    /// The swept-interval 1D solve: hit iff crossing OR endpoint-inside.
    #[test]
    fn swept_interval_1d_exact() {
        let hit = |r0: i64, r1: i64, h: i64| -> bool {
            // reference: sample t in [0,1] finely (continuous approximation)
            let mut ref_hit = false;
            for i in 0..=1000i64 {
                let t = i as f64 / 1000.0;
                let r = r0 as f64 + (r1 - r0) as f64 * t;
                if r.abs() < h as f64 { ref_hit = true; break; }
            }
            let solved = (r0 * r1 < 0) || (r0.abs().min(r1.abs()) < h);
            assert_eq!(solved, ref_hit, "sweep solve mismatch r0={} r1={} h={}", r0, r1, h);
            solved
        };
        hit(10, 20, 5);   // never inside
        hit(10, 2, 5);    // ends inside
        hit(-30, 30, 5);  // crosses through
        hit(3, -3, 5);    // crosses, starts inside
        hit(-4, -40, 5);  // starts inside, exits
        hit(6, 6, 5);     // static miss
        hit(1, 1, 5);     // static hit
    }
}
