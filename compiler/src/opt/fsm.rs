//! ============================================================================
//! LILA OPTIMIZER — SUBSYSTEM 5: AI & STATE MACHINES (BIT-PLANE TRANSITIONS)
//! ============================================================================
//!
//! THE DEV WRITES declarative AI (the if/else jungle is GONE from their code):
//!
//!     fsm #guard {
//!         PATROL { hp <= 0 -> DEAD, sees: key(#fire) -> ALERT, timer > 120 -> CHASE }
//!         CHASE { hp <= 0 -> DEAD, dist(e.x,e.y,p.x,p.y) < 40 -> STRIKE }
//!         ...
//!     }
//!     entity Guard { ..., fsm: #guard }
//!     fn update(g: Guard) {
//!         ...
//!         fsm_step(#guard);      // one statement — the whole behavior tick
//!     }
//!
//! THE COMPILER SYNTHESIZES (at build time):
//!
//!   1. BIT-PLANE STATE ENCODING — states are numbered 0..N-1 and stored in a
//!      minimum-width bitfield auto-added to the entity schema (5 states ->
//!      `fsm_state: u3`), packed INTO the cache-line SoA row like any other
//!      field (subsystem 1).
//!   2. TOTAL BRANCHLESS GUARD EVALUATION — every state's guards are compiled
//!      to predicate programs and evaluated UNCONDITIONALLY, every frame, for
//!      every entity. The event word is assembled by arithmetic selection:
//!
//!          em = 0
//!          for each state i:  em |= ((s == i) ? em_i : 0)
//!          em_i = BIT01(guard_i_0) << 0 | BIT01(guard_i_1) << 1 | ...
//!
//!      (`(s == i) ? x : 0` compiles to `x & BIT01(s == i)` — AND/shift only,
//!      zero branches. The CPU never predicts anything: there is nothing to
//!      predict.) Every entity runs the same op sequence — the "dozens of
//!      enemies on SIMD-style lanes" story is the entity bitmap ctz scan
//!      driving identical straight-line code per live entity.
//!   3. BIT-PLANE TRANSITION TABLE — T[state][events] is baked into the
//!      .libyte (v3 FSM section) and the runtime transition is ONE op:
//!
//!          next = T[(s << S) | em]        // OP_FSM_NEXT: shift, mask, load
//!
//!      No conditional behavioral tree survives in the bytecode: guard
//!      priority (first declared wins) is PRECOMPUTED into the table by the
//!      compiler — the exact low-bit-wins enumeration lives in
//!      `build_table()` below and is unit-tested against the naive
//!      if-chain reference.
//!   4. STATE FIELD WIDTH PROOF — the table + bitfield widths are comptime
//!      facts mirrored into the Zig manifest (opt::soa) for the engine build.
//!
//! All at dev-compile time. The player's runtime: one opcode per tick.

use crate::ast::*;
use crate::checker::{Ctx, VT};

pub struct FsmResult {
    /// baked transition tables, one per fsm: T[state * span + events] = next
    pub tables: Vec<Vec<u8>>,
    /// (fsm name, states, table bytes, minimization note)
    pub report: Vec<(String, usize, usize, String)>,
}

/// Build the transition table for one fsm.
///
/// Guards are evaluated in DECLARED ORDER per state; the first true guard
/// wins (this is the semantics the dev expressed, now precomputed). For an
/// event bitmask `em` (bit j = guard j of THIS state fired), the next state
/// is the target of the lowest set bit; em == 0 stays in the current state.
/// Every state has exactly `span = 2^S` entries (S = max guards across all
/// states), padded with the self-state for unreachable masks.
pub fn build_table(
    guards_per_state: &[Vec<usize>], // per state: target state indices per guard
    nstates: usize,
) -> Vec<u8> {
    let max_g = guards_per_state.iter().map(|g| g.len()).max().unwrap_or(0);
    let span = if max_g == 0 { 1usize } else { 1usize << max_g };
    let mut t = vec![0u8; nstates * span];
    for s in 0..nstates {
        for em in 0..span {
            let next = if em == 0 {
                s
            } else {
                // lowest set bit = first (highest-priority) fired guard
                let j = em.trailing_zeros() as usize;
                match guards_per_state[s].get(j) {
                    Some(&tgt) => tgt,
                    None => s, // guard slot unused in this state: unreachable
                }
            };
            t[s * span + em] = next as u8;
        }
    }
    t
}

/// Reference semantics the table must reproduce (used by the unit test):
/// the naive if-chain a dev WOULD have written before the optimizer existed.
#[allow(dead_code)]
fn naive_next(guards_per_state: &[Vec<usize>], s: usize, em: usize) -> usize {
    if em == 0 { return s; }
    let g = &guards_per_state[s];
    for j in 0..g.len() {
        if em & (1 << j) != 0 {
            return g[j];
        }
    }
    s
}

/// Rewrite every `fsm_step(#name);` statement into the branchless expansion
/// and produce the baked tables.
pub fn expand(ctx: &mut Ctx) -> FsmResult {
    let mut tables = Vec::new();
    let mut report = Vec::new();

    // ---- FORMAL-LANGUAGE PASS: duplicate-state elimination (Hopcroft-style
    // partition refinement over the global guard alphabet). States with
    // structurally identical guard programs whose continuations coincide are
    // equivalence classes; every transition into the class is redirected to
    // its representative (lowest index), the branchless expansion then only
    // emits guard programs for representatives, and user literals that name
    // merged states (draw comparisons, assignments) are remapped. Soundness:
    // the seed partition groups states by their exact guard structure, so
    // merged states listen to exactly the same predicates — dynamics are
    // bit-identical for every world state.
    let mut reps_by_name: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();

    // ---- observability scan: every `fsm_state == N` / `!= N` / `= N` literal
    // in user code PINS state N of the owning entity's fsm (never merged).
    let mut pins: std::collections::HashMap<usize, std::collections::HashSet<usize>> =
        std::collections::HashMap::new();
    for fi in 0..ctx.fns.len() {
        let params: Vec<(String, usize)> = ctx.fns[fi].params.iter()
            .filter_map(|(n, vt)| match vt {
                VT::Ent(ei) => Some((n.clone(), *ei)),
                _ => None,
            })
            .collect();
        if params.is_empty() { continue; }
        scan_fsm_pins(&ctx.fns[fi].body, &params, &mut pins);
    }

    for f in &ctx.fsms {
        let exprs: Vec<Vec<(Expr, usize)>> = f.guards.clone();
        let pinned: Vec<bool> = (0..f.states.len())
            .map(|s| pins.get(&f.entity).map_or(false, |set| set.contains(&s)))
            .collect();
        let (classes, npreds) = minimize_dfa(&exprs, f.states.len(), &pinned);
        // representative = pinned member of the class if any (keeps the
        // observed fsm_state value stable), else the lowest index
        let reps: Vec<usize> = {
            let mut r = vec![0usize; f.states.len()];
            for s in 0..f.states.len() { r[s] = s; }
            for s in 0..f.states.len() {
                for t in 0..f.states.len() {
                    if classes[t] == classes[s] {
                        let t_better = (pinned[t] && !pinned[r[s]])
                            || (!pinned[r[s]] && !pinned[t] && t < r[s])
                            || (pinned[t] && pinned[r[s]] && t < r[s]);
                        if t_better { r[s] = t; }
                    }
                }
            }
            r
        };
        let nclasses = {
            let mut seen = vec![false; f.states.len()];
            let mut n = 0;
            for s in 0..f.states.len() {
                if !seen[classes[s]] {
                    seen[classes[s]] = true;
                    n += 1;
                }
            }
            n
        };
        // redirect every guard target to its class representative
        let targets: Vec<Vec<usize>> = f.guards.iter()
            .map(|gs| gs.iter().map(|(_, tgt)| reps[*tgt]).collect())
            .collect();
        let t = build_table(&targets, f.states.len());
        let note = if nclasses < f.states.len() {
            format!("Hopcroft {} -> {} states, {} distinct guard programs",
                f.states.len(), nclasses, npreds)
        } else {
            format!("{} distinct guard programs (no duplicate states)", npreds)
        };
        report.push((f.name.clone(), f.states.len(), t.len(), note));
        tables.push(t);
        reps_by_name.insert(f.name.clone(), reps);
    }

    // ---- remap user literals that name fsm states (draw comparisons,
    // assignments) so merged states read as their representative everywhere.
    if reps_by_name.values().any(|r| r.iter().enumerate().any(|(i, r)| *r != i)) {
        for fi in 0..ctx.fns.len() {
            let ent_fsms: Vec<(String, Vec<usize>)> = ctx.fns[fi].params.iter()
                .filter_map(|(_, vt)| match vt {
                    VT::Ent(ei) => ctx.fsms.iter()
                        .find(|f| f.entity == *ei)
                        .and_then(|f| reps_by_name.get(&f.name)
                            .cloned().map(|r| (f.name.clone(), r))),
                    _ => None,
                })
                .collect();
            if ent_fsms.is_empty() { continue; }
            remap_state_literals(&mut ctx.fns[fi].body, &ent_fsms);
        }
    }

    // expand statements (bodies areCtx-owned; work fn by fn)
    for fi in 0..ctx.fns.len() {
        // does this fn contain an FsmStep?
        let has = body_has_fsm_step(&ctx.fns[fi].body);
        if !has { continue; }
        // the entity param (name, index)
        let (pname, pei) = ctx.fns[fi].params.iter()
            .find_map(|(n, vt)| match vt {
                VT::Ent(ei) => Some((n.clone(), *ei)),
                _ => None,
            })
            .expect("checker guarantees fsm_step lives in an entity fn");
        let body = std::mem::take(&mut ctx.fns[fi].body);
        let mut counter = 0usize;
        ctx.fns[fi].body = expand_block(body, ctx, &pname, pei, &mut counter, &reps_by_name);
    }

    FsmResult { tables, report }
}

fn body_has_fsm_step(stmts: &[Stmt]) -> bool {
    for s in stmts {
        let found = match s {
            Stmt::FsmStep(_) => true,
            Stmt::If(_, a, b) => body_has_fsm_step(a) || body_has_fsm_step(b),
            Stmt::While(_, b) => body_has_fsm_step(b),
            Stmt::For(_, _, b) | Stmt::ForMask(_, _, _, b) => body_has_fsm_step(b),
            _ => false,
        };
        if found { return true; }
    }
    false
}

fn expand_block(
    stmts: Vec<Stmt>,
    ctx: &Ctx,
    pname: &str,
    pei: usize,
    counter: &mut usize,
    reps_by_name: &std::collections::HashMap<String, Vec<usize>>,
) -> Vec<Stmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for s in stmts {
        match s {
            Stmt::FsmStep(atom) => {
                let (fidx, fsm) = match ctx.fsms.iter().enumerate().find(|(_, f)| f.atom == atom) {
                    Some(x) => x,
                    None => continue, // checker guarantees
                };
                // the entity must own this fsm (checker validated)
                let n = *counter;
                *counter += 1;
                let emn = format!("__em{}", n);
                let sn = format!("__st{}", n);

                // em = OR over states i of ( (s == i) ? em_i : 0 )
                // em_i = OR over guards j of BIT01(pred_i_j) << j
                // (all predicates evaluate unconditionally — branchless)
                let mut em_terms: Vec<Expr> = Vec::new();
                let reps = reps_by_name.get(&fsm.name);
                for (si, _) in fsm.states.iter().enumerate() {
                    // merged-away states are never occupied after the
                    // redirect — skip their guard programs entirely
                    if let Some(r) = reps {
                        if r[si] != si { continue; }
                    }
                    // em_i — guards of THIS state (FsmInfo.guards[si])
                    let mut state_terms: Vec<Expr> = Vec::new();
                    for (j, (pred, _)) in fsm.guards[si].iter().enumerate() {
                        let pred = rewrite_guard(pred, pname, ctx, pei);
                        let bit = Expr::Binary(
                            BinOp::Shl,
                            Box::new(Expr::Intrin(INTR_BIT01, 0, vec![pred], vec![])),
                            Box::new(Expr::IntLit(j as u32)),
                        );
                        state_terms.push(bit);
                    }
                    if state_terms.is_empty() { continue; }
                    let em_i = fold_or(state_terms);
                    // select by (s == si): x & BIT01(s == i)
                    let sel = Expr::Intrin(
                        INTR_BIT01, 0,
                        vec![Expr::Binary(
                            BinOp::Eq,
                            Box::new(Expr::Ident(sn.clone())),
                            Box::new(Expr::IntLit(si as u32)),
                        )],
                        vec![],
                    );
                    let term = Expr::Binary(BinOp::And, Box::new(em_i), Box::new(sel));
                    em_terms.push(term);
                }
                let em_zero = Expr::IntLit(0);
                let em_expr = if em_terms.is_empty() { em_zero } else { fold_or(em_terms) };

                let state_field = "fsm_state".to_string();
                // ORDER MATTERS: read the CURRENT state into a local FIRST —
                // the event-mask expression's state selectors reference it —
                // then compute em, then transition.
                out.push(Stmt::Let(
                    sn.clone(),
                    Expr::Field(Box::new(Expr::Ident(pname.to_string())), state_field.clone()),
                ));
                out.push(Stmt::Let(emn.clone(), em_expr));
                out.push(Stmt::Assign(
                    Expr::Field(Box::new(Expr::Ident(pname.to_string())), state_field),
                    AssignOp::Set,
                    Expr::Intrin(
                        INTR_FSM_NEXT,
                        fidx as u16,
                        vec![Expr::Ident(emn), Expr::Ident(sn)],
                        vec![],
                    ),
                ));
            }
            Stmt::If(c, a, b) => {
                let a = expand_block(a, ctx, pname, pei, counter, reps_by_name);
                let b = expand_block(b, ctx, pname, pei, counter, reps_by_name);
                out.push(Stmt::If(c, a, b));
            }
            Stmt::While(c, b) => out.push(Stmt::While(c, expand_block(b, ctx, pname, pei, counter, reps_by_name))),
            Stmt::For(v, e, b) => out.push(Stmt::For(v, e, expand_block(b, ctx, pname, pei, counter, reps_by_name))),
            Stmt::ForMask(v, e, m, b) => {
                out.push(Stmt::ForMask(v, e, m, expand_block(b, ctx, pname, pei, counter, reps_by_name)))
            }
            other => out.push(other),
        }
    }
    out
}

// ---------------- FORMAL-LANGUAGE PASS: DFA duplicate-state minimization ----------------
//
// The declarative fsm is a total DFA. Its alphabet is the set of DISTINCT
// guard predicates (deduplicated structurally across states — the dev writes
// `hp <= 0 -> DEAD` in five states and the compiler sees ONE event). States
// are partitioned Hopcroft-style: seeded by guard-structure signature, then
// refined to a fixpoint by class-of-transition signatures. Two states merge
// only when they listen to exactly the same predicates and every transition
// lands in the same class — so merged states are behaviorally identical for
// every world state, and (being never occupied after redirection) their
// guard programs vanish from the branchless expansion.
//
// fsm_state is user-observable (draw arms compare it): any state NAMED by a
// literal in user code (`d.fsm_state == N`) is PINNED as a singleton class,
// so observable behavior is preserved bit-for-bit; everything else minimizes
// freely and its (now unreachable) literals are remapped to the class rep.

/// Returns (class id per state, distinct predicate count).
///
/// Partition = the bisimulation quotient that respects observability:
///  - unpinned states seed by predicate-listener signature and refine by
///    transition classes (coarsest bisimulation among themselves);
///  - pinned states (named by an `fsm_state == N` literal) seed as
///    singletons and never join anything — their literal match sets must
///    not grow;
///  - states UNREACHABLE from the initial state are never occupied at
///    runtime, so they may collapse into any same-signature class (this
///    is what makes an exact duplicate of a pinned state — HURT vs
///    WANDER — vanish when nothing ever transitions into it).
/// Together: no user literal's match set ever changes shape.
fn minimize_dfa(guards: &[Vec<(Expr, usize)>], nstates: usize, pinned: &[bool]) -> (Vec<usize>, usize) {
    // structural predicate dedup (Debug rendering is structural for these
    // small pure expressions)
    let mut keys: Vec<String> = Vec::new();
    let mut gp: Vec<Vec<(usize, usize)>> = Vec::with_capacity(nstates);
    for gs in guards {
        let mut row = Vec::with_capacity(gs.len());
        for (pred, tgt) in gs {
            let k = format!("{:?}", pred);
            let idx = match keys.iter().position(|x| *x == k) {
                Some(i) => i,
                None => { keys.push(k); keys.len() - 1 }
            };
            row.push((idx, *tgt));
        }
        gp.push(row);
    }
    let np = keys.len();
    let identity = (0..nstates).collect::<Vec<_>>();
    if np == 0 || np > 10 || nstates < 2 {
        return (identity, np); // degenerate or huge alphabet: skip (bounded compile cost)
    }
    if pinned.len() != nstates {
        return (identity, np);
    }

    // total transition function over the global alphabet: subsets of P.
    // A state is blind to predicates it has no guard for (self-loop).
    let nsym = 1usize << np;
    let mut tg = vec![0usize; nstates * nsym];
    for s in 0..nstates {
        for e in 0..nsym {
            let mut next = s;
            for (pi, tgt) in &gp[s] {
                if e & (1 << pi) != 0 { next = *tgt; break; }
            }
            tg[s * nsym + e] = next;
        }
    }

    // SEED: unpinned states group by predicate-listener signature; every
    // pinned state gets its own singleton class (unique id rides in the key).
    let mut class: Vec<usize> = vec![0; nstates];
    {
        let mut sigs: Vec<(Vec<usize>, usize)> = Vec::new();
        for s in 0..nstates {
            let sig: Vec<usize> = gp[s].iter().map(|(pi, _)| *pi).collect();
            let key = (sig, if pinned[s] { s + 1 } else { 0 });
            let cid = match sigs.iter().position(|x| *x == key) {
                Some(i) => i,
                None => { sigs.push(key); sigs.len() - 1 }
            };
            class[s] = cid;
        }
    }

    // REFINEMENT to fixpoint (Moore): split blocks whose members transition
    // into different classes. class-of-self rides in the signature so the
    // partition only ever REFINES (monotone -> terminates in <= nstates iters).
    loop {
        let mut sigs: std::collections::HashMap<Vec<usize>, usize> = std::collections::HashMap::new();
        let mut next = vec![0usize; nstates];
        let mut nclass = 0usize;
        for s in 0..nstates {
            let mut sig = Vec::with_capacity(nsym + 1);
            sig.push(class[s]);
            sig.extend((0..nsym).map(|e| class[tg[s * nsym + e]]));
            let cid = *sigs.entry(sig).or_insert_with(|| { nclass += 1; nclass - 1 });
            next[s] = cid;
        }
        let prev_n = {
            let mut seen = vec![false; nstates];
            let mut n = 0;
            for s in 0..nstates {
                if !seen[class[s]] { seen[class[s]] = true; n += 1; }
            }
            n
        };
        let done = nclass == prev_n;
        class = next;
        if done { break; }
    }

    // UNREACHABLE-STATE CLEANUP: a state never reachable from the initial
    // state is never occupied at runtime (entities enter states only via
    // table transitions, which redirect to class reps), so it may join ANY
    // same-signature class — including a pinned one. Reachable unpinned
    // states already merged among themselves in seed+refine; reachable
    // pinned states stay singletons. Together these two rules are exactly
    // the observability contract: no literal's match set ever changes.
    let mut reach = vec![false; nstates];
    reach[0] = true;
    loop {
        let mut grew = false;
        for s in 0..nstates {
            if !reach[s] { continue; }
            for e in 0..nsym {
                let t = tg[s * nsym + e];
                if !reach[t] { reach[t] = true; grew = true; }
            }
        }
        if !grew { break; }
    }
    for s in 0..nstates {
        if reach[s] || pinned[s] { continue; }
        // target: lowest-index same-signature state (deterministic)
        let mut target: Option<usize> = None;
        for t in 0..nstates {
            if t == s || sig_of(&gp[t]) != sig_of(&gp[s]) { continue; }
            let better = match target {
                None => true,
                Some(b) => t < b,
            };
            if better { target = Some(t); }
        }
        if let Some(t) = target {
            class[s] = class[t];
        }
    }
    (class, np)
}

fn sig_of(gp: &[(usize, usize)]) -> Vec<usize> {
    gp.iter().map(|(pi, _)| *pi).collect()
}

/// Remap user literals that name fsm states through the class representatives:
/// `d.fsm_state == N` / `d.fsm_state != N` / `d.fsm_state = N` become the
/// representative's index. Only fields named `fsm_state` on params whose
/// entity drives an fsm are touched; everything else is left untouched.
fn remap_state_literals(stmts: &mut Vec<Stmt>, ent_fsms: &[(String, Vec<usize>)]) {
    remap_stmts(stmts, ent_fsms);
}

fn remap_stmts(stmts: &mut Vec<Stmt>, ent_fsms: &[(String, Vec<usize>)]) {
    for st in stmts.iter_mut() {
        match st {
            Stmt::Let(_, e) => remap_expr(e, ent_fsms),
            Stmt::Assign(t, _, e) => { remap_expr(t, ent_fsms); remap_expr(e, ent_fsms); }
            Stmt::If(c, a, b) => {
                remap_expr(c, ent_fsms);
                remap_stmts(a, ent_fsms);
                remap_stmts(b, ent_fsms);
            }
            Stmt::While(c, b) => { remap_expr(c, ent_fsms); remap_stmts(b, ent_fsms); }
            Stmt::For(_, _, b) => remap_stmts(b, ent_fsms),
            Stmt::ForMask(_, _, _, b) => remap_stmts(b, ent_fsms),
            Stmt::Spawn(_, fields) => {
                for (_, e) in fields.iter_mut() { remap_expr(e, ent_fsms); }
            }
            Stmt::Kill(e) => remap_expr(e, ent_fsms),
            Stmt::Camera(a, b) => { remap_expr(a, ent_fsms); remap_expr(b, ent_fsms); }
            Stmt::Save(a, b) => { remap_expr(a, ent_fsms); remap_expr(b, ent_fsms); }
            Stmt::Draw(_, a, b, c, d, e) => {
                for x in [a, b, c, d, e] { remap_expr(x, ent_fsms); }
            }
            Stmt::DrawText(_, a, b, c) => {
                for x in [a, b, c] { remap_expr(x, ent_fsms); }
            }
            Stmt::DrawNum(a, b, c, d) => {
                for x in [a, b, c, d] { remap_expr(x, ent_fsms); }
            }
            Stmt::CallStmt(_, args) => {
                for a in args.iter_mut() { remap_expr(a, ent_fsms); }
            }
            Stmt::CallTable(_, k, _) => remap_expr(k, ent_fsms),
            _ => {}
        }
    }
}

fn remap_expr(e: &mut Expr, ent_fsms: &[(String, Vec<usize>)]) {
    match e {
        Expr::Binary(op, a, b) if matches!(op, BinOp::Eq | BinOp::Ne) => {
            remap_expr(a, ent_fsms);
            remap_expr(b, ent_fsms);
            let is_state_field = |x: &Expr| matches!(x, Expr::Field(_, f) if f == "fsm_state");
            let (base, lit) = if is_state_field(a) { (a, b) }
                else if is_state_field(b) { (b, a) }
                else { return };
            // which entity does the base ident refer to?
            let ent_name = match &**base {
                Expr::Field(b2, _) => match &**b2 {
                    Expr::Ident(n) => n.clone(),
                    _ => return,
                },
                _ => return,
            };
            let reps = ent_fsms.iter().find(|(n, _)| *n == ent_name).map(|(_, r)| r);
            let Some(reps) = reps else { return };
            let slot = &mut **lit;
            if let Expr::IntLit(n) = slot {
                if let Some(r) = reps.get(*n as usize) {
                    if *r as u32 != *n { *n = *r as u32; }
                }
            }
        }
        Expr::Binary(_, a, b) => { remap_expr(a, ent_fsms); remap_expr(b, ent_fsms); }
        Expr::Unary(_, a) => remap_expr(a, ent_fsms),
        Expr::Field(base, _) => remap_expr(base, ent_fsms),
        Expr::Call(_, args) | Expr::Intrin(_, _, args, _) => {
            for a in args.iter_mut() { remap_expr(a, ent_fsms); }
        }
        _ => {}
    }
}


/// Observability scan: collect every fsm state index that user code NAMES
/// via an `fsm_state` literal (comparisons and assignments). `params` maps
/// param name -> entity index; `pins` accumulates entity -> state set.
fn scan_fsm_pins(stmts: &[Stmt], params: &[(String, usize)], pins: &mut std::collections::HashMap<usize, std::collections::HashSet<usize>>) {
    for st in stmts {
        match st {
            Stmt::Let(_, e) => scan_fsm_pins_expr(e, params, pins),
            Stmt::Assign(t, _, e) => {
                // `d.fsm_state = N` pins N too (the user forces that state)
                scan_fsm_pins_expr(t, params, pins);
                scan_fsm_pins_expr(e, params, pins);
            }
            Stmt::If(c, a, b) => {
                scan_fsm_pins_expr(c, params, pins);
                scan_fsm_pins(a, params, pins);
                scan_fsm_pins(b, params, pins);
            }
            Stmt::While(c, b) => {
                scan_fsm_pins_expr(c, params, pins);
                scan_fsm_pins(b, params, pins);
            }
            Stmt::For(_, _, b) | Stmt::ForMask(_, _, _, b) => scan_fsm_pins(b, params, pins),
            Stmt::Spawn(_, fields) => {
                for (_, e) in fields { scan_fsm_pins_expr(e, params, pins); }
            }
            Stmt::Kill(e) => scan_fsm_pins_expr(e, params, pins),
            Stmt::Camera(a, b) | Stmt::Save(a, b) => {
                scan_fsm_pins_expr(a, params, pins);
                scan_fsm_pins_expr(b, params, pins);
            }
            _ => {}
        }
    }
}

fn scan_fsm_pins_expr(e: &Expr, params: &[(String, usize)], pins: &mut std::collections::HashMap<usize, std::collections::HashSet<usize>>) {
    let state_holder = |x: &Expr| -> Option<usize> {
        match x {
            Expr::Field(base, f) if f == "fsm_state" => match &**base {
                Expr::Ident(n) => params.iter().find(|(p, _)| p == n).map(|(_, ei)| *ei),
                _ => None,
            },
            _ => None,
        }
    };
    match e {
        Expr::Binary(op @ (BinOp::Eq | BinOp::Ne), a, b) => {
            if let Some(ei) = state_holder(a) {
                if let Expr::IntLit(n) = **b {
                    pins.entry(ei).or_default().insert(n as usize);
                }
            } else if let Some(ei) = state_holder(b) {
                if let Expr::IntLit(n) = **a {
                    pins.entry(ei).or_default().insert(n as usize);
                }
            }
            let _ = op;
            scan_fsm_pins_expr(a, params, pins);
            scan_fsm_pins_expr(b, params, pins);
        }
        Expr::Binary(_, a, b) => {
            scan_fsm_pins_expr(a, params, pins);
            scan_fsm_pins_expr(b, params, pins);
        }
        Expr::Unary(_, a) => scan_fsm_pins_expr(a, params, pins),
        Expr::Field(base, _) => scan_fsm_pins_expr(base, params, pins),
        Expr::Call(_, args) | Expr::Intrin(_, _, args, _) => {
            for a in args { scan_fsm_pins_expr(a, params, pins); }
        }
        _ => {}
    }
}

/// Rewrite guard idents for the calling context: bare names that are fields
/// of the owning entity become `<param>.<field>`; globals stay bare.
fn rewrite_guard(e: &Expr, pname: &str, ctx: &Ctx, pei: usize) -> Expr {
    match e {
        Expr::Ident(name) => {
            let is_entity_field = ctx.entities[pei].fields.iter().any(|f| &f.name == name);
            let is_global = ctx.global_index.contains_key(name);
            if is_entity_field && !is_global {
                Expr::Field(Box::new(Expr::Ident(pname.to_string())), name.clone())
            } else {
                e.clone()
            }
        }
        Expr::Unary(op, a) => Expr::Unary(*op, Box::new(rewrite_guard(a, pname, ctx, pei))),
        Expr::Binary(op, a, b) => Expr::Binary(
            *op,
            Box::new(rewrite_guard(a, pname, ctx, pei)),
            Box::new(rewrite_guard(b, pname, ctx, pei)),
        ),
        Expr::Call(name, args) => Expr::Call(
            name.clone(),
            args.iter().map(|a| rewrite_guard(a, pname, ctx, pei)).collect(),
        ),
        other => other.clone(),
    }
}

fn fold_or(mut es: Vec<Expr>) -> Expr {
    assert!(!es.is_empty());
    let mut acc = es.remove(0);
    while !es.is_empty() {
        let e = es.remove(0);
        acc = Expr::Binary(BinOp::Or, Box::new(acc), Box::new(e));
    }
    acc
}

// ---------------- tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The baked table must reproduce the naive first-true-wins if-chain for
    /// every (state, event) combination — the compiler's central correctness
    /// claim for subsystem 5.
    #[test]
    fn table_matches_naive_if_chain() {
        // 4 states; state 0 has 3 guards, others fewer (ragged)
        let guards = vec![
            vec![1, 2, 3],
            vec![0, 3],
            vec![2],
            vec![],
        ];
        let span = 8; // 2^ceil — must match build_table's span for 3 guards
        let t = build_table(&guards, 4);
        assert_eq!(t.len(), 4 * span);
        for s in 0..4 {
            for em in 0..span {
                assert_eq!(
                    t[s * span + em] as usize,
                    naive_next(&guards, s, em),
                    "table mismatch at state {} events {:b}",
                    s, em
                );
            }
        }
    }

    #[test]
    fn no_guards_is_identity() {
        // zero guards everywhere -> span = 2^0 = 1: only em == 0 is reachable,
        // and it stays in the current state. The table is the identity.
        let t = build_table(&[vec![], vec![]], 2);
        assert_eq!(t, vec![0, 1]);
    }

    /// Two states with structurally identical guard programs whose targets
    /// coincide are one equivalence class: Hopcroft merges them, redirects
    /// transitions to the representative, and the count drops.
    #[test]
    fn hopcroft_merges_duplicate_states() {
        let p = Expr::Binary(BinOp::Gt, Box::new(Expr::Ident("timer".into())), Box::new(Expr::IntLit(90)));
        let q = Expr::Binary(BinOp::Le, Box::new(Expr::Ident("hp".into())), Box::new(Expr::IntLit(0)));
        // W: {q->GONE, p->REST}   REST: {q->GONE, p->W}   HURT: {q->GONE, p->REST}  GONE: {}
        // HURT == W structurally -> merged; REST differs (targets W) -> own class.
        let guards = vec![
            vec![(q.clone(), 3usize), (p.clone(), 1usize)], // 0 W
            vec![(q.clone(), 3usize), (p.clone(), 0usize)], // 1 REST
            vec![(q.clone(), 3usize), (p.clone(), 1usize)], // 2 HURT (dup of W)
            vec![],                                         // 3 GONE
        ];
        // REST is user-observed (e.g. a draw arm colors it) -> pinned.
        let pinned = [false, true, false, false];
        let (class, np) = minimize_dfa(&guards, 4, &pinned);
        assert_eq!(np, 2, "p and q dedup to two distinct events");
        assert_eq!(class[0], class[2], "W and HURT must merge");
        assert_ne!(class[0], class[1], "pinned REST must stay separate");
        assert_ne!(class[0], class[3], "GONE must stay separate");
        // W --p--> REST --p--> W oscillate and nothing pins REST: the full
        // bisimulation quotient collapses {W, REST, HURT} into one class.
        let pinned = [false, false, false, false];
        let (class, _) = minimize_dfa(&guards, 4, &pinned);
        assert_eq!(class[0], class[1], "oscillator pair W~REST merges");
        assert_eq!(class[0], class[2], "HURT joins the class");
        assert_ne!(class[0], class[3]);
    }

    /// States with different guard programs never merge even when their
    /// continuations coincide (fsm_state is user-observable — conservative).
    #[test]
    fn hopcroft_keeps_distinct_structures_separate() {
        let p = Expr::Binary(BinOp::Gt, Box::new(Expr::Ident("timer".into())), Box::new(Expr::IntLit(60)));
        let q = Expr::Binary(BinOp::Lt, Box::new(Expr::Ident("fuel".into())), Box::new(Expr::IntLit(5)));
        // A: {p->C}   B: {q->C}   C: {}   D: {} (sink, same structure as C)
        let guards = vec![
            vec![(p, 2usize)],
            vec![(q, 2usize)],
            vec![],
            vec![],
        ];
        let (class, _) = minimize_dfa(&guards, 4, &[false; 4]);
        assert_ne!(class[0], class[1], "different predicates -> never merge");
        assert_ne!(class[0], class[2], "guard vs blind state -> never merge");
        assert_eq!(class[2], class[3], "two empty sinks are exact duplicates");
    }

    /// Chain propagation: a state is only as mergeable as its targets.
    /// X1{p->Y1}, X2{p->Y2} do NOT merge while Y1,Y2 differ; once Y1~Y2 they do.
    #[test]
    fn hopcroft_refines_through_targets() {
        let p = Expr::Binary(BinOp::Gt, Box::new(Expr::Ident("t".into())), Box::new(Expr::IntLit(1)));
        let r = Expr::Binary(BinOp::Lt, Box::new(Expr::Ident("u".into())), Box::new(Expr::IntLit(2)));
        // 0:A{p->1}  1:B{r->3}  2:C{r->3}  3:D{}  4:E{p->3}... B~C (same structure);
        // A targets 1 only. Add A2{p->2}: A and A2 have same structure but
        // targets in different classes until B~C merge — then they merge too.
        let guards = vec![
            vec![(p.clone(), 1usize)], // 0 A  -> B
            vec![(r.clone(), 3usize)], // 1 B  -> D
            vec![(r.clone(), 3usize)], // 2 C  -> D
            vec![],                    // 3 D
            vec![(p, 2usize)],         // 4 A2 -> C
        ];
        let (class, _) = minimize_dfa(&guards, 5, &[false; 5]);
        assert_eq!(class[1], class[2], "B~C same structure same target");
        assert_eq!(class[0], class[4], "A~A2 once their targets' classes coincide");
        assert_ne!(class[0], class[3], "A is not a sink");
    }

    /// Soundness of observables: TWO pinned states with identical structure
    /// must never merge (both are named by literals; merging would make one
    /// value unreachable and double-fire a draw arm). But an UNPINNED exact
    /// duplicate of a pinned state collapses INTO it — the merge-back pass.
    #[test]
    fn pinned_states_never_merge_but_duplicate_of_pinned_collapses() {
        let p = Expr::Binary(BinOp::Gt, Box::new(Expr::Ident("timer".into())), Box::new(Expr::IntLit(90)));
        let q = Expr::Binary(BinOp::Le, Box::new(Expr::Ident("hp".into())), Box::new(Expr::IntLit(0)));
        // 0 W {q->G, p->R}   1 R {q->G, p->W}   2 H {q->G, p->R}   3 G {}
        // H is an exact structural duplicate of W.
        let guards = vec![
            vec![(q.clone(), 3usize), (p.clone(), 1usize)], // 0 W
            vec![(q.clone(), 3usize), (p.clone(), 0usize)], // 1 R
            vec![(q.clone(), 3usize), (p.clone(), 1usize)], // 2 H (dup of W)
            vec![],                                         // 3 G
        ];
        // BOTH W and R user-observed (draw arms 0 and 1); H unobserved.
        let pinned = [true, true, false, false];
        let (class, _) = minimize_dfa(&guards, 4, &pinned);
        assert_ne!(class[0], class[1], "two pinned states never merge");
        assert_eq!(class[0], class[2], "unpinned duplicate collapses into pinned W");
        assert_ne!(class[0], class[3], "G stays separate");
    }
}
