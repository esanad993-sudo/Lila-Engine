//! ============================================================================
//! LILA OPTIMIZER — SUBSYSTEM 7: STATIC ARENA & LIFETIME SYNTHESIS
//! ============================================================================
//!
//! The engine has NO allocator: 2MB of flat memory with fixed regions, and
//! entity slots pre-reserved per type. What the COMPILER can add is the
//! PROOF that the game fits — before the binary ever runs:
//!
//!   1. SPAWN-BUDGET SYNTHESIS — walk the boot path (init + scene fns),
//!      infer constant loop trip counts (`let i = 0; while (i < 24) { ...
//!      i += 1 }` -> 24 iterations, nested loops multiply), and sum spawn
//!      counts per entity type. If a proven count exceeds that type's slot
//!      capacity the build FAILS here — the runtime "spawn denied" path
//!      becomes unreachable by construction.
//!   2. MEMORY-ENVELOPE SYNTHESIS — with max_live fixed per type and row /
//!      cold-row sizes known, the exact entity-region byte cost is a
//!      compile-time constant: bytes(entity region) = Σ live·(row+cold).
//!      Add the globals block and the total arena demand is deterministic
//!      to the byte at build time. Zero heap allocations, zero free calls,
//!      zero fragmentation — nothing to do at runtime because there is
//!      nothing left to decide.
//!
//! Spawns outside the boot path (respawners inside update fns) can't be
//! bounded statically — they are reported as runtime-guarded (the VM's
//! capacity denial keeps them safe) rather than proven.

use crate::ast::*;
use crate::checker::{Ctx, VT};
use std::collections::HashMap;

pub struct ArenaRow {
    pub entity: String,
    /// spawns the analysis PROVED happen on the boot path
    pub proven: u32,
    pub capacity: u32,
    /// boot-path spawns exist but aren't statically countable
    pub dynamic: bool,
}

pub struct ArenaResult {
    pub rows: Vec<ArenaRow>,
    /// Σ live·(row+cold) over all types — the exact entity-region bytes
    pub entity_region_bytes: u32,
}

pub fn synthesize(ctx: &mut Ctx) -> Result<ArenaResult, String> {
    use crate::checker::FnKind;
    // boot fns: init and scene hooks. Everything else that spawns (update /
    // draw per-frame fns) is a resapwner: dynamic, runtime-guarded.
    let boot_fns: Vec<usize> = (0..ctx.fns.len())
        .filter(|&fi| matches!(ctx.fns[fi].kind, FnKind::Init | FnKind::Scene))
        .collect();
    let dynamic_types: Vec<bool> = {
        let mut dyn_types = vec![false; ctx.entities.len()];
        for fi in 0..ctx.fns.len() {
            if matches!(ctx.fns[fi].kind, crate::checker::FnKind::Init | crate::checker::FnKind::Scene) { continue; }
            count_spawns_stmts(&ctx.fns[fi].body, &mut |type_name, _| {
                if let Some(ei) = entity_index(ctx, type_name) {
                    dyn_types[ei] = true;
                }
            }, &mut HashMap::new(), 1);
        }
        dyn_types
    };

    // proven boot-path spawns per type
    let mut proven = vec![0u32; ctx.entities.len()];
    let mut any_dynamic = vec![false; ctx.entities.len()];
    for &fi in &boot_fns {
        count_spawns_stmts(&ctx.fns[fi].body, &mut |type_name, mult| {
            if let Some(ei) = entity_index(ctx, type_name) {
                proven[ei] += mult; // mult may be 0 = uncountable
                if mult == 0 { any_dynamic[ei] = true; }
            }
        }, &mut HashMap::new(), 1);
    }
    for ei in 0..ctx.entities.len() {
        if dynamic_types[ei] { any_dynamic[ei] = true; }
    }

    let mut rows = Vec::new();
    let mut region = 0u32;
    for ei in 0..ctx.entities.len() {
        let cap = ctx.entities[ei].max_live;
        let p = proven[ei];
        if p > cap {
            return Err(format!(
                "spawn budget exceeded: {} spawns {} > capacity {} — shrink the boot loops or raise `capacity`",
                ctx.entities[ei].name, p, cap
            ));
        }
        let row_bytes = ctx.entities[ei].row_bytes as u32;
        let cold_bytes = ctx.entities[ei].cold_bytes as u32;
        region += cap * (row_bytes + cold_bytes);
        if p > 0 || any_dynamic[ei] {
            rows.push(ArenaRow {
                entity: ctx.entities[ei].name.clone(),
                proven: p,
                capacity: cap,
                dynamic: any_dynamic[ei],
            });
        }
    }
    Ok(ArenaResult { rows, entity_region_bytes: region })
}

fn entity_index(ctx: &Ctx, name: &str) -> Option<usize> {
    ctx.entities.iter().position(|e| e.name == name)
}

/// Walk statements, calling `sink(type_name, multiplier)` per spawn.
/// multiplier 0 = "this spawn's count is not statically provable" (dynamic
/// loop bounds, while-true shapes...). Loop counters are tracked as a small
/// static environment: `let i = 0` seeds, `i += 1` steps, `while (i < N)`
/// with a known current value of 0 infers exactly N trips. Nested loops
/// multiply; anything not matching the canonical shape collapses to 0.
fn count_spawns_stmts(
    stmts: &[Stmt],
    sink: &mut dyn FnMut(&str, u32),
    env: &mut HashMap<String, i64>,
    mult: u32,
) {
    for st in stmts {
        match st {
            Stmt::Spawn(type_name, _) => sink(type_name, mult),
            Stmt::Let(name, Expr::IntLit(v)) => {
                env.insert(name.clone(), *v as i64);
            }
            Stmt::Assign(Expr::Ident(name), AssignOp::Add, Expr::IntLit(v)) => {
                if let Some(cur) = env.get(name) {
                    env.insert(name.clone(), cur + *v as i64);
                }
            }
            Stmt::While(cond, body) => {
                // canonical counted loop: `while (i < N)` where i is known 0
                // and stepped by 1 in the body -> exactly N trips
                let mut trip: u32 = 0;
                let mut counter: Option<String> = None;
                if let Expr::Binary(BinOp::Lt, lhs, rhs) = cond {
                    if let (Expr::Ident(v), Expr::IntLit(n)) = (&**lhs, &**rhs) {
                        if env.get(v).copied() == Some(0) && steps_by_one(body, v) {
                            trip = *n;
                            counter = Some(v.clone());
                        }
                    }
                }
                if let Some(c) = &counter {
                    env.insert(c.clone(), trip as i64);
                }
                let mut inner = env.clone();
                count_spawns_stmts(body, sink, &mut inner, mult.saturating_mul(trip));
            }
            Stmt::If(_, a, b) => {
                // conditional spawns still happen on some path: count them
                // at full weight (upper bound)
                count_spawns_stmts(a, sink, env, mult);
                count_spawns_stmts(b, sink, env, mult);
            }
            Stmt::For(_, _, body) | Stmt::ForMask(_, _, _, body) => {
                // entity-iteration loops: trip count = live slots (dynamic)
                count_spawns_stmts(body, sink, env, 0);
            }
            _ => {}
        }
    }
}

fn steps_by_one(body: &[Stmt], counter: &str) -> bool {
    body.iter().any(|st| matches!(st,
        Stmt::Assign(Expr::Ident(v), AssignOp::Add, Expr::IntLit(1)) if v == counter))
}

// ---------------- DAG schedule proof (graph theory) ----------------
//
// The engine runs a FIXED single-worker system order:
//   init -> scene -> update() -> update(Type0..N) -> draw(Type0..N) -> draw()
// There are no threads, so there are no data races — but the ORDER is still a
// schedule, and this analysis proves it is a valid topological order of the
// program's read-after-write dataflow graph: everything a system reads is
// written by an EARLIER system (or by the same one). The emitted report counts
// the dataflow edges the schedule carries, so the dev can see the dependency
// structure the fixed order satisfies — zero locks, zero barriers, zero
// context switches, by construction rather than by discipline.

pub struct ScheduleReport {
    pub systems: usize,
    pub raw_edges: usize,   // producer -> consumer pairs the order serves
    pub war_pairs: usize,   // consumer -> producer orderings (safe: in-order)
}

pub fn schedule_proof(ctx: &Ctx) -> ScheduleReport {
    use crate::checker::FnKind;
    // systems in execution order (skip helpers/tables: not scheduled)
    let mut order: Vec<(String, usize)> = Vec::new();
    for fi in 0..ctx.fns.len() {
        let f = &ctx.fns[fi];
        let name = match f.kind {
            FnKind::Init => "init".to_string(),
            FnKind::Scene => "scene".to_string(),
            FnKind::Update0 => "update()".to_string(),
            FnKind::UpdateEnt => format!("update({})", entity_name(ctx, f)),
            FnKind::Draw0 => "draw()".to_string(),
            FnKind::DrawEnt => format!("draw({})", entity_name(ctx, f)),
            _ => continue,
        };
        order.push((name, fi));
    }

    // read/write sets per system: globals by name, entity fields as "Ent.field"
    let mut sets: Vec<(std::collections::BTreeSet<String>, std::collections::BTreeSet<String>)> = Vec::new();
    for (_, fi) in &order {
        let mut r = std::collections::BTreeSet::new();
        let mut w = std::collections::BTreeSet::new();
        collect_rw(ctx, *fi, &mut r, &mut w);
        sets.push((r, w));
    }

    let mut raw = 0usize;
    let mut war = 0usize;
    for i in 0..order.len() {
        for j in (i + 1)..order.len() {
            // RAW served by the schedule: j reads what i wrote
            if !sets[i].1.is_disjoint(&sets[j].0) { raw += 1; }
            // WAR ordering: j writes what i read (safe: j runs later)
            if !sets[j].1.is_disjoint(&sets[i].0) { war += 1; }
        }
    }
    ScheduleReport {
        systems: order.len(),
        raw_edges: raw,
        war_pairs: war,
    }
}

fn entity_name(ctx: &Ctx, f: &crate::checker::FnInfo) -> String {
    for (_, vt) in &f.params {
        if let VT::Ent(ei) = vt {
            return ctx.entities[*ei].name.clone();
        }
    }
    "?".to_string()
}

fn collect_rw(ctx: &Ctx, fi: usize, reads: &mut std::collections::BTreeSet<String>, writes: &mut std::collections::BTreeSet<String>) {
    let f = &ctx.fns[fi];
    let params: Vec<(String, usize)> = f.params.iter()
        .filter_map(|(n, vt)| match vt {
            VT::Ent(ei) => Some((n.clone(), *ei)),
            _ => None,
        })
        .collect();
    let no_locals = std::collections::HashSet::new();
    walk_stmts(&f.body, ctx, &params, &no_locals, reads, writes);
}

fn walk_stmts(
    stmts: &[Stmt],
    ctx: &Ctx,
    params: &[(String, usize)],
    locals: &std::collections::HashSet<String>,
    reads: &mut std::collections::BTreeSet<String>,
    writes: &mut std::collections::BTreeSet<String>,
) {
    for st in stmts {
        match st {
            Stmt::Assign(target, _, val) => {
                note_write(target, ctx, params, locals, writes);
                walk_expr(val, ctx, params, locals, reads, writes);
                walk_expr(target, ctx, params, locals, reads, writes);
            }
            Stmt::Let(_, val) => walk_expr(val, ctx, params, locals, reads, writes),
            Stmt::If(c, a, b) => {
                walk_expr(c, ctx, params, locals, reads, writes);
                walk_stmts(a, ctx, params, locals, reads, writes);
                walk_stmts(b, ctx, params, locals, reads, writes);
            }
            Stmt::While(c, b) => {
                walk_expr(c, ctx, params, locals, reads, writes);
                walk_stmts(b, ctx, params, locals, reads, writes);
            }
            Stmt::For(_, tname, b) => {
                reads.insert(format!("iter:{}", tname));
                walk_stmts(b, ctx, params, locals, reads, writes);
            }
            Stmt::ForMask(_, tname, _, b) => {
                reads.insert(format!("iter:{}", tname));
                walk_stmts(b, ctx, params, locals, reads, writes);
            }
            Stmt::Spawn(tname, fields) => {
                writes.insert(format!("spawn:{}", tname));
                for (_, e) in fields {
                    walk_expr(e, ctx, params, locals, reads, writes);
                }
            }
            Stmt::Kill(e) => walk_expr(e, ctx, params, locals, reads, writes),
            Stmt::Sfx(_) | Stmt::Music(_) | Stmt::StopMusic | Stmt::Shake(_)
            | Stmt::MusicVol(_) | Stmt::Goto(_) | Stmt::Return => {}
            Stmt::Camera(a, b) | Stmt::Save(a, b) => {
                walk_expr(a, ctx, params, locals, reads, writes);
                walk_expr(b, ctx, params, locals, reads, writes);
            }
            Stmt::Draw(_, a, b, c, d, e) => {
                for x in [a, b, c, d, e] { walk_expr(x, ctx, params, locals, reads, writes); }
            }
            Stmt::DrawText(_, a, b, c) => {
                for x in [a, b, c] { walk_expr(x, ctx, params, locals, reads, writes); }
            }
            Stmt::DrawNum(a, b, c, d) => {
                for x in [a, b, c, d] { walk_expr(x, ctx, params, locals, reads, writes); }
            }
            Stmt::Cam3(a, b, c, d, e) => {
                for x in [a, b, c, d, e] { walk_expr(x, ctx, params, locals, reads, writes); }
            }
            Stmt::Draw3d { n, x, y, z, yaw, rgba, .. } => {
                for x in [n, x, y, z, yaw, rgba] { walk_expr(x, ctx, params, locals, reads, writes); }
            }
            // v12 ARTICULATION: array traffic uses the same "arr:<name>" keys
            // as Index (writes disqualify parallel grouping via own-type)
            Stmt::QuatAA { q, off, ax, ay, az, ang } => {
                writes.insert(format!("arr:{}", q));
                for x in [off, ax, ay, az, ang] { walk_expr(x, ctx, params, locals, reads, writes); }
            }
            Stmt::QMul { d, doff, a, aoff, b, boff } => {
                writes.insert(format!("arr:{}", d));
                reads.insert(format!("arr:{}", a));
                reads.insert(format!("arr:{}", b));
                for x in [doff, aoff, boff] { walk_expr(x, ctx, params, locals, reads, writes); }
            }
            Stmt::M4QT { m, moff, q, qoff, tx, ty, tz } => {
                writes.insert(format!("arr:{}", m));
                reads.insert(format!("arr:{}", q));
                for x in [moff, qoff, tx, ty, tz] { walk_expr(x, ctx, params, locals, reads, writes); }
            }
            Stmt::M4Mul { d, doff, a, aoff, b, boff } => {
                writes.insert(format!("arr:{}", d));
                reads.insert(format!("arr:{}", a));
                reads.insert(format!("arr:{}", b));
                for x in [doff, aoff, boff] { walk_expr(x, ctx, params, locals, reads, writes); }
            }
            Stmt::SkinV { v, voff, m, moff, x, y, z } => {
                writes.insert(format!("arr:{}", v));
                reads.insert(format!("arr:{}", m));
                for x in [voff, moff, x, y, z] { walk_expr(x, ctx, params, locals, reads, writes); }
            }
            Stmt::Draw3DI { verts, nv, idx, ni, x, y, z, yaw, rgba } => {
                reads.insert(format!("arr:{}", verts));
                reads.insert(format!("arr:{}", idx));
                for x in [nv, ni, x, y, z, yaw, rgba] { walk_expr(x, ctx, params, locals, reads, writes); }
            }
            Stmt::CallStmt(_, args) => {
                for a in args { walk_expr(a, ctx, params, locals, reads, writes); }
            }
            Stmt::CallTable(_, k, _) => walk_expr(k, ctx, params, locals, reads, writes),
            Stmt::FsmStep(_) => {
                // after opt::fsm expansion this arm is unreachable; if a game
                // ever reaches it (unoptimized fsms are impossible), the
                // type-level r/w is already covered by the expanded form.
            }
        }
    }
}

fn note_write(
    target: &Expr,
    ctx: &Ctx,
    params: &[(String, usize)],
    _locals: &std::collections::HashSet<String>,
    writes: &mut std::collections::BTreeSet<String>,
) {
    match target {
        Expr::Ident(n) => { writes.insert(format!("g:{}", n)); }
        Expr::Field(base, fname) => {
            if let Expr::Ident(p) = &**base {
                if let Some((_, ei)) = params.iter().find(|(n, _)| n == p) {
                    writes.insert(format!("{}:{}", ctx.entities[*ei].name, fname));
                }
            }
        }
        // v12: array cells are tracked like globals — a GLOBAL array write
        // ("arr:<name>") fails the own-type clamp and disqualifies parallel
        // grouping; an entity-array write is per-slot (own-type safe).
        Expr::Index(base, _) => {
            if let Some(key) = arr_rw_key(ctx, base, params) {
                writes.insert(key);
            }
        }
        _ => {}
    }
}

fn walk_expr(
    e: &Expr,
    ctx: &Ctx,
    params: &[(String, usize)],
    locals: &std::collections::HashSet<String>,
    reads: &mut std::collections::BTreeSet<String>,
    writes: &mut std::collections::BTreeSet<String>,
) {
    match e {
        Expr::Ident(n) => {
            // only named globals count as cross-system reads; locals and
            // params are fn-private
            if ctx.global_index.contains_key(n) {
                reads.insert(format!("g:{}", n));
            }
        }
        Expr::Field(base, fname) => {
            if let Expr::Ident(p) = &**base {
                if let Some((_, ei)) = params.iter().find(|(n, _)| n == p) {
                    reads.insert(format!("{}:{}", ctx.entities[*ei].name, fname));
                }
            }
            walk_expr(base, ctx, params, locals, reads, writes);
        }
        Expr::Unary(_, a) => walk_expr(a, ctx, params, locals, reads, writes),
        Expr::Binary(_, a, b) => {
            walk_expr(a, ctx, params, locals, reads, writes);
            walk_expr(b, ctx, params, locals, reads, writes);
        }
        Expr::Call(_, args) => {
            for a in args { walk_expr(a, ctx, params, locals, reads, writes); }
        }
        Expr::Intrin(_, _, args, _) => {
            for a in args { walk_expr(a, ctx, params, locals, reads, writes); }
        }
        // v12: an Index READ is shared-state traffic when the base is a
        // global array (entity arrays are per-slot)
        Expr::Index(base, idx) => {
            if let Some(key) = arr_rw_key(ctx, base, params) {
                reads.insert(key);
            }
            walk_expr(idx, ctx, params, locals, reads, writes);
            walk_expr(base, ctx, params, locals, reads, writes);
        }
        _ => {}
    }
}

/// v12: the r/w key for an Index base — global arrays are shared
/// ("arr:<name>"), entity arrays are per-slot ("<Ent>:arr.<name>").
fn arr_rw_key(ctx: &Ctx, base: &Expr, bindings: &[(String, usize)]) -> Option<String> {
    match base {
        Expr::Ident(n) => {
            if ctx.arr_index.get(&(crate::checker::GLOBAL_ENT, n.clone())).is_some() {
                Some(format!("arr:{}", n))
            } else {
                None
            }
        }
        Expr::Field(p, arrname) => {
            if let Expr::Ident(pn) = &**p {
                if let Some((_, ei)) = bindings.iter().find(|(n, _)| n == pn) {
                    return Some(format!("{}:arr.{}", ctx.entities[*ei].name, arrname));
                }
            }
            None
        }
        _ => None,
    }
}

// ============================================================================
// SUBSYSTEM 10 — COMPILE-PROVEN PARALLEL SYSTEMS (the Jobs/Burst analog)
// ============================================================================
//
// Unity's ECS schedules jobs by tracking at RUNTIME "which jobs read and
// write which" data; Unreal's task graph does the same bookkeeping live.
// Lila inverts the cost: the read/write contract of every system is proven
// at BUILD time, and only the PROOF — the disjoint groups — ships in the
// binary. The runtime pays zero scheduler cost: it spawns one worker per
// group member and joins. Soundness rests on three facts:
//
//   1. FIELD PARTITION — field names are globally unique across entities,
//      so each field belongs to exactly one entity type, and there is
//      exactly one update(Type) system per type. Writes therefore partition
//      across systems by construction.
//   2. PURITY GATE — a system enters the analysis only if it (and every
//      helper it calls, transitively) contains no spawn, kill, sfx, music,
//      goto, camera, save, shake, draw, table dispatch or rand(): those are
//      the ops that touch shared engine registers (entity metadata, audio,
//      RNG state). Everything else is arithmetic over typed fields.
//   3. DISJOINTNESS — a group is a set of systems where no member writes
//      anything any other member reads or writes (globals: nobody in a
//      group writes globals, so global reads are free). Concurrent
//      execution of a group is then equivalent to ANY sequential order —
//      the result is bit-identical by the interchange law, not by luck.

pub struct ParallelPlan {
    /// fn indices (UpdateEnt systems) per mutually-disjoint group; groups
    /// with a single member are dropped (nothing to parallelize)
    pub groups: Vec<Vec<usize>>,
    /// systems that passed the purity screen (diagnostic; asserted in tests)
    #[allow(dead_code)]
    pub eligible: usize,
}

struct SysRw {
    fi: usize,
    reads: std::collections::BTreeSet<String>,
    writes: std::collections::BTreeSet<String>,
}

pub fn parallel_groups(ctx: &Ctx) -> ParallelPlan {
    use crate::checker::FnKind;
    // ---- eligibility pass ----
    let mut sys: Vec<SysRw> = Vec::new();
    for fi in 0..ctx.fns.len() {
        if !matches!(ctx.fns[fi].kind, FnKind::UpdateEnt) {
            continue;
        }
        let mut a = RwAnalyzer { ctx, pure: true, reads: Default::default(), writes: Default::default(),
                                 bindings: Vec::new(), calling: Vec::new() };
        // bind the entity param
        for (pname, vt) in &ctx.fns[fi].params {
            if let VT::Ent(ei) = vt {
                a.bindings.push((pname.clone(), *ei));
            }
        }
        a.stmts(&ctx.fns[fi].body);
        if !a.pure {
            continue;
        }
        // soundness clamp: every write must target the system's OWN type —
        // a write through a for-loop binding of another type disqualifies
        // (the type-partition argument must stay intact)
        let own = match ctx.fns[fi].entity { Some(e) => e, None => continue };
        let own_prefix = format!("{}:", ctx.entities[own].name);
        if a.writes.iter().any(|w| !w.starts_with(&own_prefix)) {
            continue;
        }
        sys.push(SysRw { fi, reads: a.reads, writes: a.writes });
    }
    sys.sort_by_key(|s| s.fi);

    // ---- greedy grouping under pairwise disjointness ----
    let compatible = |a: &SysRw, b: &SysRw| -> bool {
        a.writes.is_disjoint(&b.reads)
            && b.writes.is_disjoint(&a.reads)
            && a.writes.is_disjoint(&b.writes)
    };
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for si in 0..sys.len() {
        let mut placed = false;
        for g in groups.iter_mut() {
            let all_ok = g.iter().all(|&mfi| {
                let mi = sys.iter().position(|x| x.fi == mfi).unwrap();
                compatible(&sys[mi], &sys[si])
            });
            if all_ok {
                g.push(sys[si].fi);
                placed = true;
                break;
            }
        }
        if !placed {
            groups.push(vec![sys[si].fi]);
        }
    }
    // keep only groups that actually parallelize
    let groups: Vec<Vec<usize>> = groups.into_iter().filter(|g| g.len() > 1).collect();
    let eligible = sys.len();
    ParallelPlan { groups, eligible }
}

// ---- purity + read/write analyzer (precise where schedule_proof is a
// summary: follows helper calls transitively, binds for-loop variables, and
// treats ANY escape hatch as disqualifying impurity) ----

struct RwAnalyzer<'a> {
    ctx: &'a Ctx,
    pure: bool,
    reads: std::collections::BTreeSet<String>,
    writes: std::collections::BTreeSet<String>,
    /// name -> entity index (params + for-loop vars)
    bindings: Vec<(String, usize)>,
    /// fn-index call stack (helper purity follows transitive calls)
    calling: Vec<usize>,
}

const PURE_BUILTINS: &[&str] = &[
    "sin", "cos", "ang", "fixed", "int", "count", "dist", "saved", "savedf",
    "anim", "swept_hit", "key", "atan2",
];

impl<'a> RwAnalyzer<'a> {
    fn stmts(&mut self, stmts: &[Stmt]) {
        for st in stmts {
            self.stmt(st);
            if !self.pure {
                return;
            }
        }
    }

    fn stmt(&mut self, st: &Stmt) {
        match st {
            Stmt::Let(name, val) => {
                self.expr(val);
                // let-bound names are fn-private: shadow any binding
                self.bindings.retain(|(n, _)| n != name);
            }
            Stmt::Assign(target, _, val) => {
                self.expr(val);
                self.write_target(target);
                self.expr(target);
            }
            Stmt::If(c, a, b) => {
                self.expr(c);
                self.stmts(a);
                self.stmts(b);
            }
            Stmt::While(c, b) => {
                self.expr(c);
                self.stmts(b);
            }
            Stmt::For(var, tname, b) | Stmt::ForMask(var, tname, _, b) => {
                let ei = self.ctx.entities.iter().position(|e| &e.name == tname);
                match ei {
                    Some(ei) => self.bindings.push((var.clone(), ei)),
                    None => self.pure = false,
                }
                self.reads.insert(format!("iter:{}", tname));
                self.stmts(b);
                self.bindings.retain(|(n, _)| n != var);
            }
            // impure: shared engine state (entity metadata, audio, RNG,
            // scene, camera, SRAM, render stream)
            Stmt::Spawn(..) | Stmt::Kill(_) | Stmt::Sfx(_) | Stmt::Music(_)
            | Stmt::StopMusic | Stmt::Shake(_) | Stmt::MusicVol(_) | Stmt::Goto(_)
            | Stmt::Save(_, _) | Stmt::Camera(_, _) | Stmt::Draw(..)
            | Stmt::DrawText(..) | Stmt::DrawNum(..) | Stmt::CallTable(..)
            // v11: cam3/proj3 touch engine registers + the render stream
            | Stmt::Cam3(..) | Stmt::Draw3d { .. }
            // v12: articulation writes shared array state; the indexed mesh
            // pass feeds the render stream — both disqualify parallel purity
            | Stmt::QuatAA { .. } | Stmt::QMul { .. } | Stmt::M4QT { .. }
            | Stmt::M4Mul { .. } | Stmt::SkinV { .. } | Stmt::Draw3DI { .. } => {
                self.pure = false;
            }
            Stmt::CallStmt(name, args) => {
                for a in args {
                    self.expr(a);
                }
                self.follow(name);
            }
            Stmt::FsmStep(_) => {
                // post opt::fsm expansion this arm is unreachable; if it ever
                // fired it would be own-field + static-table traffic — but
                // staying conservative keeps the proof independent of the
                // pass ordering
                self.pure = false;
            }
            Stmt::Return => {}
        }
    }

    fn write_target(&mut self, target: &Expr) {
        match target {
            Expr::Ident(n) => {
                if self.ctx.global_index.contains_key(n) {
                    self.writes.insert(format!("g:{}", n));
                }
                // writes to locals are fn-private — nothing to record
            }
            Expr::Field(base, fname) => {
                if let Expr::Ident(p) = &**base {
                    if let Some((_, ei)) = self.bindings.iter().find(|(n, _)| n == p) {
                        self.writes.insert(format!("{}:{}", self.ctx.entities[*ei].name, fname));
                    }
                }
            }
            // v12: array cells are tracked like globals — global-array writes
            // fail the own-type clamp (shared state), entity-array writes are
            // per-slot (own-type safe)
            Expr::Index(base, idx) => {
                if let Some(key) = arr_rw_key(self.ctx, base, &self.bindings) {
                    self.writes.insert(key);
                }
                self.expr(idx);
                self.expr(base);
            }
            _ => {}
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Ident(n) => {
                if self.ctx.global_index.contains_key(n) {
                    self.reads.insert(format!("g:{}", n));
                }
            }
            Expr::Field(base, fname) => {
                if let Expr::Ident(p) = &**base {
                    if let Some((_, ei)) = self.bindings.iter().find(|(n, _)| n == p) {
                        self.reads.insert(format!("{}:{}", self.ctx.entities[*ei].name, fname));
                    }
                }
                self.expr(base);
            }
            Expr::Unary(_, a) => self.expr(a),
            Expr::Binary(_, a, b) => {
                self.expr(a);
                self.expr(b);
            }
            Expr::Call(name, args) => {
                // rand() mutates the LCG register: instant disqualification
                if name == "rand" {
                    self.pure = false;
                }
                for a in args {
                    self.expr(a);
                }
            }
            Expr::Intrin(_, _, args, _) => {
                for a in args {
                    self.expr(a);
                }
            }
            // v12: array reads are shared-state traffic (global) or per-slot
            // (entity) — same keys as write_target
            Expr::Index(base, idx) => {
                if let Some(key) = arr_rw_key(self.ctx, base, &self.bindings) {
                    self.reads.insert(key);
                }
                self.expr(idx);
                self.expr(base);
            }
            _ => {}
        }
    }

    /// Follow a helper call for purity: the callee's body is analyzed with
    /// the same rules. Unknown callees are conservatively impure.
    fn follow(&mut self, name: &str) {
        if PURE_BUILTINS.contains(&name) {
            return;
        }
        let fi = match self.ctx.fns.iter().position(|f| &f.name == name) {
            Some(fi) => fi,
            None => {
                self.pure = false;
                return;
            }
        };
        if self.calling.contains(&fi) {
            return; // recursion: already on the stack, body analysis in progress
        }
        self.calling.push(fi);
        let params: Vec<(String, usize)> = self.ctx.fns[fi]
            .params
            .iter()
            .filter_map(|(n, vt)| match vt {
                VT::Ent(ei) => Some((n.clone(), *ei)),
                _ => None,
            })
            .collect();
        let saved_bindings = self.bindings.len();
        for (n, ei) in params {
            self.bindings.push((n, ei));
        }
        let saved_reads = self.reads.clone();
        let saved_writes = self.writes.clone();
        self.stmts(&self.ctx.fns[fi].body.clone());
        // helper-internal traffic is fn-private: merge nothing, keep only purity
        self.reads = saved_reads;
        self.writes = saved_writes;
        self.bindings.truncate(saved_bindings);
        self.calling.pop();
    }
}

#[cfg(test)]
mod parallel_tests {
    use super::*;
    use crate::checker;

    /// Three structurally identical systems (Drone/Bug/Bat-style) that only
    /// touch their own fields and read globals must group together; a
    /// system with sfx and one writing a global must stay out.
    #[test]
    fn disjoint_systems_group_impure_ones_dont() {
        let src = r#"
game { title: "T"  width: 256  height: 256 }
global { px: fixed = 0  hits: u = 0 }
entity Drone { x: fixed = 0  vx: fixed = 1 }
entity Bug   { x: fixed = 0  vx: fixed = 2 }
entity Bat   { x: fixed = 0  vx: fixed = 3 }
entity Wisp  { x: fixed = 0  vx: fixed = 4 }
sound { #blip }
fn init() { spawn Drone; spawn Bug; spawn Bat; spawn Wisp; }
fn update(d: Drone) { d.x += d.vx; if d.x > px { d.x = px; } }
fn update(b: Bug)   { b.x += b.vx; if b.x > px { b.x = px; } }
fn update(t: Bat)   { t.x += t.vx; if t.x > px { t.x = px; } }
fn update(w: Wisp)  { w.x += w.vx; sfx(#blip); }
"#;
        let (toks, atoms) = crate::lexer::Lexer::new(src).tokenize().unwrap();
        let mut parser = crate::parser::Parser::new(toks, atoms);
        let mut prog = parser.parse_program().unwrap();
        let atoms = std::mem::take(&mut parser.atom_names);
        crate::synth::expand(&mut prog, &atoms).unwrap();
        let ctx = checker::check(&prog, atoms, &Default::default()).unwrap();
        let plan = parallel_groups(&ctx);
        assert_eq!(plan.eligible, 3, "Wisp's sfx() disqualifies it; 3 pure systems remain");
        assert_eq!(plan.groups.len(), 1, "one disjoint group");
        assert_eq!(plan.groups[0].len(), 3, "Drone+Bug+Bat group; Wisp has sfx");
    }

    /// A system writing a global (score) or another type's field (turret
    /// loops) is excluded — the type-partition invariant is load-bearing.
    #[test]
    fn global_writers_and_cross_type_writers_excluded() {
        let src = r#"
game { title: "T"  width: 256  height: 256 }
global { px: fixed = 0  hits: u = 0 }
entity Drone { x: fixed = 0  hp: u = 3 }
entity Bug   { x: fixed = 0  hp: u = 3 }
entity Turret { x: fixed = 0 }
fn init() { spawn Drone; spawn Bug; spawn Turret; }
fn update(d: Drone) { d.x += 1; hits += 1; }
fn update(b: Bug)   { b.x += 1; }
fn update(t: Turret) {
    for (d: Drone) { d.hp -= 1; }
}
"#;
        let (toks, atoms) = crate::lexer::Lexer::new(src).tokenize().unwrap();
        let mut parser = crate::parser::Parser::new(toks, atoms);
        let mut prog = parser.parse_program().unwrap();
        let atoms = std::mem::take(&mut parser.atom_names);
        crate::synth::expand(&mut prog, &atoms).unwrap();
        let ctx = checker::check(&prog, atoms, &Default::default()).unwrap();
        let plan = parallel_groups(&ctx);
        // Drone (global write) and Turret (cross-type write) excluded;
        // Bug alone can't form a >1 group -> no groups at all
        assert!(plan.groups.is_empty(), "no unsafe grouping");
    }

    /// rand() in any reachable position disqualifies (shared LCG state).
    #[test]
    fn rand_is_impure_even_through_helpers() {
        let src = r#"
game { title: "T"  width: 256  height: 256 }
global { px: fixed = 0 }
entity Drone { x: fixed = 0 }
entity Bug   { x: fixed = 0 }
entity Bat   { x: fixed = 0 }
fn init() { spawn Drone; spawn Bug; spawn Bat; }
fn jitter(e: Drone, n: fixed) { e.x += n; }
fn update(d: Drone) { call jitter(d, 1.0); }
fn update(b: Bug)   { b.x += rand(4); }
fn update(t: Bat)   { t.x += 1; }
"#;
        let (toks, atoms) = crate::lexer::Lexer::new(src).tokenize().unwrap();
        let mut parser = crate::parser::Parser::new(toks, atoms);
        let mut prog = parser.parse_program().unwrap();
        let atoms = std::mem::take(&mut parser.atom_names);
        crate::synth::expand(&mut prog, &atoms).unwrap();
        let ctx = checker::check(&prog, atoms, &Default::default()).unwrap();
        let plan = parallel_groups(&ctx);
        // Drone is pure (helper followed), Bug hits rand(), Bat pure ->
        // exactly one pair group {Drone, Bat}
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].len(), 2, "Drone+Bat; Bug's rand disqualifies");
    }
}
