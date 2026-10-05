//! ============================================================================
//! LILA OPTIMIZER — SUBSYSTEM 6: CONTROL-FLOW SUPEROPTIMIZATION (BRANCHLESS)
//! ============================================================================
//!
//! Branches inside hot per-entity code cost real time: a JZ/JMP pair is a
//! control edge the CPU must PREDICT, and a mispredict flushes the pipeline.
//! Data has no such hazard — a conditional move executes in constant time
//! regardless of the input. This pass converts the two conditional-assignment
//! shapes every game is full of into data edges:
//!
//!   if (c) { x = a; } else { x = b; }   ->  x = SEL(c, a, b)      (diamond)
//!   if (c) { x = a; }                   ->  x = SEL(c, a, x)      (cond-move)
//!
//! SEL compiles to ONE opcode (OP_SEL, 0x4E): pop else, pop then, pop cond,
//! push selected. The VM never redirects its program counter.
//!
//! SOUNDNESS CONTRACT (both rewrites):
//!   1. PURITY — the value expressions become UNCONDITIONAL, so they may not
//!      observe or mutate evaluation order: no calls at all (which bans
//!      rand() — its RNG stream position is observable), no intrinsics.
//!      Division/modulo/shifts are also excluded: the VM can trap or saturate
//!      on them, and a guard that used to skip `x/0` must keep skipping it.
//!   2. TYPE COHERENCE — both values must resolve to the same value class
//!      (bool / int / fixed) via the checker's schema, so the single SEL
//!      result converts to the target exactly once, exactly as before.
//!   3. TARGET PURITY — the cond-move reads its own target (x = SEL(c,a,x));
//!      targets are plain fields/globals, always pure reads.
//!
//! Everything here is dev-time: the emitted bytecode is v3 as before, one
//! extra opcode in the VM, zero runtime configuration.

use crate::ast::*;
use crate::checker::{Ctx, VT};

pub fn rewrite(ctx: &mut Ctx) -> usize {
    let mut total = 0usize;
    for fi in 0..ctx.fns.len() {
        let body = std::mem::take(&mut ctx.fns[fi].body);
        let (body, n) = rewrite_block(body, ctx, fi);
        ctx.fns[fi].body = body;
        total += n;
    }
    total
}

fn rewrite_block(stmts: Vec<Stmt>, ctx: &Ctx, fi: usize) -> (Vec<Stmt>, usize) {
    let mut out = Vec::with_capacity(stmts.len());
    let mut count = 0usize;
    for s in stmts {
        match s {
            Stmt::If(cond, then_b, else_b) => {
                let (then_b, n1) = rewrite_block(then_b, ctx, fi);
                let (else_b, n2) = rewrite_block(else_b, ctx, fi);
                count += n1 + n2;
                // Case A: both arms assign the SAME target once (plain Set)
                if let (Some((t1, v1)), Some((t2, v2))) = (single_set(&then_b), single_set(&else_b)) {
                    if same_target(&t1, &t2)
                        && pure_value(&v1)
                        && pure_value(&v2)
                        && value_class(ctx, &v1, fi).is_some()
                        && value_class(ctx, &v1, fi) == value_class(ctx, &v2, fi)
                    {
                        out.push(Stmt::Assign(
                            t1,
                            AssignOp::Set,
                            Expr::Intrin(INTR_SEL, 0, vec![cond, v1, v2], vec![]),
                        ));
                        count += 1;
                        continue;
                    }
                }
                // Case B: conditional move — x = SEL(c, a, x). The read of x
                // happens once, exactly where the skipped path left it.
                if else_b.is_empty() {
                    if let Some((t, v)) = single_set(&then_b) {
                        if pure_value(&v) && pure_value(&t)
                            && value_class(ctx, &v, fi).is_some()
                            && value_class(ctx, &v, fi) == value_class(ctx, &t, fi)
                        {
                            out.push(Stmt::Assign(
                                t.clone(),
                                AssignOp::Set,
                                Expr::Intrin(INTR_SEL, 0, vec![cond, v, t], vec![]),
                            ));
                            count += 1;
                            continue;
                        }
                    }
                }
                out.push(Stmt::If(cond, then_b, else_b));
            }
            Stmt::While(c, b) => {
                let (b, n) = rewrite_block(b, ctx, fi);
                count += n;
                out.push(Stmt::While(c, b));
            }
            Stmt::For(v, e, b) => {
                let (b, n) = rewrite_block(b, ctx, fi);
                count += n;
                out.push(Stmt::For(v, e, b));
            }
            Stmt::ForMask(v, e, m, b) => {
                let (b, n) = rewrite_block(b, ctx, fi);
                count += n;
                out.push(Stmt::ForMask(v, e, m, b));
            }
            other => out.push(other),
        }
    }
    (out, count)
}

/// The block is exactly one plain assignment: Some((target, value)).
fn single_set(block: &[Stmt]) -> Option<(Expr, Expr)> {
    if block.len() != 1 {
        return None;
    }
    match &block[0] {
        Stmt::Assign(t, AssignOp::Set, v) => Some((t.clone(), v.clone())),
        _ => None,
    }
}

fn same_target(a: &Expr, b: &Expr) -> bool {
    format!("{:?}", a) == format!("{:?}", b)
}

/// Purity: literals, plain idents/fields, and trap-free arithmetic only.
/// ANY call is rejected (rand() advances the shared RNG stream — its call
/// count is observable game state). Div/mod/shift can trap or saturate, so
/// a guard that used to skip them must keep skipping them.
fn pure_value(e: &Expr) -> bool {
    match e {
        Expr::IntLit(_) | Expr::FixLit(_) | Expr::BoolLit(_) | Expr::Atom(_) | Expr::Ident(_) => true,
        Expr::Field(base, _) => pure_value(base),
        Expr::Unary(_, a) => pure_value(a),
        Expr::Binary(op, a, b) => {
            let trap_free = !matches!(op, BinOp::Div | BinOp::Mod | BinOp::Shl | BinOp::Shr);
            trap_free && pure_value(a) && pure_value(b)
        }
        // calls and intrinsics: rejected wholesale (conservative)
        _ => false,
    }
}

/// Value class via the checker's schema: 0=bool, 1=int, 2=fixed/angle.
/// Returns None when the class can't be proven (rewrite skipped).
fn value_class(ctx: &Ctx, e: &Expr, fi: usize) -> Option<u8> {
    let class_of_vt = |vt: &VT| -> Option<u8> {
        match vt {
            VT::Bool => Some(0),
            VT::Int => Some(1),
            // Ang is fixed-format at VM level — bit-compatible with Fixed
            VT::Fixed | VT::Ang => Some(2),
            VT::Ent(_) => None,
        }
    };
    match e {
        Expr::IntLit(_) => Some(1),
        Expr::FixLit(_) => Some(2),
        Expr::BoolLit(_) => Some(0),
        Expr::Ident(name) => {
            let gi = *ctx.global_index.get(name)?;
            class_of_vt(&ctx.typed_fields[gi].vt)
        }
        Expr::Field(base, fname) => {
            let ename = match &**base {
                Expr::Ident(n) => n,
                _ => return None,
            };
            // this fn's param entity owns the field
            let (_, vt) = ctx.fns[fi].params.iter().find(|(n, _)| n == ename)?;
            let pei = match vt {
                VT::Ent(ei) => *ei,
                _ => return None,
            };
            let f = ctx.entities[pei].fields.iter().find(|f| &f.name == fname)?;
            class_of_vt(&f.vt)
        }
        _ => None,
    }
}

// ---------------- tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn purity_rejects_calls_and_traps() {
        let ok = Expr::Binary(BinOp::Add, Box::new(Expr::Ident("x".into())), Box::new(Expr::IntLit(1)));
        assert!(pure_value(&ok));
        let rand_call = Expr::Call("rand".into(), vec![Expr::IntLit(10)]);
        assert!(!pure_value(&rand_call), "rand() must never go unconditional");
        let div = Expr::Binary(BinOp::Div, Box::new(Expr::Ident("x".into())), Box::new(Expr::IntLit(0)));
        assert!(!pure_value(&div), "div can trap — guard must keep skipping it");
        let shl = Expr::Binary(BinOp::Shl, Box::new(Expr::Ident("x".into())), Box::new(Expr::Ident("y".into())));
        assert!(!pure_value(&shl), "shifts excluded from unconditional evaluation");
        let dist_call = Expr::Call("dist".into(), vec![]);
        assert!(!pure_value(&dist_call), "even pure calls stay conservative");
    }

    #[test]
    fn same_target_matches_field_pairs() {
        let a = Expr::Field(Box::new(Expr::Ident("d".into())), "flash".into());
        let b = Expr::Field(Box::new(Expr::Ident("d".into())), "flash".into());
        let c = Expr::Field(Box::new(Expr::Ident("d".into())), "hp".into());
        assert!(same_target(&a, &b));
        assert!(!same_target(&a, &c));
    }
}
