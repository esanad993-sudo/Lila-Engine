//! Expression typing: lvalue types, engine registers, the VT algebra
//! (int widens to fixed), and every builtin call's signature.
//! (split out of the checker monolith)

use super::*;


pub(crate) fn lvalue_type(e: &Expr, env: &[(String, VT)], f: &FnInfo, r: &CtxRefs) -> Result<(VT, String), String> {
    match e {
        Expr::Ident(name) => {
            if let Some((_, vt)) = env.iter().find(|(n, _)| n == name) {
                return Ok((*vt, format!("local '{}'", name)));
            }
            if let Some(gi) = r.global_index.get(name) {
                return Ok((r.globals[*gi].vt, format!("global '{}'", name)));
            }
            // params are in env.locals (pre-registered) — covered above
            Err(format!("unknown variable '{}'", name))
        }
        Expr::Field(base, fname) => {
            // base must be a simple variable of entity type
            match &**base {
                Expr::Ident(v) => {
                    if let Some((_, VT::Ent(ei))) = env.iter().find(|(n, _)| n == v) {
                        let fi = r.entities[*ei].fields.iter().position(|x| &x.name == fname)
                            .ok_or_else(|| format!("entity {} has no field '{}'",
                                r.entities[*ei].name, fname))?;
                        return Ok((r.entities[*ei].fields[fi].vt, format!("{}.{}", v, fname)));
                    }
                    Err(format!("'{}' is not an entity variable", v))
                }
                _ => Err("field access target must be a variable".into()),
            }
        }
        Expr::Index(base, ix) => {
            // v11 ARRAYS: `name[i] = v` / `e.arr[i] += v`. Element type from
            // the registry; literal indices bounds-proven (same as reads).
            let id = r.resolve_arr(base, env)?;
            let elem = r.check_index(id, ix, env, f)?;
            Ok((elem, "array element".to_string()))
        }
        _ => Err("invalid assignment target".into()),
    }
}

// v7: engine registers readable as expressions (declaration of these names
// is rejected at check time, so a match here is unambiguous).
pub(crate) fn engine_reg_type(name: &str) -> Option<VT> {
    match name {
        "cam_x" | "cam_y" => Some(VT::Fixed),
        _ => None,
    }
}

pub fn expr_type(e: &Expr, env: &[(String, VT)], f: &FnInfo, r: &CtxRefs) -> Result<VT, String> {
    match e {
        Expr::IntLit(_) => Ok(VT::Int),
        Expr::FixLit(_) => Ok(VT::Fixed),
        Expr::BoolLit(_) => Ok(VT::Bool),
        Expr::Atom(a) => Err(format!("line {}: atom #{} can only be used as an argument to key/sfx/music/goto/draw/draw_text/anim",
            f.line, r.aname(*a))),
        Expr::Ident(name) => {
            if let Some((_, vt)) = env.iter().find(|(n, _)| n == name) {
                return Ok(*vt);
            }
            if let Some(gi) = r.global_index.get(name) {
                return Ok(r.globals[*gi].vt);
            }
            if let Some(vt) = engine_reg_type(name) {
                return Ok(vt);
            }
            Err(format!("line {}: unknown variable '{}'", f.line, name))
        }
        Expr::Field(..) => {
            // field reads type exactly like the lvalue path (checked there)
            let (vt, _) = lvalue_type(e, env, f, r)?;
            Ok(vt)
        }
        // v11 ARRAYS: element type of the resolved array; literal indices are
        // bounds-proven here (out-of-range literals never compile).
        Expr::Index(base, ix) => {
            let id = r.resolve_arr(base, env).map_err(|e| format!("line {}: {}", f.line, e))?;
            r.check_index(id, ix, env, f)
        }
        Expr::Unary(UnOp::Neg, inner) => {
            let vt = expr_type(inner, env, f, r)?;
            match vt {
                VT::Fixed | VT::Int => Ok(vt),
                _ => Err(format!("line {}: cannot negate {:?}", f.line, vt)),
            }
        }
        Expr::Unary(UnOp::Not, inner) => {
            let vt = expr_type(inner, env, f, r)?;
            if vt == VT::Bool { Ok(VT::Bool) }
            else { Err(format!("line {}: '!' needs bool, got {:?}", f.line, vt)) }
        }
        Expr::Binary(op, a, b) => {
            let at = expr_type(a, env, f, r)?;
            let bt = expr_type(b, env, f, r)?;
            use BinOp::*;
            match op {
                Add | Sub | Mul => {
                    match (at, bt) {
                        (VT::Fixed, VT::Fixed) | (VT::Fixed, VT::Int) | (VT::Int, VT::Fixed) => Ok(VT::Fixed),
                        (VT::Int, VT::Int) => Ok(VT::Int),
                        (VT::Ang, VT::Int) | (VT::Int, VT::Ang) if *op == Add || *op == Sub => Ok(VT::Ang),
                        _ => Err(format!("line {}: bad operand types for {:?}: {:?} and {:?}", f.line, op, at, bt)),
                    }
                }
                Div | Mod => {
                    match (at, bt) {
                        (VT::Fixed, VT::Fixed) | (VT::Fixed, VT::Int) | (VT::Int, VT::Fixed) => Ok(VT::Fixed),
                        (VT::Int, VT::Int) => Ok(VT::Int),
                        _ => Err(format!("line {}: bad operand types for {:?}: {:?} and {:?}", f.line, op, at, bt)),
                    }
                }
                And | Or | Xor | Shl | Shr => {
                    match (at, bt) {
                        (VT::Int, VT::Int) => Ok(VT::Int),
                        _ => Err(format!("line {}: bitwise ops need int operands, got {:?} and {:?}", f.line, at, bt)),
                    }
                }
                Eq | Ne => {
                    match (at, bt) {
                        (VT::Fixed, VT::Fixed) | (VT::Fixed, VT::Int) | (VT::Int, VT::Fixed) => Ok(VT::Bool),
                        (VT::Int, VT::Int) | (VT::Bool, VT::Bool) | (VT::Ang, VT::Ang) | (VT::Ang, VT::Int) | (VT::Int, VT::Ang) => Ok(VT::Bool),
                        (VT::Ent(_), VT::Ent(_)) => Ok(VT::Bool),
                        _ => Err(format!("line {}: cannot compare {:?} and {:?}", f.line, at, bt)),
                    }
                }
                Lt | Gt | Le | Ge => {
                    match (at, bt) {
                        (VT::Fixed, VT::Fixed) | (VT::Fixed, VT::Int) | (VT::Int, VT::Fixed) => Ok(VT::Bool),
                        (VT::Int, VT::Int) => Ok(VT::Bool),
                        _ => Err(format!("line {}: cannot order-compare {:?} and {:?}", f.line, at, bt)),
                    }
                }
                LAnd | LOr => {
                    if at == VT::Bool && bt == VT::Bool { Ok(VT::Bool) }
                    else { Err(format!("line {}: &&/|| need bool operands, got {:?} and {:?}", f.line, at, bt)) }
                }
            }
        }
        Expr::Call(name, args) => check_builtin(name, args, env, f, r),
        // Optimizer intrinsics exist only AFTER checking (the passes inject
        // them); codegen re-types expressions containing them through this
        // recursive walker, where they carry their PROVEN result type — all
        // three intrinsics yield Int at the VM level. During real checking
        // this arm is unreachable (no Intrin exists in source).
        Expr::Intrin(..) => Ok(VT::Int),
    }
}

pub(crate) fn check_builtin(name: &str, args: &[Expr], env: &[(String, VT)], f: &FnInfo, r: &CtxRefs) -> Result<VT, String> {
    match name {
        "key" => {
            if args.len() != 1 || !matches!(args[0], Expr::Atom(_)) {
                return Err(format!("line {}: key(#atom) takes exactly one atom", f.line));
            }
            if let Expr::Atom(a) = &args[0] {
                r.domain_pos(r.keys, *a, "key")?;
            }
            Ok(VT::Bool)
        }
        "sin" | "cos" => {
            if args.len() != 1 {
                return Err(format!("line {}: {} takes one ang argument", f.line, name));
            }
            let vt = expr_type(&args[0], env, f, r)?;
            if vt != VT::Ang {
                return Err(format!("line {}: {} needs an ang argument (use ang(x)); got {:?}", f.line, name, vt));
            }
            Ok(VT::Fixed)
        }
        "ang" => {
            if args.len() != 1 {
                return Err(format!("line {}: ang() takes one argument", f.line));
            }
            let vt = expr_type(&args[0], env, f, r)?;
            match vt {
                VT::Int | VT::Fixed | VT::Ang => Ok(VT::Ang),
                _ => Err(format!("line {}: ang() needs int/fixed/ang, got {:?}", f.line, vt)),
            }
        }
        "fixed" => {
            if args.len() != 1 {
                return Err(format!("line {}: fixed() takes one argument", f.line));
            }
            let vt = expr_type(&args[0], env, f, r)?;
            match vt {
                VT::Int | VT::Fixed => Ok(VT::Fixed),
                _ => Err(format!("line {}: fixed() needs int/fixed, got {:?}", f.line, vt)),
            }
        }
        "int" => {
            if args.len() != 1 {
                return Err(format!("line {}: int() takes one argument", f.line));
            }
            let vt = expr_type(&args[0], env, f, r)?;
            match vt {
                VT::Int | VT::Fixed => Ok(VT::Int),
                _ => Err(format!("line {}: int() needs int/fixed, got {:?}", f.line, vt)),
            }
        }
        "rand" => {
            if args.len() != 1 {
                return Err(format!("line {}: rand() takes one int argument", f.line));
            }
            let vt = expr_type(&args[0], env, f, r)?;
            if vt != VT::Int {
                return Err(format!("line {}: rand() needs int argument, got {:?}", f.line, vt));
            }
            Ok(VT::Int)
        }
        "anim" => {
            if args.len() != 2 || !matches!(args[0], Expr::Atom(_)) {
                return Err(format!("line {}: anim(#atom, t) takes an atom and an int", f.line));
            }
            if let Expr::Atom(a) = &args[0] {
                r.domain_pos(r.anim_atoms, *a, "anim")?;
            }
            let vt = expr_type(&args[1], env, f, r)?;
            if vt != VT::Int {
                return Err(format!("line {}: anim() t argument must be int, got {:?}", f.line, vt));
            }
            Ok(VT::Fixed)
        }
        "count" => {
            if args.len() != 1 {
                return Err(format!("line {}: count(Entity) takes one entity name", f.line));
            }
            match &args[0] {
                Expr::Ident(ent) => {
                    if r.ent_index.contains_key(ent) { Ok(VT::Int) }
                    else { Err(format!("line {}: count() needs an entity type, got '{}'", f.line, ent)) }
                }
                _ => Err(format!("line {}: count() needs an entity type name", f.line)),
            }
        }
        // v7: SRAM persistence. save(ix, v) stores the RAW 32-bit value;
        // saved(ix) reads it back as int, savedf(ix) as fixed — the game
        // chooses the type at READ time (raw bits, no conversion emitted).
        "saved" | "savedf" => {
            if args.len() != 1 {
                return Err(format!("line {}: {}(ix) takes one int slot index (0..63)", f.line, name));
            }
            let vt = expr_type(&args[0], env, f, r)?;
            if vt != VT::Int {
                return Err(format!("line {}: {}() slot index must be int, got {:?}", f.line, name, vt));
            }
            Ok(if name == "savedf" { VT::Fixed } else { VT::Int })
        }
        // v7: distance in Q24.8 — computed in i64 internally, so it cannot
        // overflow the way hand-written dx*dx + dy*dy does in big worlds.
        "dist" => {
            if args.len() != 4 {
                return Err(format!("line {}: dist(x1, y1, x2, y2) takes four args", f.line));
            }
            for (i, a) in args.iter().enumerate() {
                let vt = expr_type(a, env, f, r)?;
                if !matches!(vt, VT::Fixed | VT::Int) {
                    return Err(format!("line {}: dist arg {} must be fixed or int, got {:?}", f.line, i + 1, vt));
                }
            }
            Ok(VT::Fixed)
        }
        // v7: screen-convention aim angle (0 = up, clockwise) of the vector
        // (dx, dy) — integer CORDIC, deterministic on both runtimes.
        "atan2" => {
            if args.len() != 2 {
                return Err(format!("line {}: atan2(dy, dx) takes two args", f.line));
            }
            for (i, a) in args.iter().enumerate() {
                let vt = expr_type(a, env, f, r)?;
                if !matches!(vt, VT::Fixed | VT::Int) {
                    return Err(format!("line {}: atan2 arg {} must be fixed or int, got {:?}", f.line, i + 1, vt));
                }
            }
            Ok(VT::Ang)
        }
        // v8 OPTIMIZER subsystem 3: swept Minkowski interval test —
        // swept_hit(ax, ay, avx, avy, bx, by, hw, hh) -> bool. Compiled to a
        // SINGLE branchless opcode (OP_SWEPT); the exact per-axis 1D solve is
        // in opt::physics's docs and unit-tested there.
        "swept_hit" => {
            if args.len() != 8 {
                return Err(format!("line {}: swept_hit(ax, ay, avx, avy, bx, by, hw, hh) takes eight args", f.line));
            }
            for (i, a) in args.iter().enumerate() {
                let vt = expr_type(a, env, f, r)?;
                if !matches!(vt, VT::Fixed | VT::Int) {
                    return Err(format!("line {}: swept_hit arg {} must be fixed or int, got {:?}", f.line, i + 1, vt));
                }
            }
            Ok(VT::Bool)
        }
        // ---------------- v11 3D ----------------
        // proj3(x, y, z) -> scale (Q24.8): perspective-project a world point
        // through the cam3 camera. Screen x/y are stashed; projx()/projy()
        // read them, projok() reports "in front of the camera". All fixed-
        // point integer math (deterministic on every runtime).
        "proj3" => {
            if args.len() != 3 {
                return Err(format!("line {}: proj3(x, y, z) takes three args", f.line));
            }
            for (i, a) in args.iter().enumerate() {
                let vt = expr_type(a, env, f, r)?;
                if !matches!(vt, VT::Fixed | VT::Int) {
                    return Err(format!("line {}: proj3 arg {} must be fixed or int, got {:?}", f.line, i + 1, vt));
                }
            }
            Ok(VT::Fixed)
        }
        "projx" | "projy" => {
            if !args.is_empty() {
                return Err(format!("line {}: {}() takes no args (reads the last proj3 stash)", f.line, name));
            }
            Ok(VT::Fixed)
        }
        "projok" => {
            if !args.is_empty() {
                return Err(format!("line {}: projok() takes no args", f.line));
            }
            Ok(VT::Bool)
        }
        // alen(name) / alen(e.arr) -> the array's compile-time capacity.
        "alen" => {
            if args.len() != 1 {
                return Err(format!("line {}: alen() takes one array name", f.line));
            }
            match &args[0] {
                Expr::Ident(name) => {
                    match r.arr_index.get(&(GLOBAL_ENT, name.clone()) ) {
                        Some(id) => {
                            if r.arrays[*id].ent != GLOBAL_ENT {
                                return Err(format!("line {}: '{}' is an entity array (use alen(e.{}))", f.line, name, name));
                            }
                            Ok(VT::Int)
                        }
                        None => Err(format!("line {}: alen(): unknown global array '{}'", f.line, name)),
                    }
                }
                Expr::Field(inner, fname) => {
                    if !matches!(&**inner, Expr::Ident(_)) {
                        return Err(format!("line {}: alen(): array base must be a variable", f.line));
                    }
                    let v = if let Expr::Ident(v) = &**inner { v } else { unreachable!() };
                    let ei = match env.iter().find(|(n, _)| n == v).map(|(_, vt)| *vt) {
                        Some(VT::Ent(ei)) => ei,
                        _ => return Err(format!("line {}: '{}' is not an entity variable", f.line, v)),
                    };
                    match r.arr_index.get(&(ei as u8, fname.clone())) {
                        Some(_) => Ok(VT::Int),
                        None => Err(format!("line {}: entity {} has no array '{}'", f.line, r.entities[ei].name, fname)),
                    }
                }
                _ => Err(format!("line {}: alen() needs an array name", f.line)),
            }
        }
        _ => Err(format!("line {}: unknown function '{}'", f.line, name)),
    }
}

