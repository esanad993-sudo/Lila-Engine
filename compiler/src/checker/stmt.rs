//! Statement checking: per-fn body rules, placement (draw fns), spawn
//! field typing, v11/v12 statement validation, assignment compatibility.
//! (split out of the checker monolith)

use super::*;
use super::expr::{expr_type, lvalue_type};

pub(crate) fn check_block(stmts: &[Stmt], env: &mut FnEnv, f: &FnInfo, r: &CtxRefs) -> Result<(), String> {
    for s in stmts {
        check_stmt(s, env, f, r)?;
    }
    Ok(())
}

pub(crate) fn is_draw_fn(f: &FnInfo) -> bool {
    matches!(f.kind, FnKind::Draw0 | FnKind::DrawEnt)
}

pub(crate) fn check_stmt(s: &Stmt, env: &mut FnEnv, f: &FnInfo, r: &CtxRefs) -> Result<(), String> {
    match s {
        Stmt::Let(name, e) => {
            if name == "cam_x" || name == "cam_y" {
                return Err(format!("line {}: local '{}' is reserved (engine camera registers)", f.line, name));
            }
            let vt = expr_type(e, &env.locals, f, r)?;
            // `let` rebinds (Python-style): an existing local of the same name
            // is overwritten and re-typed; VM slots are untyped i32.
            if let Some(slot) = env.locals.iter_mut().find(|(n, _)| n == name) {
                slot.1 = vt;
            } else {
                env.locals.push((name.clone(), vt));
            }
            Ok(())
        }
        Stmt::Assign(target, op, val) => {
            let (tvt, desc) = lvalue_type(target, &env.locals, f, r)?;
            let vvt = expr_type(val, &env.locals, f, r)?;
            check_assign_compat(tvt, vvt, *op, &desc, f.line)?;
            Ok(())
        }
        Stmt::If(cond, a, b) => {
            let cvt = expr_type(cond, &env.locals, f, r)?;
            if cvt != VT::Bool {
                return Err(format!("line {}: if condition must be bool, got {:?}", f.line, cvt));
            }
            check_block(a, env, f, r)?;
            check_block(b, env, f, r)?;
            Ok(())
        }
        Stmt::While(cond, body) => {
            let cvt = expr_type(cond, &env.locals, f, r)?;
            if cvt != VT::Bool {
                return Err(format!("line {}: while condition must be bool, got {:?}", f.line, cvt));
            }
            check_block(body, env, f, r)?;
            Ok(())
        }
        Stmt::For(var, ent, body) => {
            let ei = *r.ent_index.get(ent)
                .ok_or_else(|| format!("line {}: for-loop over unknown entity '{}'", f.line, ent))?;
            env.locals.push((var.clone(), VT::Ent(ei)));
            check_block(body, env, f, r)?;
            env.locals.pop();
            Ok(())
        }
        Stmt::Spawn(ent, inits) => {
            if is_draw_fn(f) {
                return Err(format!("line {}: spawn is not allowed in draw fns", f.line));
            }
            let ei = *r.ent_index.get(ent)
                .ok_or_else(|| format!("line {}: spawn of unknown entity '{}'", f.line, ent))?;
            for (fname, e) in inits {
                let fi = r.entities[ei].fields.iter().position(|x| &x.name == fname)
                    .ok_or_else(|| format!("line {}: entity {} has no field '{}'", f.line, ent, fname))?;
                let fvt = r.entities[ei].fields[fi].vt;
                let vvt = expr_type(e, &env.locals, f, r)?;
                check_assign_compat(fvt, vvt, AssignOp::Set, &format!("spawn {}.{}", ent, fname), f.line)?;
            }
            Ok(())
        }
        Stmt::Kill(e) => {
            if is_draw_fn(f) {
                return Err(format!("line {}: kill is not allowed in draw fns", f.line));
            }
            let vt = expr_type(e, &env.locals, f, r)?;
            match vt {
                VT::Ent(_) => Ok(()),
                _ => Err(format!("line {}: kill needs an entity reference, got {:?}", f.line, vt)),
            }
        }
        Stmt::Sfx(a) => {
            r.domain_pos(r.sound, *a, "sound")?;
            Ok(())
        }
        Stmt::Music(a) => {
            r.domain_pos(r.music_atoms, *a, "music")?;
            Ok(())
        }
        Stmt::StopMusic => Ok(()),
        Stmt::Shake(_) => Ok(()), // range enforced by parser (literal 0..255)
        Stmt::Camera(x, y) => {
            // v7: presentation-only — allowed in draw fns (like Unity moving
            // the camera in any pass); the write is clamped to the world.
            check_num_arg(x, env, f, r, "camera x")?;
            check_num_arg(y, env, f, r, "camera y")?;
            Ok(())
        }
        Stmt::Cam3(x, y, z, yaw, pitch) => {
            // v11 3D camera: gameplay-aimable presentation. x/y/z fixed|int,
            // yaw/pitch ang|int-literal (same rule as draw rot).
            check_num_arg(x, env, f, r, "cam3 x")?;
            check_num_arg(y, env, f, r, "cam3 y")?;
            check_num_arg(z, env, f, r, "cam3 z")?;
            for (e, what) in [(yaw, "cam3 yaw"), (pitch, "cam3 pitch")] {
                let vt = expr_type(e, &env.locals, f, r)?;
                match vt {
                    VT::Ang => {}
                    VT::Int => {
                        if !matches!(e, Expr::IntLit(_)) {
                            return Err(format!("line {}: {} must be ang-typed (or an int literal); use ang()", f.line, what));
                        }
                    }
                    _ => return Err(format!("line {}: {} must be ang", f.line, what)),
                }
            }
            Ok(())
        }
        Stmt::Draw3d { arr, n, x, y, z, yaw, rgba } => {
            // v11 3D mesh pass: DRAW-PASS ONLY (it emits into the per-frame
            // vertex stream — from update fns the verts would be reset away).
            if !is_draw_fn(f) {
                return Err(format!("line {}: draw3d is a draw-pass statement (move it into draw())", f.line));
            }
            let id = r.arr_index.get(&(GLOBAL_ENT, arr.clone())).copied()
                .ok_or_else(|| format!("line {}: draw3d mesh '{}' is not a global array (meshes are shared geometry)", f.line, arr))?;
            let nvt = expr_type(n, &env.locals, f, r)?;
            if nvt != VT::Int {
                return Err(format!("line {}: draw3d vertex count must be int, got {:?}", f.line, nvt));
            }
            if let Expr::IntLit(v) = n {
                if v > &768 {
                    return Err(format!("line {}: draw3d vertex count {} exceeds the per-call budget 768 (256 triangles)", f.line, v));
                }
                if (*v as u64) * 3 > r.arrays[id].cap as u64 {
                    return Err(format!("line {}: draw3d needs {} x 3 cells but array '{}' holds only {}", f.line, v, arr, r.arrays[id].cap));
                }
            }
            check_num_arg(x, env, f, r, "draw3d x")?;
            check_num_arg(y, env, f, r, "draw3d y")?;
            check_num_arg(z, env, f, r, "draw3d z")?;
            let rvt = expr_type(yaw, &env.locals, f, r)?;
            match rvt {
                VT::Ang => {}
                VT::Int => {
                    if !matches!(yaw, Expr::IntLit(_)) {
                        return Err(format!("line {}: draw3d yaw must be ang-typed (or an int literal); use ang()", f.line));
                    }
                }
                _ => return Err(format!("line {}: draw3d yaw must be ang", f.line)),
            }
            let cvt = expr_type(rgba, &env.locals, f, r)?;
            if cvt != VT::Int {
                return Err(format!("line {}: draw3d color must be int-typed (0xRRGGBBAA; alpha 0 = wireframe)", f.line));
            }
            Ok(())
        }
        Stmt::Draw3DI { verts, nv, idx, ni, x, y, z, yaw, rgba } => {
            // v12 indexed mesh pass — draw-pass only, same budget family as
            // draw3d: nv unique verts (3 cells each), ni indices (3 per tri).
            if !is_draw_fn(f) {
                return Err(format!("line {}: draw3di is a draw-pass statement (move it into draw())", f.line));
            }
            let vid = r.arr_index.get(&(GLOBAL_ENT, verts.clone())).copied()
                .ok_or_else(|| format!("line {}: draw3di vertex buffer '{}' is not a global array", f.line, verts))?;
            let iid = r.arr_index.get(&(GLOBAL_ENT, idx.clone())).copied()
                .ok_or_else(|| format!("line {}: draw3di index buffer '{}' is not a global array", f.line, idx))?;
            if iid == vid {
                return Err(format!("line {}: draw3di vertex and index buffers must be different arrays", f.line));
            }
            let nvt = expr_type(nv, &env.locals, f, r)?;
            if nvt != VT::Int {
                return Err(format!("line {}: draw3di vertex count must be int, got {:?}", f.line, nvt));
            }
            let nit = expr_type(ni, &env.locals, f, r)?;
            if nit != VT::Int {
                return Err(format!("line {}: draw3di index count must be int, got {:?}", f.line, nit));
            }
            if let Expr::IntLit(v) = nv {
                if v > &768 {
                    return Err(format!("line {}: draw3di vertex count {} exceeds the per-call budget 768", f.line, v));
                }
                if (*v as u64) * 3 > r.arrays[vid].cap as u64 {
                    return Err(format!("line {}: draw3di needs {} x 3 vertex cells but array '{}' holds only {}", f.line, v, verts, r.arrays[vid].cap));
                }
            }
            if let Expr::IntLit(v) = ni {
                if v % 3 != 0 {
                    return Err(format!("line {}: draw3di index count {} must be a multiple of 3 (one tri = 3 indices)", f.line, v));
                }
                if v > &768 {
                    return Err(format!("line {}: draw3di index count {} exceeds the per-call budget 768 (256 triangles)", f.line, v));
                }
                if (*v as u64) > r.arrays[iid].cap as u64 {
                    return Err(format!("line {}: draw3di needs {} index cells but array '{}' holds only {}", f.line, v, idx, r.arrays[iid].cap));
                }
            }
            check_num_arg(x, env, f, r, "draw3di x")?;
            check_num_arg(y, env, f, r, "draw3di y")?;
            check_num_arg(z, env, f, r, "draw3di z")?;
            let rvt = expr_type(yaw, &env.locals, f, r)?;
            match rvt {
                VT::Ang => {}
                VT::Int => {
                    if !matches!(yaw, Expr::IntLit(_)) {
                        return Err(format!("line {}: draw3di yaw must be ang-typed (or an int literal); use ang()", f.line));
                    }
                }
                _ => return Err(format!("line {}: draw3di yaw must be ang", f.line)),
            }
            let cvt = expr_type(rgba, &env.locals, f, r)?;
            if cvt != VT::Int {
                return Err(format!("line {}: draw3di color must be int-typed (0xRRGGBBAA; alpha 0 = wireframe)", f.line));
            }
            Ok(())
        }
        Stmt::QuatAA { q, off, ax, ay, az, ang } => {
            // state mutation: gameplay-side only (draw fns are presentation)
            if is_draw_fn(f) {
                return Err(format!("line {}: quat_aa is not allowed in draw fns (it writes rig state)", f.line));
            }
            let id = r.arr_index.get(&(GLOBAL_ENT, q.clone())).copied()
                .ok_or_else(|| format!("line {}: quat_aa target '{}' is not a global array (quats live in shared rig storage)", f.line, q))?;
            check_arr_off(off, id, 4, "quat_aa", q, env, f, r)?;
            check_num_arg(ax, env, f, r, "quat_aa ax")?;
            check_num_arg(ay, env, f, r, "quat_aa ay")?;
            check_num_arg(az, env, f, r, "quat_aa az")?;
            let avt = expr_type(ang, &env.locals, f, r)?;
            match avt {
                VT::Ang => {}
                VT::Int => {
                    if !matches!(ang, Expr::IntLit(_)) {
                        return Err(format!("line {}: quat_aa angle must be ang-typed (or an int literal); use ang()", f.line));
                    }
                }
                _ => return Err(format!("line {}: quat_aa angle must be ang", f.line)),
            }
            Ok(())
        }
        Stmt::QMul { d, doff, a, aoff, b, boff } => {
            if is_draw_fn(f) {
                return Err(format!("line {}: qmul is not allowed in draw fns (it writes rig state)", f.line));
            }
            check_quat_operands(d, doff, a, aoff, b, boff, "qmul", env, f, r)?;
            Ok(())
        }
        Stmt::M4QT { m, moff, q, qoff, tx, ty, tz } => {
            if is_draw_fn(f) {
                return Err(format!("line {}: m4qt is not allowed in draw fns (it writes rig state)", f.line));
            }
            let mid = r.arr_index.get(&(GLOBAL_ENT, m.clone())).copied()
                .ok_or_else(|| format!("line {}: m4qt target '{}' is not a global array (matrices live in shared rig storage)", f.line, m))?;
            let qid = r.arr_index.get(&(GLOBAL_ENT, q.clone())).copied()
                .ok_or_else(|| format!("line {}: m4qt quat '{}' is not a global array", f.line, q))?;
            check_arr_off(moff, mid, 16, "m4qt mat", m, env, f, r)?;
            check_arr_off(qoff, qid, 4, "m4qt quat", q, env, f, r)?;
            check_num_arg(tx, env, f, r, "m4qt tx")?;
            check_num_arg(ty, env, f, r, "m4qt ty")?;
            check_num_arg(tz, env, f, r, "m4qt tz")?;
            Ok(())
        }
        Stmt::M4Mul { d, doff, a, aoff, b, boff } => {
            if is_draw_fn(f) {
                return Err(format!("line {}: m4mul is not allowed in draw fns (it writes rig state)", f.line));
            }
            check_mat_operands(d, doff, a, aoff, b, boff, "m4mul", env, f, r)?;
            Ok(())
        }
        Stmt::SkinV { v, voff, m, moff, x, y, z } => {
            if is_draw_fn(f) {
                return Err(format!("line {}: skinv is not allowed in draw fns (it writes rig state)", f.line));
            }
            let vid = r.arr_index.get(&(GLOBAL_ENT, v.clone())).copied()
                .ok_or_else(|| format!("line {}: skinv vertex buffer '{}' is not a global array", f.line, v))?;
            let mid = r.arr_index.get(&(GLOBAL_ENT, m.clone())).copied()
                .ok_or_else(|| format!("line {}: skinv matrix '{}' is not a global array", f.line, m))?;
            check_arr_off(voff, vid, 3, "skinv verts", v, env, f, r)?;
            check_arr_off(moff, mid, 16, "skinv mat", m, env, f, r)?;
            check_num_arg(x, env, f, r, "skinv x")?;
            check_num_arg(y, env, f, r, "skinv y")?;
            check_num_arg(z, env, f, r, "skinv z")?;
            Ok(())
        }
        Stmt::MusicVol(_) => Ok(()), // range enforced by parser (literal 0..16)
        Stmt::Save(ix, v) => {
            // v7: persistence is gameplay state, not presentation.
            if is_draw_fn(f) {
                return Err(format!("line {}: save is not allowed in draw fns", f.line));
            }
            let ivt = expr_type(ix, &env.locals, f, r)?;
            if ivt != VT::Int {
                return Err(format!("line {}: save slot index must be int, got {:?}", f.line, ivt));
            }
            let vvt = expr_type(v, &env.locals, f, r)?;
            if !matches!(vvt, VT::Int | VT::Fixed) {
                return Err(format!("line {}: save value must be int or fixed, got {:?}", f.line, vvt));
            }
            Ok(())
        }
        Stmt::Goto(a) => {
            if is_draw_fn(f) {
                return Err(format!("line {}: goto is not allowed in draw fns", f.line));
            }
            r.domain_pos(r.scene_atom_ids, *a, "scene")?;
            Ok(())
        }
        Stmt::Draw(spr, x, y, rot, scale, rgba) => {
            if !r.sprites.iter().any(|s| s.atom == *spr) {
                return Err(format!("line {}: #{} is not a sprite", f.line, r.aname(*spr)));
            }
            check_num_arg(x, env, f, r, "draw x")?;
            check_num_arg(y, env, f, r, "draw y")?;
            // rot: ang expr or int literal
            let rvt = expr_type(rot, &env.locals, f, r)?;
            match rvt {
                VT::Ang => {}
                VT::Int => {
                    if !matches!(rot, Expr::IntLit(_)) {
                        return Err(format!("line {}: draw rot must be ang-typed (or an int literal); use ang()", f.line));
                    }
                }
                _ => return Err(format!("line {}: draw rot must be ang", f.line)),
            }
            check_num_arg(scale, env, f, r, "draw scale")?;
            let cvt = expr_type(rgba, &env.locals, f, r)?;
            // v10: rgba may be any int-typed expression (particles compute
            // per-particle alpha from ttl). Int literals remain the idiomatic
            // form; previously-rejected programs are the only delta, so
            // existing games keep byte-identical output.
            if cvt != VT::Int {
                return Err(format!("line {}: draw color must be int-typed (0xRRGGBBAA literal or expression)", f.line));
            }
            Ok(())
        }
        Stmt::DrawText(a, x, y, rgba) => {
            if !r.texts.iter().any(|(ta, _)| ta == a) {
                return Err(format!("line {}: #{} is not a text atom", f.line, r.aname(*a)));
            }
            check_num_arg(x, env, f, r, "draw_text x")?;
            check_num_arg(y, env, f, r, "draw_text y")?;
            let cvt = expr_type(rgba, &env.locals, f, r)?;
            if cvt != VT::Int || !matches!(rgba, Expr::IntLit(_)) {
                return Err(format!("line {}: draw_text color must be an integer literal", f.line));
            }
            Ok(())
        }
        Stmt::DrawNum(v, x, y, rgba) => {
            let vvt = expr_type(v, &env.locals, f, r)?;
            if !matches!(vvt, VT::Int | VT::Fixed) {
                return Err(format!("line {}: draw_num value must be int or fixed, got {:?}", f.line, vvt));
            }
            check_num_arg(x, env, f, r, "draw_num x")?;
            check_num_arg(y, env, f, r, "draw_num y")?;
            let cvt = expr_type(rgba, &env.locals, f, r)?;
            if cvt != VT::Int || !matches!(rgba, Expr::IntLit(_)) {
                return Err(format!("line {}: draw_num color must be an integer literal", f.line));
            }
            Ok(())
        }
        Stmt::CallStmt(name, args) => {
            let fi = *r.fn_index.get(name)
                .ok_or_else(|| format!("line {}: call to unknown fn '{}'", f.line, name))?;
            let target = &r.fns[fi];
            match target.kind {
                FnKind::Scene | FnKind::Helper => {
                    if target.params.len() != args.len() {
                        return Err(format!("line {}: fn '{}' takes {} args, got {}",
                            f.line, name, target.params.len(), args.len()));
                    }
                    for (i, (pn, pvt)) in target.params.iter().enumerate() {
                        let avt = expr_type(&args[i], &env.locals, f, r)?;
                        check_call_arg(*pvt, avt, &format!("arg '{}' of {}", pn, name), f.line)?;
                        // entity arg must be a plain variable (slot resolvable)
                        if matches!(pvt, VT::Ent(_)) && !matches!(args[i], Expr::Ident(_)) {
                            return Err(format!("line {}: entity arg '{}' of {} must be a variable", f.line, pn, name));
                        }
                    }
                    Ok(())
                }
                _ => Err(format!("line {}: cannot call fn '{}' directly (it is auto-run)", f.line, name)),
            }
        }
        Stmt::CallTable(tname, key, entvar) => {
            let tinfo = r.tables.iter().find(|t| &t.name == tname)
                .ok_or_else(|| format!("line {}: unknown table '{}'", f.line, tname))?;
            let kvt = expr_type(key, &env.locals, f, r)?;
            if kvt != VT::Int {
                return Err(format!("line {}: table key must be int, got {:?}", f.line, kvt));
            }
            // entity var must exist and match the table fns' entity type
            let need_ent = tinfo.entries.first()
                .and_then(|(_, fi)| r.fns.get(*fi as usize))
                .and_then(|tf| tf.entity);
            let v = entvar.as_str();
            match env.locals.iter().find(|(n, _)| n == v).map(|(_, vt)| *vt) {
                Some(VT::Ent(ei)) => {
                    if let Some(ne) = need_ent {
                        if ne != ei {
                            return Err(format!("line {}: table '{}' operates on a different entity type", f.line, tname));
                        }
                    }
                    Ok(())
                }
                _ => Err(format!("line {}: '{}' is not an entity variable", f.line, v)),
            }
        }
        Stmt::Return => Ok(()),
        // v8 OPTIMIZER subsystem 5: `fsm_step(#name);` — must live in an
        // entity fn (update/table/helper with an entity param) whose entity
        // DRIVES that fsm. opt::fsm rewrites it into the branchless form.
        Stmt::FsmStep(atom) => {
            if is_draw_fn(f) {
                return Err(format!("line {}: fsm_step is not allowed in draw fns", f.line));
            }
            let Some(ei) = f.entity else {
                return Err(format!("line {}: fsm_step needs an entity fn (update(e: T))", f.line));
            };
            let fi = r.fsms.iter().find(|ff| ff.atom == *atom)
                .ok_or_else(|| format!("line {}: fsm_step: unknown fsm '{}'", f.line, r.aname(*atom)))?;
            if fi.entity != ei {
                return Err(format!(
                    "line {}: fsm '{}' drives entity {}, not this fn's entity type",
                    f.line, fi.name, r.entities[fi.entity].name));
            }
            Ok(())
        }
        // optimizer-only forms must never reach the checker from source
        Stmt::ForMask(..) => Err("internal: ForMask is produced only by the physics pass".into()),
    }
}

pub(crate) fn check_num_arg(e: &Expr, env: &mut FnEnv, f: &FnInfo, r: &CtxRefs, what: &str) -> Result<(), String> {
    let vt = expr_type(e, &env.locals, f, r)?;
    match vt {
        VT::Fixed | VT::Int => Ok(()),
        _ => Err(format!("line {}: {} must be fixed or int, got {:?}", f.line, what, vt)),
    }
}

// ---- v12 ARTICULATION helpers (mat4/quat over global arrays) ----

/// One (array, offset) pair: array must be global, offset int-typed; a
/// LITERAL offset is bounds-proven against `need` trailing cells exactly
/// like array indexing and the draw3d budgets — config errors are compile
/// errors, never runtime wraps.
pub(crate) fn check_arr_off(off: &Expr, id: usize, need: u32, what: &str, name: &str, env: &mut FnEnv, f: &FnInfo, r: &CtxRefs) -> Result<(), String> {
    let ovt = expr_type(off, &env.locals, f, r)?;
    if ovt != VT::Int {
        return Err(format!("line {}: {} offset must be int, got {:?}", f.line, what, ovt));
    }
    if let Expr::IntLit(v) = off {
        if (*v as u64) + need as u64 > r.arrays[id].cap as u64 {
            return Err(format!("line {}: {} needs cells {}..{} but array '{}' holds only {}", f.line, what, v, *v as i64 + need as i64 - 1, name, r.arrays[id].cap));
        }
    }
    Ok(())
}

/// qmul/m4mul: three global (array, offset) pairs; quat ops need 4 cells,
/// mat ops 16.
pub(crate) fn check_quat_operands(d: &str, doff: &Expr, a: &str, aoff: &Expr, b: &str, boff: &Expr, what: &str, env: &mut FnEnv, f: &FnInfo, r: &CtxRefs) -> Result<(), String> {
    let did = r.arr_index.get(&(GLOBAL_ENT, d.to_string())).copied()
        .ok_or_else(|| format!("line {}: {} destination '{}' is not a global array", f.line, what, d))?;
    let aid = r.arr_index.get(&(GLOBAL_ENT, a.to_string())).copied()
        .ok_or_else(|| format!("line {}: {} operand '{}' is not a global array", f.line, what, a))?;
    let bid = r.arr_index.get(&(GLOBAL_ENT, b.to_string())).copied()
        .ok_or_else(|| format!("line {}: {} operand '{}' is not a global array", f.line, what, b))?;
    check_arr_off(doff, did, 4, &format!("{} dest", what), d, env, f, r)?;
    check_arr_off(aoff, aid, 4, &format!("{} a", what), a, env, f, r)?;
    check_arr_off(boff, bid, 4, &format!("{} b", what), b, env, f, r)?;
    Ok(())
}

pub(crate) fn check_mat_operands(d: &str, doff: &Expr, a: &str, aoff: &Expr, b: &str, boff: &Expr, what: &str, env: &mut FnEnv, f: &FnInfo, r: &CtxRefs) -> Result<(), String> {
    let did = r.arr_index.get(&(GLOBAL_ENT, d.to_string())).copied()
        .ok_or_else(|| format!("line {}: {} destination '{}' is not a global array", f.line, what, d))?;
    let aid = r.arr_index.get(&(GLOBAL_ENT, a.to_string())).copied()
        .ok_or_else(|| format!("line {}: {} operand '{}' is not a global array", f.line, what, a))?;
    let bid = r.arr_index.get(&(GLOBAL_ENT, b.to_string())).copied()
        .ok_or_else(|| format!("line {}: {} operand '{}' is not a global array", f.line, what, b))?;
    check_arr_off(doff, did, 16, &format!("{} dest", what), d, env, f, r)?;
    check_arr_off(aoff, aid, 16, &format!("{} a", what), a, env, f, r)?;
    check_arr_off(boff, bid, 16, &format!("{} b", what), b, env, f, r)?;
    Ok(())
}

pub(crate) fn check_call_arg(pvt: VT, avt: VT, what: &str, line: u32) -> Result<(), String> {
    match (pvt, avt) {
        (VT::Fixed, VT::Fixed) | (VT::Fixed, VT::Int) => Ok(()), // int widens
        (VT::Int, VT::Int) => Ok(()),
        (VT::Bool, VT::Bool) => Ok(()),
        (VT::Ang, VT::Ang) => Ok(()),
        (VT::Ang, VT::Int) => Ok(()), // int literal-ish widening for ang
        (VT::Ent(a), VT::Ent(b)) if a == b => Ok(()),
        _ => Err(format!("line {}: {}: expected {:?}, got {:?}", line, what, pvt, avt)),
    }
}

pub(crate) fn check_assign_compat(tvt: VT, vvt: VT, op: AssignOp, what: &str, line: u32) -> Result<(), String> {
    match op {
        AssignOp::Set => match (tvt, vvt) {
            (VT::Fixed, VT::Fixed) | (VT::Fixed, VT::Int) => Ok(()),
            (VT::Int, VT::Int) => Ok(()),
            (VT::Bool, VT::Bool) => Ok(()),
            (VT::Ang, VT::Ang) | (VT::Ang, VT::Int) => Ok(()),
            (t, v) => Err(format!("line {}: {}: cannot assign {:?} to {:?}", line, what, v, t)),
        },
        AssignOp::Add | AssignOp::Sub => match (tvt, vvt) {
            (VT::Fixed, VT::Fixed) | (VT::Fixed, VT::Int) => Ok(()),
            (VT::Int, VT::Int) => Ok(()),
            (VT::Ang, VT::Int) => Ok(()),
            (t, v) => Err(format!("line {}: {}: cannot apply {} with {:?} to {:?}", line, what,
                if op == AssignOp::Add { "+=" } else { "-=" }, v, t)),
        },
        AssignOp::Mul => match (tvt, vvt) {
            (VT::Fixed, VT::Fixed) | (VT::Fixed, VT::Int) | (VT::Int, VT::Int) => Ok(()),
            (t, v) => Err(format!("line {}: {}: cannot apply *= with {:?} to {:?}", line, what, v, t)),
        },
    }
}
