//! Field layout: bit-width inference and dense/cold row packing.
//! (split out of the checker monolith — the layout math is testable on its own)

use super::*;

// ---------------- field layout ----------------

pub(crate) fn bits_for_uint(v: u64) -> u8 {
    let mut w = 4u8;
    while w < 16 && v >= (1u64 << w) { w += 1; }
    w
}

/// Rewrite bare identifiers that name fields of entity `ei` into
/// `__ent.<field>` accesses (used to type-check fsm guards against a
/// synthetic parameter). Globals and builtins stay untouched.
pub(crate) fn wrap_entity_fields(e: &Expr, ei: usize, entities: &[EntityInfo]) -> Expr {
    match e {
        Expr::Ident(name) => {
            let is_field = entities[ei].fields.iter().any(|f| &f.name == name);
            if is_field {
                Expr::Field(Box::new(Expr::Ident("__ent".into())), name.clone())
            } else {
                e.clone()
            }
        }
        Expr::Unary(op, a) => Expr::Unary(*op, Box::new(wrap_entity_fields(a, ei, entities))),
        Expr::Binary(op, a, b) => Expr::Binary(
            *op,
            Box::new(wrap_entity_fields(a, ei, entities)),
            Box::new(wrap_entity_fields(b, ei, entities)),
        ),
        Expr::Call(name, args) => Expr::Call(
            name.clone(),
            args.iter().map(|a| wrap_entity_fields(a, ei, entities)).collect(),
        ),
        other => other.clone(),
    }
}

pub(crate) fn bits_for_sint(v: i64) -> u8 {
    let mut w = 4;
    loop {
        let min = -(1i64 << (w - 1));
        let max = (1i64 << (w - 1)) - 1;
        if v >= min && v <= max { break; }
        w += 1;
        if w > 16 { break; }
    }
    w.min(16)
}

pub(crate) fn build_fields(fields: &[FieldDecl], _ent: Option<&str>, what: &str, cap_ent: u32, cap_override: Option<u32>) -> Result<(EntityInfo, u32), String> {
    let mut out: Vec<FieldInfo> = Vec::new();
    let mut dense_bits: u32 = 0;
    let mut cold_bits: u32 = 0;
    for fd in fields {
        let (vt, bits, default) = match &fd.ty {
            Ty::Fixed => {
                // Int literal defaults are whole units: `x: fixed = 256` means
                // 256 pixels -> 256 << 8 in Q24.8. Fixed literals (`= 1.0`) are
                // already Q24.8.
                let d = fd.default_fix
                    .or(fd.default_int.map(|v| (v as i64 * 256) as i32))
                    .unwrap_or(0);
                (VT::Fixed, 32u8, d)
            }
            Ty::Bool => {
                let d = if fd.default_bool.unwrap_or(false) { 1 } else { 0 };
                (VT::Bool, 1u8, d)
            }
            Ty::Ang => {
                let d = fd.default_int.unwrap_or(0) as i32;
                (VT::Ang, 16u8, d)
            }
            Ty::UInt(w) => {
                let w = if *w == 0 {
                    match fd.default_int {
                        Some(v) if v >= 0 => bits_for_uint(v as u64),
                        Some(_) => return Err(format!("{} field '{}': unsigned default must be >= 0", what, fd.name)),
                        None => 8,
                    }
                } else { *w };
                let d = fd.default_int.unwrap_or(0) as i32;
                (VT::Int, w, d)
            }
            Ty::SInt(w) => {
                let w = if *w == 0 {
                    match fd.default_int {
                        Some(v) => bits_for_sint(v),
                        None => 8,
                    }
                } else { *w };
                let d = fd.default_int.unwrap_or(0) as i32;
                (VT::Int, w, d)
            }
            Ty::Entity(_) => {
                return Err(format!("{} field '{}': entity refs cannot be stored fields (handle-free design)", what, fd.name));
            }
        };
        let off_bits = if fd.cold { cold_bits as u16 } else { dense_bits as u16 };
        let c_off_bits = if fd.cold { cold_bits as u16 } else { 0 };
        if fd.cold {
            cold_bits += bits as u32;
        } else {
            dense_bits += bits as u32;
        }
        let signed = matches!(&fd.ty, Ty::SInt(_));
        out.push(FieldInfo {
            name: fd.name.clone(), vt, signed, bits, cold: fd.cold,
            off_bits, c_off_bits, default,
        });
    }
    let row_bytes = ((dense_bits + 7) / 8) as u16;
    let cold_bytes = ((cold_bits + 7) / 8) as u16;
    // per-type live cap: particles pools override; else game-declared
    // capacity (v2) or the engine default. The runtime enforces this at
    // spawn time AND validates it at load.
    let max_live = if what == "global" { 0 } else { cap_override.unwrap_or(cap_ent) };
    Ok((EntityInfo {
        name: what.to_string(),
        fields: out, row_bytes, cold_bytes, max_live,
    }, dense_bits))
}

