//! ============================================================================
//! LILA v10 LANGUAGE SYNTHESIS — prefabs + particles, resolved BEFORE check.
//! ============================================================================
//!
//! Two engine-grade authoring features that UE/Unity ship as RUNTIME
//! subsystems, lowered here into plain AST so the checker, codegen, VM and
//! file format never know they existed:
//!
//! 1. PREFAB INHERITANCE (Unity prefab analog)
//!       entity FastDrone from Drone { speed: 2.0  regen: u3 = 1 }
//!    merges the parent's fields (recursively, cycles rejected) under the
//!    child's overrides — same name must keep the parent's type, new names
//!    append. The child's `fsm:` wins if declared, otherwise inherited.
//!    After expansion every entity is flat: inheritance costs ZERO bytes
//!    and ZERO runtime work — it is authoring sugar, compiled away.
//!
//! 2. DECLARATIVE PARTICLES (Niagara/VFX-Graph analog)
//!       particles #boomfx (192) { sprite: #pspark  life: 22  speed: 1.8
//!                                 jitter: 0.5  drag: 0.92  grav: 0.0
//!                                 burst: 12  color: 0xFFB040  fade: true }
//!       emit(#boomfx, ex, ey);          // <- anywhere an update can run
//!    lowers to an ordinary entity type (`boomfx`: x/y/vx/vy/ttl), an
//!    `emit_boomfx` helper (burst spawns with deterministic LCG angle +
//!    speed jitter), an `update(boomfx)` (drag / gravity / ttl / kill) and
//!    a `draw(boomfx)` (sprite with ttl-driven scale + alpha fade).
//!    Everything downstream treats them as hand-written code: SoA packing,
//!    spawn budgets, compiler-emitted integration (particles get the
//!    auto-integration pass for free), draw batching. There is no particle
//!    ENGINE in the runtime — the particle system IS the bytecode.

use crate::ast::*;

/// Entry point: resolve prefab chains, then lower every particles decl.
/// Returns (kind, detail) rows for the build report. After this, no `from`
/// and no `particles` remain in the program.
pub fn expand(prog: &mut Program, atom_names: &[String]) -> Result<Vec<(String, String)>, String> {
    let mut rows = Vec::new();
    rows.extend(resolve_prefabs(prog)?);
    rows.extend(lower_particles(prog, atom_names)?);
    Ok(rows)
}

// ============================================================================
// 1. PREFAB INHERITANCE
// ============================================================================

fn resolve_prefabs(prog: &mut Program) -> Result<Vec<(String, String)>, String> {
    let n = prog.entities.len();
    let mut resolved: Vec<Option<(Vec<FieldDecl>, Vec<ArrDecl>, Option<String>)>> = vec![None; n];
    let mut visiting: Vec<usize> = Vec::new();

    for i in 0..n {
        resolve_one(prog, i, &mut resolved, &mut visiting)?;
    }
    // write the merged forms back (from: cleared — nothing downstream sees it)
    let mut rows = Vec::new();
    for (i, r) in resolved.into_iter().enumerate() {
        if let Some((fields, arrs, fsm)) = r {
            if prog.entities[i].from.is_none() {
                continue; // not a prefab (no parent): nothing to report
            }
            let parent = prog.entities[i].from.clone().unwrap_or_default();
            prog.entities[i].fields = fields;
            prog.entities[i].arrs = arrs;
            prog.entities[i].fsm = fsm;
            prog.entities[i].from = None;
            rows.push((
                format!("prefab {}", prog.entities[i].name),
                format!(
                    "{} fields merged from '{}' — inheritance compiled away, 0 runtime bytes",
                    prog.entities[i].fields.len(),
                    parent
                ),
            ));
        }
    }
    Ok(rows)
}

fn resolve_one(
    prog: &mut Program,
    i: usize,
    memo: &mut Vec<Option<(Vec<FieldDecl>, Vec<ArrDecl>, Option<String>)>>,
    visiting: &mut Vec<usize>,
) -> Result<(Vec<FieldDecl>, Vec<ArrDecl>, Option<String>), String> {
    if let Some(r) = &memo[i] {
        return Ok(r.clone());
    }
    let parent_name = prog.entities[i].from.clone();
    let (fields, arrs, fsm) = match parent_name {
        None => (
            prog.entities[i].fields.clone(),
            prog.entities[i].arrs.clone(),
            prog.entities[i].fsm.clone(),
        ),
        Some(pname) => {
            if visiting.contains(&i) {
                return Err(format!(
                    "prefab cycle: entity '{}' inherits through itself",
                    prog.entities[i].name
                ));
            }
            let pi = prog.entities.iter().position(|e| e.name == pname).ok_or_else(|| {
                format!(
                    "entity '{}': unknown prefab parent '{}' (declared later is fine, missing is not)",
                    prog.entities[i].name, pname
                )
            })?;
            visiting.push(i);
            let (mut merged, mut merged_arrs, pfsm) = resolve_one(prog, pi, memo, visiting)?;
            visiting.pop();
            // child fields: same-name = default override (type must match),
            // new names append in child order.
            for f in prog.entities[i].fields.clone() {
                if let Some(p) = merged.iter_mut().find(|p| p.name == f.name) {
                    if std::mem::discriminant(&p.ty) != std::mem::discriminant(&f.ty)
                        || p.ty != f.ty
                    {
                        return Err(format!(
                            "entity '{}': field '{}' must keep parent type {:?} (got {:?})",
                            prog.entities[i].name, f.name, p.ty, f.ty
                        ));
                    }
                    *p = f;
                } else {
                    merged.push(f);
                }
            }
            // v11 ARRAYS: prefab inheritance merges arrays the same way —
            // same (name, elem, cap) = redeclaration is fine, anything else
            // that clashes is a build error (arrays have no defaults to
            // override: capacity IS the type).
            for a in prog.entities[i].arrs.clone() {
                if let Some(p) = merged_arrs.iter_mut().find(|p| p.name == a.name) {
                    if p.elem != a.elem || p.cap != a.cap {
                        return Err(format!(
                            "entity '{}': array '{}' must keep parent type/capacity {:?}[{}] (got {:?}[{}])",
                            prog.entities[i].name, a.name, p.elem, p.cap, a.elem, a.cap
                        ));
                    }
                } else {
                    merged_arrs.push(a);
                }
            }
            (merged, merged_arrs, prog.entities[i].fsm.clone().or(pfsm))
        }
    };
    memo[i] = Some((fields.clone(), arrs.clone(), fsm.clone()));
    Ok((fields, arrs, fsm))
}

// ============================================================================
// 2. DECLARATIVE PARTICLES
// ============================================================================

fn width_for(v: u32) -> u8 {
    let mut w: u8 = 4;
    while w < 16 && (v as u64) >= (1u64 << w) {
        w += 1;
    }
    w
}

fn lower_particles(prog: &mut Program, atom_names: &[String]) -> Result<Vec<(String, String)>, String> {
    let mut rows: Vec<(String, String)> = Vec::new();
    if prog.particles.is_empty() {
        return Ok(rows);
    }
    let mut decls: Vec<(ParticleDecl, String)> = Vec::new();
    for pd in &prog.particles {
        let name = atom_names
            .get(pd.atom as usize)
            .cloned()
            .ok_or_else(|| format!("particles: unknown atom #{}", pd.atom))?;
        if prog.entities.iter().any(|e| e.name == name) {
            return Err(format!(
                "particles #{}: entity '{}' already exists — rename the particle atom",
                name, name
            ));
        }
        decls.push((pd.clone(), name));
    }

    for (pd, name) in decls {
        let f = |s: &str| format!("{}_{}", name, s); // field namer: boomfx_x ...
        let fx = f("x");
        let fy = f("y");
        let fvx = f("vx");
        let fvy = f("vy");
        let ft = f("t");

        // ---- entity: x/y/vx/vy fixed + ttl countdown ----
        prog.entities.push(EntityDecl {
            name: name.clone(),
            fields: vec![
                FieldDecl { name: fx.clone(), ty: Ty::Fixed, default_int: None, default_fix: Some(0), default_bool: None, cold: false },
                FieldDecl { name: fy.clone(), ty: Ty::Fixed, default_int: None, default_fix: Some(0), default_bool: None, cold: false },
                FieldDecl { name: fvx.clone(), ty: Ty::Fixed, default_int: None, default_fix: Some(0), default_bool: None, cold: false },
                FieldDecl { name: fvy.clone(), ty: Ty::Fixed, default_int: None, default_fix: Some(0), default_bool: None, cold: false },
                FieldDecl {
                    name: ft.clone(),
                    ty: Ty::UInt(width_for(pd.life)),
                    default_int: Some(pd.life as i64),
                    default_fix: None,
                    default_bool: None,
                    cold: false,
                },
            ],
            fsm: None,
            from: None,
            max_live_override: Some(pd.cap),
            arrs: Vec::new(),
        });

        // ---- emit_<name>(ex, ey): burst spawns, deterministic LCG angle ----
        // let i = 0; while (i < burst) {
        //     let a  = rand(4096);
        //     let sp = speed * (256 - rand(jitter));   // Q8: exact speed at jitter 0
        //     spawn <name> { x: ex, y: ey, vx: sin(ang(a))*sp, vy: cos(ang(a))*sp, t: life };
        //     i += 1;
        // }
        let mut loop_body = vec![
            Stmt::Let("a".into(), Expr::Call("rand".into(), vec![Expr::IntLit(4096)])),
            Stmt::Let(
                "sp".into(),
                Expr::Binary(
                    BinOp::Mul,
                    Box::new(Expr::FixLit(pd.speed_fix)),
                    Box::new(Expr::Binary(
                        BinOp::Sub,
                        Box::new(Expr::IntLit(256)),
                        Box::new(Expr::Call("rand".into(), vec![Expr::IntLit(pd.jitter_q8)])),
                    )),
                ),
            ),
            Stmt::Spawn(
                name.clone(),
                vec![
                    (fx.clone(), Expr::Ident("ex".into())),
                    (fy.clone(), Expr::Ident("ey".into())),
                    (
                        fvx.clone(),
                        Expr::Binary(
                            BinOp::Mul,
                            Box::new(Expr::Call(
                                "sin".into(),
                                vec![Expr::Call("ang".into(), vec![Expr::Ident("a".into())])],
                            )),
                            Box::new(Expr::Ident("sp".into())),
                        ),
                    ),
                    (
                        fvy.clone(),
                        Expr::Binary(
                            BinOp::Mul,
                            Box::new(Expr::Call(
                                "cos".into(),
                                vec![Expr::Call("ang".into(), vec![Expr::Ident("a".into())])],
                            )),
                            Box::new(Expr::Ident("sp".into())),
                        ),
                    ),
                    (ft.clone(), Expr::IntLit(pd.life)),
                ],
            ),
            Stmt::Assign(Expr::Ident("i".into()), AssignOp::Add, Expr::IntLit(1)),
        ];
        // jitter: 0 lowers to no rand call at all (constant speed, less code)
        if pd.jitter_q8 == 0 {
            loop_body[1] = Stmt::Let("sp".into(), Expr::FixLit(pd.speed_fix));
        }
        prog.fns.push(FnDecl {
            name: format!("emit_{}", name),
            params: vec![("ex".into(), Ty::Fixed), ("ey".into(), Ty::Fixed)],
            body: vec![
                Stmt::Let("i".into(), Expr::IntLit(0)),
                Stmt::While(
                    Expr::Binary(BinOp::Lt, Box::new(Expr::Ident("i".into())), Box::new(Expr::IntLit(pd.burst))),
                    loop_body,
                ),
                Stmt::Return,
            ],
            line: 0,
        });

        // ---- update(e: name): drag -> gravity -> ttl -> kill ----
        let ef = |s: &str| Expr::Field(Box::new(Expr::Ident("e".into())), s.to_string());
        let mut body: Vec<Stmt> = Vec::new();
        if pd.drag_q8 != 256 {
            // vx *= drag (Q8 multiply — exact no-op at 256, skipped above)
            body.push(Stmt::Assign(
                ef(&fvx),
                AssignOp::Set,
                Expr::Binary(BinOp::Mul, Box::new(ef(&fvx)), Box::new(Expr::IntLit(pd.drag_q8))),
            ));
            let vy = Expr::Binary(BinOp::Mul, Box::new(ef(&fvy)), Box::new(Expr::IntLit(pd.drag_q8)));
            body.push(Stmt::Assign(
                ef(&fvy),
                AssignOp::Set,
                if pd.grav_fix != 0 {
                    Expr::Binary(BinOp::Add, Box::new(vy), Box::new(Expr::FixLit(pd.grav_fix)))
                } else {
                    vy
                },
            ));
        } else if pd.grav_fix != 0 {
            body.push(Stmt::Assign(ef(&fvy), AssignOp::Add, Expr::FixLit(pd.grav_fix)));
        }
        body.push(Stmt::Assign(ef(&ft), AssignOp::Sub, Expr::IntLit(1)));
        body.push(Stmt::If(
            Expr::Binary(BinOp::Eq, Box::new(ef(&ft)), Box::new(Expr::IntLit(0))),
            vec![Stmt::Kill(Expr::Ident("e".into()))],
            vec![],
        ));
        prog.fns.push(FnDecl {
            name: "update".into(),
            params: vec![("e".into(), Ty::Entity(name.clone()))],
            body,
            line: 0,
        });

        // ---- draw(e: name): sprite, ttl-driven scale + alpha fade ----
        // draw(#sprite, e.x, e.y, ang(0), scale, color | ((t*255)/life)<<24)
        // scale MUST be a FIXED-typed expr: (t * 1.0) / life lowers to
        // mulF(t,256)=t then divF(t,life) = t*256/life Q8 — an int-typed
        // form would get an extra I2F (x256) from the int->fixed widening.
        let scale: Expr = if pd.fade {
            Expr::Binary(
                BinOp::Div,
                Box::new(Expr::Binary(
                    BinOp::Mul,
                    Box::new(ef(&ft)),
                    Box::new(Expr::FixLit(256)),
                )),
                Box::new(Expr::IntLit(pd.life)),
            )
        } else {
            Expr::FixLit(256)
        };
        let rgba: Expr = if pd.fade {
            Expr::Binary(
                BinOp::Or,
                Box::new(Expr::IntLit(pd.color)),
                Box::new(Expr::Binary(
                    BinOp::Shl,
                    Box::new(Expr::Binary(
                        BinOp::Div,
                        Box::new(Expr::Binary(
                            BinOp::Mul,
                            Box::new(ef(&ft)),
                            Box::new(Expr::IntLit(255)),
                        )),
                        Box::new(Expr::IntLit(pd.life)),
                    )),
                    Box::new(Expr::IntLit(24)),
                )),
            )
        } else {
            Expr::IntLit(pd.color | 0xFF000000)
        };
        prog.fns.push(FnDecl {
            name: "draw".into(),
            params: vec![("e".into(), Ty::Entity(name.clone()))],
            body: vec![Stmt::Draw(
                pd.sprite,
                ef(&fx),
                ef(&fy),
                Expr::Call("ang".into(), vec![Expr::IntLit(0)]),
                scale,
                rgba,
            )],
            line: 0,
        });
        rows.push((
            format!("particles #{}", name),
            format!(
                "{} slots, burst {}, life {}f — entity + emit/update/draw synthesized (drag {:.2}, grav {:.2}); 0 runtime bytes, 0 new opcodes",
                pd.cap, pd.burst, pd.life,
                pd.drag_q8 as f32 / 256.0,
                pd.grav_fix as f32 / 256.0
            ),
        ));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(name: &str, ty: Ty, fix: i32) -> FieldDecl {
        FieldDecl { name: name.into(), ty, default_int: None, default_fix: Some(fix), default_bool: None, cold: false }
    }

    fn prog_with(entities: Vec<EntityDecl>) -> Program {
        Program { entities, ..Default::default() }
    }

    /// Child inherits every parent field; overrides swap defaults; new
    /// fields append; `from` is cleared so downstream sees flat entities.
    #[test]
    fn prefab_inherits_overrides_and_appends() {
        let mut p = prog_with(vec![
            EntityDecl {
                name: "Drone".into(),
                fields: vec![field("x", Ty::Fixed, 0), field("hp", Ty::UInt(4), 3)],
                fsm: None, from: None, max_live_override: None, arrs: Vec::new(),
            },
            EntityDecl {
                name: "FastDrone".into(),
                fields: vec![field("hp", Ty::UInt(4), 1), field("boost", Ty::Fixed, 128)],
                fsm: None,
                from: Some("Drone".into()),
                max_live_override: None,
                arrs: Vec::new(),
            },
        ]);
        let rows = expand(&mut p, &[]).unwrap();
        assert_eq!(rows.len(), 1, "one prefab row reported");
        let fd = &p.entities[1];
        assert!(fd.from.is_none(), "inheritance must be compiled away");
        assert_eq!(fd.fields.len(), 3, "x + hp + boost");
        assert_eq!(fd.fields[0].name, "x", "parent field first, parent default");
        assert_eq!(fd.fields[0].default_fix, Some(0));
        assert_eq!(fd.fields[1].name, "hp");
        assert_eq!(fd.fields[1].default_fix, Some(1), "child default overrides");
        assert_eq!(fd.fields[2].name, "boost", "new field appended");
        // parent untouched
        assert_eq!(p.entities[0].fields.len(), 2);
    }

    /// Forward references work (parent declared after the child) and unknown
    /// parents are loud build errors.
    #[test]
    fn prefab_forward_reference_and_unknown_parent() {
        let mut p = prog_with(vec![
            EntityDecl {
                name: "B".into(),
                fields: vec![field("v", Ty::Fixed, 5)],
                fsm: None,
                from: Some("A".into()),
                max_live_override: None,
                arrs: Vec::new(),
            },
            EntityDecl {
                name: "A".into(),
                fields: vec![field("x", Ty::Fixed, 0)],
                fsm: None, from: None, max_live_override: None, arrs: Vec::new(),
            },
        ]);
        expand(&mut p, &[]).unwrap();
        assert_eq!(p.entities[0].fields.len(), 2);
        assert_eq!(p.entities[0].fields[0].name, "x");

        let mut bad = prog_with(vec![EntityDecl {
            name: "C".into(), fields: vec![], fsm: None,
            from: Some("Ghost".into()), max_live_override: None, arrs: Vec::new(),
        }]);
        assert!(expand(&mut bad, &[]).is_err(), "unknown parent must fail the build");
    }

    /// Cycles are rejected (A from B from A) — not infinite-looped.
    #[test]
    fn prefab_cycle_rejected() {
        let mut p = prog_with(vec![
            EntityDecl {
                name: "A".into(), fields: vec![], fsm: None,
                from: Some("B".into()), max_live_override: None, arrs: Vec::new(),
            },
            EntityDecl {
                name: "B".into(), fields: vec![], fsm: None,
                from: Some("A".into()), max_live_override: None, arrs: Vec::new(),
            },
        ]);
        assert!(expand(&mut p, &[]).is_err());
    }

    /// A particles decl lowers to entity + emit_ helper + update + draw;
    /// field names are atom-prefixed so they cannot collide with user
    /// fields; the ttl width fits the lifetime; the capacity override rides
    /// along for the checker.
    #[test]
    fn particles_lower_to_entity_and_fns() {
        let mut p = Program::default();
        p.particles.push(ParticleDecl {
            atom: 0,
            cap: 192,
            sprite: 3,
            life: 22,
            speed_fix: 460,
            jitter_q8: 0,
            drag_q8: 235,
            grav_fix: 32,
            burst: 12,
            color: 0xFFB040,
            fade: true,
        });
        expand(&mut p, &["boomfx".to_string()]).unwrap();
        assert_eq!(p.entities.len(), 1);
        let e = &p.entities[0];
        assert_eq!(e.name, "boomfx");
        assert_eq!(e.max_live_override, Some(192));
        let names: Vec<&str> = e.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["boomfx_x", "boomfx_y", "boomfx_vx", "boomfx_vy", "boomfx_t"]);
        assert_eq!(e.fields[4].default_int, Some(22), "ttl starts at life");
        assert!(e.fields[4].ty == Ty::UInt(5), "22 fits u5");
        assert_eq!(p.fns.len(), 3, "emit + update + draw");
        assert_eq!(p.fns[0].name, "emit_boomfx");
        assert_eq!(p.fns[1].name, "update");
        assert_eq!(p.fns[2].name, "draw");
        // update body: drag on vx (Q8 const), drag+grav on vy, ttl--, kill at 0
        let up = &p.fns[1].body;
        assert_eq!(up.len(), 4, "vx drag, vy drag+grav, ttl--, if/kill");
    }

    /// Duplicate entity name vs a particles atom is a loud error.
    #[test]
    fn particles_name_collision_is_loud() {
        let mut p = prog_with(vec![EntityDecl {
            name: "boomfx".into(), fields: vec![], fsm: None, from: None, max_live_override: None, arrs: Vec::new(),
        }]);
        p.particles.push(ParticleDecl {
            atom: 0, cap: 64, sprite: 0, life: 10, speed_fix: 256, jitter_q8: 0,
            drag_q8: 256, grav_fix: 0, burst: 4, color: 0xFFFFFF, fade: false,
        });
        assert!(expand(&mut p, &["boomfx".to_string()]).is_err());
    }
}
