//! LILA codegen: typed AST -> instruction stream.
//! Types are computed via the checker's typing rules (expr_type) so codegen
//! and the checker can never disagree.

use crate::ast::*;
use crate::checker::{Ctx, CtxRefs, FnInfo, FnKind, VT, GLOBAL_ENT};
use std::collections::HashMap;

// ---- opcodes (must match the Zig VM) ----
pub const OP_PUSH_S8: u8 = 0x01;
pub const OP_PUSH_POOL: u8 = 0x02;
pub const OP_LD_LOCAL: u8 = 0x03;
pub const OP_ST_LOCAL: u8 = 0x04;
pub const OP_LD_GLBL: u8 = 0x05;
pub const OP_ST_GLBL: u8 = 0x06;
pub const OP_LD_ENT: u8 = 0x07;
pub const OP_ST_ENT: u8 = 0x08;
pub const OP_ADD_F: u8 = 0x09;
pub const OP_SUB_F: u8 = 0x0A;
pub const OP_MUL_F: u8 = 0x0B;
pub const OP_DIV_F: u8 = 0x0C;
pub const OP_ADD_I: u8 = 0x0D;
pub const OP_SUB_I: u8 = 0x0E;
pub const OP_MUL_I: u8 = 0x0F;
pub const OP_DIV_I: u8 = 0x10;
pub const OP_MOD_I: u8 = 0x11;
pub const OP_AND_I: u8 = 0x12;
pub const OP_OR_I: u8 = 0x13;
pub const OP_XOR_I: u8 = 0x14;
pub const OP_SHL: u8 = 0x15;
pub const OP_SHR: u8 = 0x16;
pub const OP_AND_B: u8 = 0x17;
pub const OP_OR_B: u8 = 0x18;
pub const OP_NOT_B: u8 = 0x19;
pub const OP_NEG: u8 = 0x1A;
pub const OP_EQ_F: u8 = 0x1B;
pub const OP_NE_F: u8 = 0x1C;
pub const OP_LT_F: u8 = 0x1D;
pub const OP_GT_F: u8 = 0x1E;
pub const OP_LE_F: u8 = 0x1F;
pub const OP_GE_F: u8 = 0x20;
pub const OP_EQ_I: u8 = 0x21;
pub const OP_NE_I: u8 = 0x22;
pub const OP_LT_I: u8 = 0x23;
pub const OP_GT_I: u8 = 0x24;
pub const OP_LE_I: u8 = 0x25;
pub const OP_GE_I: u8 = 0x26;
pub const OP_SIN: u8 = 0x27;
pub const OP_COS: u8 = 0x28;
pub const OP_RAND_MAX: u8 = 0x29;
pub const OP_ANIM: u8 = 0x2A;
pub const OP_KEY: u8 = 0x2B;
pub const OP_JMP: u8 = 0x2C;
pub const OP_JZ: u8 = 0x2D;
pub const OP_JNZ: u8 = 0x2E;
pub const OP_FOR_BGN: u8 = 0x2F;
pub const OP_FOR_ADV: u8 = 0x30;
pub const OP_SPAWN: u8 = 0x31;
pub const OP_KILL: u8 = 0x32;
pub const OP_SFX: u8 = 0x33;
pub const OP_MUSIC: u8 = 0x34;
pub const OP_STOP_MUSIC: u8 = 0x35;
pub const OP_GOTO: u8 = 0x36;
pub const OP_DRAW: u8 = 0x37;
pub const OP_DRAW_TEXT: u8 = 0x38;
pub const OP_DRAW_NUM: u8 = 0x39;
pub const OP_CALL_TBL: u8 = 0x3A;
pub const OP_COUNT: u8 = 0x3B;
pub const OP_RET: u8 = 0x3C;
pub const OP_CALL_FN: u8 = 0x3D;
pub const OP_I2F: u8 = 0x3E; // int -> fixed (x << 8)
pub const OP_F2I: u8 = 0x3F; // fixed -> int (x >> 8 arithmetic)
pub const OP_SHAKE: u8 = 0x40; // screen-shake: u8 amplitude (px)
// ---- v7: camera / persistence / aim math / music volume ----
pub const OP_CAMERA: u8 = 0x41;    // pops y, x (fixed) -> clamped camera center
pub const OP_MUSIC_VOL: u8 = 0x42; // u8 master music volume 0..16
pub const OP_SAVE: u8 = 0x43;      // pops v, ix -> SRAM slot (ix & 63)
pub const OP_SAVED: u8 = 0x44;     // pops ix -> pushes slot value (raw i32)
pub const OP_DIST: u8 = 0x45;     // pops y2 x2 y1 x1 -> pushes Q24.8 distance
pub const OP_ATAN2: u8 = 0x46;    // pops dx, dy -> pushes screen-convention ang
pub const OP_CAM_X: u8 = 0x47;    // pushes camera center x (fixed)
pub const OP_CAM_Y: u8 = 0x48;    // pushes camera center y (fixed)
// ---- v8 OPTIMIZER opcodes: physics / AI bit-planes ----
// (all branchless, all decided at compile time by the opt:: passes)
pub const OP_SWEPT: u8 = 0x49;         // pops hh hw by bx avy avx ay ax -> bool swept Minkowski interval
pub const OP_COLLIDE_MASK: u8 = 0x4A;  // pops cy cx -> u32 live-slot mask (Galois 32 lanes); a=type, b/c=pool hw/hh, extra=[x_tf,y_tf]
pub const OP_FOR_MASK_BGN: u8 = 0x4B;  // pops mask -> iterate set slots (a=local slot, b=type, c=exit rel)
pub const OP_FOR_MASK_ADV: u8 = 0x4C;  // advance ctz scan (a=body rel)
pub const OP_FSM_NEXT: u8 = 0x4D;      // pops s, em -> next = T[s*span + em] (bit-plane transition)
pub const OP_SEL: u8 = 0x4E;           // pops else, then, cond -> branchless conditional move

// ---- v11 ARRAYS + 3D ----
pub const OP_LD_GARR: u8 = 0x4F;       // a=array id; pops idx -> pushes elem (global arrays)
pub const OP_ST_GARR: u8 = 0x50;       // a=array id; pops val, idx -> store (bounds: mask|mod)
pub const OP_LD_EARR: u8 = 0x51;       // a=local slot, b=array id; pops idx -> pushes elem
pub const OP_ST_EARR: u8 = 0x52;       // a=local slot, b=array id; pops val, idx -> store
pub const OP_DUP: u8 = 0x53;           // duplicate top of stack (compound array assigns)
pub const OP_CAM3: u8 = 0x54;          // pops pitch, yaw, z, y, x -> 3D camera registers
pub const OP_PROJ3: u8 = 0x55;         // pops z, y, x -> scale; stashes screen x/y + ok flag
pub const OP_PROJ_X: u8 = 0x56;        // pushes stashed screen x (fixed)
pub const OP_PROJ_Y: u8 = 0x57;        // pushes stashed screen y (fixed)
pub const OP_PROJ_OK: u8 = 0x58;       // pushes stashed ok flag (bool: in front of camera)
pub const OP_DRAW3D: u8 = 0x59;        // a=array id; pops rgba, yaw, z, y, x, n -> mesh pass

// ---- v12 ARTICULATION: mat4/quat builtins over arrays + indexed meshes ----
// (all stack-arg ops: args pushed in source order, VM pops reversed)
pub const OP_QUAT_AA: u8 = 0x5A;       // pops ang, az, ay, ax, off, q_aid -> quat = axis-angle
pub const OP_Q_MUL: u8 = 0x5B;         // pops boff,b_aid,aoff,a_aid,doff,d_aid -> d = a*b (Hamilton)
pub const OP_M4_QT: u8 = 0x5C;         // pops tz,ty,tx,q_off,q_aid,m_off,m_aid -> M = T(t)*R(q)
pub const OP_M4_MUL: u8 = 0x5D;        // pops boff,b_aid,aoff,a_aid,doff,d_aid -> d = a*b (alias-safe)
pub const OP_SKIN3: u8 = 0x5E;         // pops z,y,x,m_off,m_aid,v_off,v_aid -> verts[voff..+2] = M*p
pub const OP_DRAW3DI: u8 = 0x5F;       // a=vert arr, b=idx arr; pops rgba, yaw, z, y, x, ni, nv -> indexed mesh pass

// operand signatures: (kind_a, kind_b, kind_c)
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum OpK { None, U8, U16, S16, S8, PoolU16 }

pub fn sig(op: u8) -> (OpK, OpK, OpK) {
    use OpK::*;
    match op {
        OP_PUSH_S8 => (S8, None, None),
        OP_PUSH_POOL => (PoolU16, None, None),
        OP_LD_LOCAL | OP_ST_LOCAL | OP_COUNT => (U8, None, None),
        OP_KILL => (U8, U8, None),
        OP_LD_GLBL | OP_ST_GLBL => (U16, None, None),
        OP_LD_ENT | OP_ST_ENT => (U8, U16, None),
        OP_JMP | OP_JZ | OP_JNZ | OP_FOR_ADV => (S16, None, None),
        OP_FOR_BGN => (U8, U8, S16),
        OP_CALL_FN => (U16, U8, None),
        OP_CALL_TBL => (U8, U8, None),
        OP_SFX | OP_MUSIC | OP_GOTO | OP_DRAW | OP_DRAW_TEXT | OP_ANIM | OP_KEY => (U16, None, None),
        OP_SHAKE => (U8, None, None),
        OP_MUSIC_VOL => (U8, None, None),
        OP_CAMERA | OP_SAVE | OP_SAVED | OP_DIST | OP_ATAN2 | OP_CAM_X | OP_CAM_Y => (None, None, None),
        OP_SPAWN => (U8, U8, None), // + extra U16 tf list (count in b)
        // v8: swept interval solve — 8 fixed args on the stack
        OP_SWEPT => (None, None, None),
        // v8: Galois collision mask — a=type, b/c = pool indices of hw/hh,
        // extra = [x_tf, y_tf] (resolved at compile time)
        OP_COLLIDE_MASK => (U8, PoolU16, PoolU16),
        OP_FOR_MASK_BGN => (U8, U8, S16),
        OP_FOR_MASK_ADV => (S16, None, None),
        OP_FSM_NEXT => (U16, None, None),
        OP_SEL => (None, None, None),
        // v11: arrays + 3D
        OP_LD_GARR | OP_ST_GARR | OP_DRAW3D => (U8, None, None),
        OP_LD_EARR | OP_ST_EARR => (U8, U8, None),
        OP_DUP | OP_CAM3 | OP_PROJ3 | OP_PROJ_X | OP_PROJ_Y | OP_PROJ_OK => (None, None, None),
        // v12: articulation — mat/quat ops are pure stack ops; the indexed
        // mesh pass carries both array ids as operands
        OP_QUAT_AA | OP_Q_MUL | OP_M4_QT | OP_M4_MUL | OP_SKIN3 => (None, None, None),
        OP_DRAW3DI => (U8, U8, None),
        _ => (None, None, None),
    }
}

#[derive(Debug, Clone)]
pub struct Instr {
    pub op: u8,
    pub a: i32,
    pub b: i32,
    pub c: i32,
    pub extra: Vec<u16>, // SPAWN tf list
}

impl Instr {
    fn new(op: u8) -> Instr { Instr { op, a: 0, b: 0, c: 0, extra: Vec::new() } }
    fn a(op: u8, a: i32) -> Instr { Instr { op, a, b: 0, c: 0, extra: Vec::new() } }
    fn ab(op: u8, a: i32, b: i32) -> Instr { Instr { op, a, b, c: 0, extra: Vec::new() } }
    fn abc(op: u8, a: i32, b: i32, c: i32) -> Instr { Instr { op, a, b, c, extra: Vec::new() } }
}

pub struct FnCode {
    pub kind: FnKind,
    pub entity: u8,      // 0xFF none
    pub nparams: u8,
    pub nlocals: u8,     // total slots incl params
    pub start: u32,      // instr index into global stream
    pub len: u32,
}

pub struct Code {
    pub instrs: Vec<Instr>,
    pub pool: Vec<i32>,
    pub fns: Vec<FnCode>,
}

struct Gen<'a> {
    ctx: &'a Ctx,
    cur_max_locals: usize,
    tf: HashMap<(u8, String), usize>,
    gidx: HashMap<String, usize>,
    eidx: HashMap<String, usize>,
    fidx: HashMap<String, usize>,
    arrs: HashMap<(u8, String), usize>,
    instrs: Vec<Instr>,
    pool: Vec<i32>,
    pool_map: HashMap<i32, u16>,
    /// mask local name -> 32-lane window base (v9 multi-window Galois)
    mask_base: HashMap<String, i32>,
}

struct Env {
    locals: Vec<(String, VT)>,
}

pub fn gen(ctx: &Ctx) -> Result<Code, String> {
    let mut tf = HashMap::new();
    for (i, t) in ctx.typed_fields.iter().enumerate() {
        let fname = if t.ent == GLOBAL_ENT {
            ctx.globals[t.field as usize].name.clone()
        } else {
            ctx.entities[t.ent as usize].fields[t.field as usize].name.clone()
        };
        tf.insert((t.ent, fname), i);
    }
    let gidx: HashMap<String, usize> = ctx.globals.iter().enumerate()
        .map(|(i, g)| (g.name.clone(), i)).collect();
    let eidx: HashMap<String, usize> = ctx.entities.iter().enumerate()
        .map(|(i, e)| (e.name.clone(), i)).collect();
    let fidx: HashMap<String, usize> = ctx.fns.iter().enumerate()
        .map(|(i, f)| (f.name.clone(), i)).collect();
    let arrs: HashMap<(u8, String), usize> = ctx.arrays.iter().enumerate()
        .map(|(i, a)| ((a.ent, a.name.clone()), i)).collect();

    let mut g = Gen { ctx, tf, gidx, eidx, fidx, arrs, instrs: Vec::new(), pool: Vec::new(), pool_map: HashMap::new(), cur_max_locals: 0, mask_base: HashMap::new() };

    // refs for typing (reuse checker typing rules)
    let refs = ctx.refs();

    let mut fns_meta = Vec::new();
    for f in &ctx.fns {
        let start = g.instrs.len() as u32;
        g.cur_max_locals = f.params.len();
        let mut env = Env { locals: f.params.clone() };
        g.gen_block(&f.body, &mut env, f, &refs)?;
        g.instrs.push(Instr::new(OP_RET));
        // v12 LOUD LIMIT: the VM frame holds MAX_LOCALS slots. Codegen used
        // to silently drop stores past the limit (reads returned 0) — a
        // miscompile that hangs loops. Reject at build time instead.
        if g.cur_max_locals > 32 {
            return Err(format!(
                "fn '{}' needs {} local slots but the engine frame holds 32 (split the fn or shrink its scope)",
                f.name, g.cur_max_locals
            ));
        }
        fns_meta.push(FnCode {
            kind: f.kind,
            entity: f.entity.map(|e| e as u8).unwrap_or(0xFF),
            nparams: f.params.len() as u8,
            nlocals: g.cur_max_locals as u8,
            start,
            len: (g.instrs.len() as u32) - start,
        });
    }

    Ok(Code { instrs: g.instrs, pool: g.pool, fns: fns_meta })
}



impl<'a> Gen<'a> {
    fn pool_idx(&mut self, v: i32) -> u16 {
        if let Some(&i) = self.pool_map.get(&v) { return i; }
        let i = self.pool.len() as u16;
        self.pool.push(v);
        self.pool_map.insert(v, i);
        i
    }

    fn push_const(&mut self, v: i32) {
        if (-128..=127).contains(&v) {
            self.instrs.push(Instr::a(OP_PUSH_S8, v));
        } else {
            let pi = self.pool_idx(v);
            self.instrs.push(Instr::a(OP_PUSH_POOL, pi as i32));
        }
    }

    fn tf_of(&self, ent: u8, name: &str) -> Result<usize, String> {
        self.tf.get(&(ent, name.to_string())).copied()
            .ok_or_else(|| crate::diag::internal(&format!("no typed field {}.{}", ent, name)))
    }

    fn dom_pos(&self, dom: &[u16], atom: u16, what: &str) -> Result<u16, String> {
        dom.iter().position(|&x| x == atom)
            .map(|p| p as u16)
            .ok_or_else(|| crate::diag::internal(&format!("atom {} not in {} domain", atom, what)))
    }

    fn gen_block(&mut self, stmts: &[Stmt], env: &mut Env, f: &FnInfo, refs: &CtxRefs) -> Result<(), String> {
        for s in stmts { self.gen_stmt(s, env, f, refs)?; }
        Ok(())
    }

    fn gen_stmt(&mut self, s: &Stmt, env: &mut Env, f: &FnInfo, refs: &CtxRefs) -> Result<(), String> {
        match s {
            Stmt::Let(name, e) => {
                let vt = self.gen_expr(e, env, f, refs)?;
                // remember the window base of a Galois mask let so the
                // matching ForMask can carry it into the loop opcode
                if let Expr::Intrin(crate::ast::INTR_COLL_MASK, _, _, consts) = e {
                    self.mask_base.insert(name.clone(), consts.get(2).copied().unwrap_or(0));
                }
                let slot = match env.locals.iter().position(|(n, _)| n == name) {
                    Some(s) => s,
                    None => env.locals.len(),
                };
                self.instrs.push(Instr::a(OP_ST_LOCAL, slot as i32));
                env_set(env, name, vt);
                if env.locals.len() > self.cur_max_locals {
                    self.cur_max_locals = env.locals.len();
                }
                Ok(())
            }
            Stmt::Assign(target, op, val) => {
                self.gen_assign(target, *op, val, env, f, refs)
            }
            Stmt::If(cond, then_b, else_b) => {
                let cvt = self.gen_expr(cond, env, f, refs)?;
                debug_assert_eq!(cvt, VT::Bool);
                let jz = self.instrs.len();
                self.instrs.push(Instr::a(OP_JZ, 0)); // patch target
                self.gen_block(then_b, env, f, refs)?;
                let jend: Option<usize> = if !else_b.is_empty() {
                    let j = self.instrs.len();
                    self.instrs.push(Instr::a(OP_JMP, 0));
                    Some(j)
                } else { None };
                // else starts here
                let else_start = self.instrs.len();
                self.instrs[jz].a = else_start as i32; // rel computed at pack time from indices
                if !else_b.is_empty() {
                    self.gen_block(else_b, env, f, refs)?;
                    let end = self.instrs.len();
                    self.instrs[jend.unwrap()].a = end as i32;
                }
                Ok(())
            }
            Stmt::While(cond, body) => {
                let top = self.instrs.len();
                let cvt = self.gen_expr(cond, env, f, refs)?;
                debug_assert_eq!(cvt, VT::Bool);
                let jz = self.instrs.len();
                self.instrs.push(Instr::a(OP_JZ, 0));
                self.gen_block(body, env, f, refs)?;
                self.instrs.push(Instr::a(OP_JMP, top as i32));
                let end = self.instrs.len();
                self.instrs[jz].a = end as i32;
                Ok(())
            }
            Stmt::For(var, ent, body) => {
                let ei = *self.eidx.get(ent).ok_or_else(|| format!("unknown entity {}", ent))?;
                let slot = match env.locals.iter().position(|(n, _)| n == var) {
                    Some(s) => s,
                    None => env.locals.len(),
                };
                env.locals.push((var.clone(), VT::Ent(ei)));
                if env.locals.len() > self.cur_max_locals {
                    self.cur_max_locals = env.locals.len();
                }
                let bgn = self.instrs.len();
                self.instrs.push(Instr::abc(OP_FOR_BGN, ei as i32, slot as i32, 0)); // c patched to exit
                self.gen_block(body, env, f, refs)?;
                self.instrs.push(Instr::a(OP_FOR_ADV, (bgn + 1) as i32)); // back to body start
                let exit = self.instrs.len();
                self.instrs[bgn].c = exit as i32;
                env.locals.pop();
                Ok(())
            }
            Stmt::Spawn(ent, inits) => {
                let ei = *self.eidx.get(ent).ok_or_else(|| format!("unknown entity {}", ent))?;
                let ent_info = &self.ctx.entities[ei];
                let mut tfs = Vec::new();
                // evaluate in source order; convert to field types
                for (fname, e) in inits {
                    let fi = ent_info.fields.iter().position(|x| &x.name == fname)
                        .ok_or_else(|| format!("entity {} has no field {}", ent, fname))?;
                    let fvt = ent_info.fields[fi].vt;
                    let vt = self.gen_expr(e, env, f, refs)?;
                    self.convert(vt, fvt);
                    tfs.push(self.tf_of(ei as u8, fname)? as u16);
                }
                let n = tfs.len() as i32;
                self.instrs.push(Instr { op: OP_SPAWN, a: ei as i32, b: n, c: 0, extra: tfs });
                Ok(())
            }
            Stmt::Kill(e) => {
                // entity expr must be a variable; static type gives entity id
                if let Expr::Ident(v) = e {
                    let slot = self.slot_of(env, v);
                    let ei = match env.locals.iter().find(|(n, _)| n == v).map(|(_, vt)| *vt) {
                        Some(VT::Ent(i)) => i,
                        _ => return Err(crate::diag::internal("kill target has no entity type")),
                    };
                    self.instrs.push(Instr::ab(OP_KILL, slot as i32, ei as i32));
                    Ok(())
                } else {
                    Err(crate::diag::internal("kill target must be a variable"))
                }
            }
            Stmt::Sfx(a) => {
                let p = self.dom_pos(&self.ctx.sound, *a, "sound")?;
                self.instrs.push(Instr::a(OP_SFX, p as i32));
                Ok(())
            }
            Stmt::Music(a) => {
                let p = self.dom_pos(&self.ctx.music.iter().map(|m| m.atom).collect::<Vec<_>>(), *a, "music")?;
                self.instrs.push(Instr::a(OP_MUSIC, p as i32));
                Ok(())
            }
            Stmt::StopMusic => { self.instrs.push(Instr::new(OP_STOP_MUSIC)); Ok(()) }
            Stmt::Shake(amp) => {
                self.instrs.push(Instr::a(OP_SHAKE, *amp as i32));
                Ok(())
            }
            Stmt::Camera(x, y) => {
                // both args to fixed; engine clamps to the world
                self.gen_num_fixed(x, env, f, refs)?;
                self.gen_num_fixed(y, env, f, refs)?;
                self.instrs.push(Instr::new(OP_CAMERA));
                Ok(())
            }
            Stmt::Cam3(x, y, z, yaw, pitch) => {
                // v11 3D: push x, y, z (fixed), yaw, pitch (ang as-is) —
                // VM pops pitch, yaw, z, y, x into the page-0 registers.
                self.gen_num_fixed(x, env, f, refs)?;
                self.gen_num_fixed(y, env, f, refs)?;
                self.gen_num_fixed(z, env, f, refs)?;
                self.gen_expr(yaw, env, f, refs)?;
                self.gen_expr(pitch, env, f, refs)?;
                self.instrs.push(Instr::new(OP_CAM3));
                Ok(())
            }
            Stmt::Draw3d { arr, n, x, y, z, yaw, rgba } => {
                // v11 3D mesh pass: array id rides in `a`; stack args are
                // n (int), x/y/z (fixed), yaw (ang), rgba (int) — the VM
                // pops rgba, yaw, z, y, x, n and hands off to render3d.
                let id = *self.arrs.get(&(GLOBAL_ENT, arr.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("draw3d array {} missing", arr)))?;
                self.gen_expr(n, env, f, refs)?;
                self.gen_num_fixed(x, env, f, refs)?;
                self.gen_num_fixed(y, env, f, refs)?;
                self.gen_num_fixed(z, env, f, refs)?;
                self.gen_expr(yaw, env, f, refs)?;
                self.gen_expr(rgba, env, f, refs)?;
                self.instrs.push(Instr::a(OP_DRAW3D, id as i32));
                Ok(())
            }
            Stmt::Draw3DI { verts, nv, idx, ni, x, y, z, yaw, rgba } => {
                // v12 indexed mesh pass: a = vertex buffer id, b = index
                // buffer id; stack args nv, ni, x, y, z, yaw, rgba pushed in
                // source order — the VM pops rgba, yaw, z, y, x, ni, nv.
                let vid = *self.arrs.get(&(GLOBAL_ENT, verts.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("draw3di vertex buffer {} missing", verts)))?;
                let iid = *self.arrs.get(&(GLOBAL_ENT, idx.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("draw3di index buffer {} missing", idx)))?;
                self.gen_expr(nv, env, f, refs)?;
                self.gen_expr(ni, env, f, refs)?;
                self.gen_num_fixed(x, env, f, refs)?;
                self.gen_num_fixed(y, env, f, refs)?;
                self.gen_num_fixed(z, env, f, refs)?;
                self.gen_expr(yaw, env, f, refs)?;
                self.gen_expr(rgba, env, f, refs)?;
                self.instrs.push(Instr::ab(OP_DRAW3DI, vid as i32, iid as i32));
                Ok(())
            }
            Stmt::QuatAA { q, off, ax, ay, az, ang } => {
                // v12: quat_aa(q, off, ax, ay, az, ang) — array id pushed as
                // a raw int constant, then off/axis/angle. One native op.
                let qid = *self.arrs.get(&(GLOBAL_ENT, q.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("quat_aa array {} missing", q)))?;
                self.push_const(qid as i32);
                self.gen_expr(off, env, f, refs)?;
                self.gen_num_fixed(ax, env, f, refs)?;
                self.gen_num_fixed(ay, env, f, refs)?;
                self.gen_num_fixed(az, env, f, refs)?;
                self.gen_expr(ang, env, f, refs)?;
                self.instrs.push(Instr::new(OP_QUAT_AA));
                Ok(())
            }
            Stmt::QMul { d, doff, a, aoff, b, boff } => {
                let did = *self.arrs.get(&(GLOBAL_ENT, d.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("qmul array {} missing", d)))?;
                let aid = *self.arrs.get(&(GLOBAL_ENT, a.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("qmul array {} missing", a)))?;
                let bid = *self.arrs.get(&(GLOBAL_ENT, b.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("qmul array {} missing", b)))?;
                self.push_const(did as i32);
                self.gen_expr(doff, env, f, refs)?;
                self.push_const(aid as i32);
                self.gen_expr(aoff, env, f, refs)?;
                self.push_const(bid as i32);
                self.gen_expr(boff, env, f, refs)?;
                self.instrs.push(Instr::new(OP_Q_MUL));
                Ok(())
            }
            Stmt::M4QT { m, moff, q, qoff, tx, ty, tz } => {
                let mid = *self.arrs.get(&(GLOBAL_ENT, m.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("m4qt array {} missing", m)))?;
                let qid = *self.arrs.get(&(GLOBAL_ENT, q.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("m4qt array {} missing", q)))?;
                self.push_const(mid as i32);
                self.gen_expr(moff, env, f, refs)?;
                self.push_const(qid as i32);
                self.gen_expr(qoff, env, f, refs)?;
                self.gen_num_fixed(tx, env, f, refs)?;
                self.gen_num_fixed(ty, env, f, refs)?;
                self.gen_num_fixed(tz, env, f, refs)?;
                self.instrs.push(Instr::new(OP_M4_QT));
                Ok(())
            }
            Stmt::M4Mul { d, doff, a, aoff, b, boff } => {
                let did = *self.arrs.get(&(GLOBAL_ENT, d.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("m4mul array {} missing", d)))?;
                let aid = *self.arrs.get(&(GLOBAL_ENT, a.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("m4mul array {} missing", a)))?;
                let bid = *self.arrs.get(&(GLOBAL_ENT, b.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("m4mul array {} missing", b)))?;
                self.push_const(did as i32);
                self.gen_expr(doff, env, f, refs)?;
                self.push_const(aid as i32);
                self.gen_expr(aoff, env, f, refs)?;
                self.push_const(bid as i32);
                self.gen_expr(boff, env, f, refs)?;
                self.instrs.push(Instr::new(OP_M4_MUL));
                Ok(())
            }
            Stmt::SkinV { v, voff, m, moff, x, y, z } => {
                let vid = *self.arrs.get(&(GLOBAL_ENT, v.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("skinv array {} missing", v)))?;
                let mid = *self.arrs.get(&(GLOBAL_ENT, m.clone()))
                    .ok_or_else(|| crate::diag::internal(&format!("skinv array {} missing", m)))?;
                self.push_const(vid as i32);
                self.gen_expr(voff, env, f, refs)?;
                self.push_const(mid as i32);
                self.gen_expr(moff, env, f, refs)?;
                self.gen_num_fixed(x, env, f, refs)?;
                self.gen_num_fixed(y, env, f, refs)?;
                self.gen_num_fixed(z, env, f, refs)?;
                self.instrs.push(Instr::new(OP_SKIN3));
                Ok(())
            }
            Stmt::MusicVol(vol) => {
                self.instrs.push(Instr::a(OP_MUSIC_VOL, *vol as i32));
                Ok(())
            }
            Stmt::Save(ix, v) => {
                // push slot index, then the RAW value (no conversion — slots
                // are untyped 32-bit words; the read side picks the type)
                self.gen_expr(ix, env, f, refs)?;
                self.gen_expr(v, env, f, refs)?;
                self.instrs.push(Instr::new(OP_SAVE));
                Ok(())
            }
            Stmt::Goto(a) => {
                let scene_atoms: Vec<u16> = self.ctx.scenes.iter().map(|(s, _)| *s).collect();
                let p = self.dom_pos(&scene_atoms, *a, "scene")?;
                self.instrs.push(Instr::a(OP_GOTO, p as i32));
                Ok(())
            }
            Stmt::Draw(spr, x, y, rot, scale, rgba) => {
                let spr_atoms: Vec<u16> = self.ctx.sprites.iter().map(|s| s.atom).collect();
                let p = self.dom_pos(&spr_atoms, *spr, "sprite")?;
                // x, y: to fixed
                self.gen_num_fixed(x, env, f, refs)?;
                self.gen_num_fixed(y, env, f, refs)?;
                // rot: ang value or int literal
                let rvt = self.gen_expr(rot, env, f, refs)?;
                if rvt == VT::Int {
                    // int literal -> ang units 1:1 (already pushed)
                }
                self.gen_num_fixed(scale, env, f, refs)?;
                // rgba
                let _cvt = self.gen_expr(rgba, env, f, refs)?;
                self.instrs.push(Instr::a(OP_DRAW, p as i32));
                Ok(())
            }
            Stmt::DrawText(a, x, y, rgba) => {
                let text_atoms: Vec<u16> = self.ctx.texts.iter().map(|(t, _)| *t).collect();
                let p = self.dom_pos(&text_atoms, *a, "text")?;
                self.gen_num_fixed(x, env, f, refs)?;
                self.gen_num_fixed(y, env, f, refs)?;
                self.gen_expr(rgba, env, f, refs)?;
                self.instrs.push(Instr::a(OP_DRAW_TEXT, p as i32));
                Ok(())
            }
            Stmt::DrawNum(v, x, y, rgba) => {
                let vvt = self.gen_expr(v, env, f, refs)?;
                if vvt == VT::Fixed {
                    self.instrs.push(Instr::new(OP_F2I));
                }
                self.gen_num_fixed(x, env, f, refs)?;
                self.gen_num_fixed(y, env, f, refs)?;
                self.gen_expr(rgba, env, f, refs)?;
                self.instrs.push(Instr::new(OP_DRAW_NUM));
                Ok(())
            }
            Stmt::CallStmt(name, args) => {
                let fi = *self.fidx.get(name)
                    .ok_or_else(|| format!("unknown fn {}", name))?;
                let target = &self.ctx.fns[fi];
                for (i, (_, pvt)) in target.params.iter().enumerate() {
                    let vt = self.gen_expr(&args[i], env, f, refs)?;
                    self.convert(vt, *pvt);
                }
                self.instrs.push(Instr::ab(OP_CALL_FN, fi as i32, args.len() as i32));
                Ok(())
            }
            Stmt::CallTable(tname, key, entvar) => {
                let ti = self.ctx.tables.iter().position(|t| &t.name == tname)
                    .ok_or_else(|| format!("unknown table {}", tname))?;
                self.gen_expr(key, env, f, refs)?;
                let slot = self.slot_of(env, entvar);
                self.instrs.push(Instr::ab(OP_CALL_TBL, ti as i32, slot as i32));
                Ok(())
            }
            Stmt::Return => { self.instrs.push(Instr::new(OP_RET)); Ok(()) }
            // v8 OPTIMIZER subsystem 3: Galois-lane candidate loop.
            // Emitted by opt::physics; the mask local was materialized by the
            // COLLIDE_MASK let right before this statement.
            Stmt::ForMask(var, ent_name, mask_local, body) => {
                let ei = *self.eidx.get(ent_name)
                    .ok_or_else(|| format!("unknown entity {}", ent_name))?;
                let slot = match env.locals.iter().position(|(n, _)| n == var) {
                    Some(s) => s,
                    None => env.locals.len(),
                };
                env.locals.push((var.clone(), VT::Ent(ei)));
                if env.locals.len() > self.cur_max_locals {
                    self.cur_max_locals = env.locals.len();
                }
                // push the mask, then open the ctz-scan loop. The window
                // base (from the mask's Let) rides in b's high bits.
                let mslot = self.slot_of(env, mask_local);
                self.instrs.push(Instr::a(OP_LD_LOCAL, mslot as i32));
                let base = self.mask_base.get(mask_local).copied().unwrap_or(0);
                let bgn = self.instrs.len();
                self.instrs.push(Instr::abc(OP_FOR_MASK_BGN, slot as i32, ei as i32 | (base << 8), 0));
                self.gen_block(body, env, f, refs)?;
                self.instrs.push(Instr::a(OP_FOR_MASK_ADV, (bgn + 1) as i32));
                let exit = self.instrs.len();
                self.instrs[bgn].c = exit as i32;
                env.locals.pop();
                Ok(())
            }
            // opt::fsm replaces these before codegen; reaching here is a bug.
            Stmt::FsmStep(_) => Err("internal: FsmStep survived the fsm pass".into()),
        }
    }

    fn gen_assign(&mut self, target: &Expr, op: AssignOp, val: &Expr, env: &mut Env, f: &FnInfo, refs: &CtxRefs) -> Result<(), String> {
        // v11 ARRAYS: element store / compound element update. The index is
        // evaluated exactly ONCE (DUP keeps it under the element for the
        // store), so rand()-carrying indices stay deterministic.
        if let Expr::Index(base, ix) = target {
            let arr_id = *refs.arr_index.get(&match &**base {
                Expr::Ident(name) => (GLOBAL_ENT, name.clone()),
                Expr::Field(inner, fname) => {
                    let v = match &**inner {
                        Expr::Ident(v) => v.clone(),
                        _ => return Err("internal: bad array base".into()),
                    };
                    let ei = match env.locals.iter().find(|(n, _)| *n == v).map(|(_, vt)| *vt) {
                        Some(VT::Ent(ei)) => ei,
                        _ => return Err(format!("'{}' is not an entity variable", v)),
                    };
                    (ei as u8, fname.clone())
                }
                _ => return Err("internal: bad array base".into()),
            }).ok_or_else(|| String::from("internal: array id missing"))?;
            let a = &self.ctx.arrays[arr_id];
            let elem_vt = a.elem;
            // index (once) ...
            self.gen_expr(ix, env, f, refs)?;
            match op {
                AssignOp::Set => {
                    // [idx] -> push val -> store
                    let vvt = self.gen_expr(val, env, f, refs)?;
                    self.convert(vvt, elem_vt);
                }
                _ => {
                    // [idx] -> DUP -> load elem -> push val -> op -> store
                    self.instrs.push(Instr::new(OP_DUP));
                    let (ld_op, _st_op, slot) = if a.ent == GLOBAL_ENT {
                        (OP_LD_GARR, OP_ST_GARR, 0usize)
                    } else {
                        let v = match &**base {
                            Expr::Field(inner, _) => match &**inner {
                                Expr::Ident(v) => v.clone(),
                                _ => unreachable!(),
                            },
                            _ => unreachable!(),
                        };
                        (OP_LD_EARR, OP_ST_EARR, self.slot_of(env, &v))
                    };
                    if a.ent == GLOBAL_ENT {
                        self.instrs.push(Instr::a(ld_op, arr_id as i32));
                    } else {
                        self.instrs.push(Instr::ab(ld_op, slot as i32, arr_id as i32));
                    }
                    let vvt = self.gen_expr(val, env, f, refs)?;
                    match (elem_vt, vvt) {
                        (VT::Fixed, VT::Int) => { self.instrs.push(Instr::new(OP_I2F)); }
                        _ => {}
                    }
                    let opn = match (elem_vt, op) {
                        (VT::Fixed, AssignOp::Add) => OP_ADD_F,
                        (VT::Fixed, AssignOp::Sub) => OP_SUB_F,
                        (VT::Fixed, AssignOp::Mul) => OP_MUL_F,
                        (_, AssignOp::Add) => OP_ADD_I,
                        (_, AssignOp::Sub) => OP_SUB_I,
                        (_, AssignOp::Mul) => OP_MUL_I,
                        (VT::Fixed, AssignOp::Set) | (_, AssignOp::Set) => unreachable!(),
                    };
                    self.instrs.push(Instr::new(opn));
                }
            }
            // store: pops result, pops idx
            if a.ent == GLOBAL_ENT {
                self.instrs.push(Instr::a(OP_ST_GARR, arr_id as i32));
            } else {
                let v = match &**base {
                    Expr::Field(inner, _) => match &**inner {
                        Expr::Ident(v) => v.clone(),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                };
                let slot = self.slot_of(env, &v);
                self.instrs.push(Instr::ab(OP_ST_EARR, slot as i32, arr_id as i32));
            }
            return Ok(());
        }
        // determine target kind + type
        let (tvt, ent_u8, tf_or_slot, is_local, name) = match target {
            Expr::Ident(name) => {
                if let Some(slot) = env.locals.iter().position(|(n, _)| n == name) {
                    let vt = env.locals[slot].1;
                    (vt, 0xFFu8, slot as usize, true, name.clone())
                } else if let Some(gi) = self.gidx.get(name) {
                    (self.ctx.globals[*gi].vt, GLOBAL_ENT, *gi, false, name.clone())
                } else { return Err(format!("unknown variable {}", name)); }
            }
            Expr::Field(base, fname) => {
                match &**base {
                    Expr::Ident(v) => {
                        if let Some((_, VT::Ent(ei))) = env.locals.iter().find(|(n, _)| n == v) {
                            let fi = self.ctx.entities[*ei].fields.iter().position(|x| &x.name == fname)
                                .ok_or_else(|| format!("entity {} no field {}", v, fname))?;
                            (self.ctx.entities[*ei].fields[fi].vt, *ei as u8, fi, false, fname.clone())
                        } else { return Err(format!("{} not entity", v)); }
                    }
                    _ => return Err("bad target".into()),
                }
            }
            _ => return Err("bad target".into()),
        };
        // tf index for non-local targets
        let tf_idx = if is_local { None } else {
            Some(self.tf_of(ent_u8, &name)?)
        };

        // load current value for compound ops
        if op != AssignOp::Set {
            if is_local {
                self.instrs.push(Instr::a(OP_LD_LOCAL, tf_or_slot as i32));
            } else if ent_u8 == GLOBAL_ENT {
                self.instrs.push(Instr::a(OP_LD_GLBL, tf_idx.unwrap() as i32));
            } else {
                let slot = self.slot_of(env, &field_var_name(target));
                self.instrs.push(Instr::ab(OP_LD_ENT, slot as i32, tf_idx.unwrap() as i32));
            }
        }

        // value
        let vvt = self.gen_expr(val, env, f, refs)?;

        use AssignOp::*;
        // conversion for compound arithmetic
        if op != Set {
            match (tvt, vvt) {
                (VT::Fixed, VT::Int) => { self.instrs.push(Instr::new(OP_I2F)); }
                _ => {}
            }
        } else {
            self.convert(vvt, tvt);
        }
        // apply op
        match (tvt, op) {
            (VT::Fixed, Add) => self.instrs.push(Instr::new(OP_ADD_F)),
            (VT::Fixed, Sub) => self.instrs.push(Instr::new(OP_SUB_F)),
            (VT::Fixed, Mul) => self.instrs.push(Instr::new(OP_MUL_F)),
            (VT::Int, Add) | (VT::Ang, Add) => self.instrs.push(Instr::new(OP_ADD_I)),
            (VT::Int, Sub) | (VT::Ang, Sub) => self.instrs.push(Instr::new(OP_SUB_I)),
            (VT::Int, Mul) => self.instrs.push(Instr::new(OP_MUL_I)),
            (VT::Bool, Set) | (VT::Ang, Set) | (VT::Int, Set) | (VT::Fixed, Set) => {}
            _ => return Err(format!("internal: bad compound assign {:?} {:?}", tvt, op)),
        }
        // auto-wrap injection: POSITION fields — exactly "x"/"y", matching
        // the auto-integrator's registered x/vx, y/vy pairs — get a bitwise
        // AND mask on store (the documented "you never write the wrap" rule).
        //
        // CRITICAL: match EXACT names, never suffixes. The old ends_with('x'/'y')
        // test also caught vx/vy (velocities!) — masking those corrupted their
        // sign domain: a stored -98px/frame became 130,974 raw, after which
        // every drag (`v -= v/40`), clamp, and accumulation operated on a
        // garbage magnitude. Negative velocities then "unwound" through the
        // wrap domain at up to 250px/frame — the ships-and-enemies-
        // teleporting bug. Only integrator-managed positions wrap; velocities
        // stay pure Q24.8 two's-complement.
        if self.ctx.game.wrap && tvt == VT::Fixed && (name == "x" || name == "y") {
            // v7: wrap domain is the WORLD (defaults to the screen — v1
            // behavior when no world_w/world_h is declared)
            let dim = if name == "x" {
                self.ctx.game.world_w.unwrap_or(self.ctx.game.width)
            } else {
                self.ctx.game.world_h.unwrap_or(self.ctx.game.height)
            };
            let mask = ((dim as i64) << 8) - 1;
            let mask = mask as i32;
            if (-128..=127).contains(&mask) {
                self.instrs.push(Instr::a(OP_PUSH_S8, mask));
            } else {
                let pi = self.pool_idx(mask);
                self.instrs.push(Instr::a(OP_PUSH_POOL, pi as i32));
            }
            self.instrs.push(Instr::new(OP_AND_I));
        }
        // store
        if is_local {
            self.instrs.push(Instr::a(OP_ST_LOCAL, tf_or_slot as i32));
        } else if ent_u8 == GLOBAL_ENT {
            self.instrs.push(Instr::a(OP_ST_GLBL, tf_idx.unwrap() as i32));
        } else {
            let slot = self.slot_of(env, &field_var_name(target));
            self.instrs.push(Instr::ab(OP_ST_ENT, slot as i32, tf_idx.unwrap() as i32));
        }
        Ok(())
    }

    fn convert(&mut self, from: VT, to: VT) {
        match (from, to) {
            (VT::Int, VT::Fixed) => self.instrs.push(Instr::new(OP_I2F)),
            (VT::Fixed, VT::Int) => self.instrs.push(Instr::new(OP_F2I)),
            _ => {}
        }
    }

    fn slot_of(&self, env: &Env, name: &str) -> usize {
        env.locals.iter().position(|(n, _)| n == name)
            .unwrap_or(0) // checker guarantees existence
    }

    fn gen_num_fixed(&mut self, e: &Expr, env: &mut Env, f: &FnInfo, refs: &CtxRefs) -> Result<(), String> {
        let vt = self.gen_expr(e, env, f, refs)?;
        if vt == VT::Int {
            self.instrs.push(Instr::new(OP_I2F));
        }
        Ok(())
    }

    /// Typed view for codegen decisions: optimizer intrinsics carry their
    /// proven result type; everything else re-derives via the checker.
    fn ty_of(&self, e: &Expr, env: &Env, f: &FnInfo, refs: &CtxRefs) -> VT {
        match e {
            Expr::Intrin(kind, _, args, _) if *kind == crate::ast::INTR_SEL => {
                // SEL takes the type of its THEN value (the select pass
                // proved both branches share one value class)
                refs.expr_type(&args[1], &env.locals, f).unwrap_or(VT::Int)
            }
            Expr::Intrin(kind, _, _, _) => match *kind {
                crate::ast::INTR_BIT01 | crate::ast::INTR_FSM_NEXT | crate::ast::INTR_COLL_MASK => VT::Int,
                _ => VT::Int,
            },
            _ => refs.expr_type(e, &env.locals, f).unwrap_or(VT::Int),
        }
    }

    fn gen_expr(&mut self, e: &Expr, env: &mut Env, f: &FnInfo, refs: &CtxRefs) -> Result<VT, String> {
        // optimizer intrinsics carry proven types: no checker re-derivation
        if let Expr::Intrin(kind, aux, args, consts) = e {
            return match *kind {
                crate::ast::INTR_BIT01 => {
                    // bools ARE 0/1 i32 at VM level: evaluate, retype to Int
                    self.gen_expr(&args[0], env, f, refs)?;
                    Ok(VT::Int)
                }
                crate::ast::INTR_FSM_NEXT => {
                    // stack: em, s -> OP_FSM_NEXT pops s then em
                    self.gen_expr(&args[0], env, f, refs)?;
                    self.gen_expr(&args[1], env, f, refs)?;
                    self.instrs.push(Instr::a(OP_FSM_NEXT, *aux as i32));
                    Ok(VT::Int)
                }
                crate::ast::INTR_COLL_MASK => {
                    // (cx, cy) fixed on the stack; hw/hh live in the pool;
                    // the type's x/y tf indices ride in `extra`, and the
                    // 32-lane WINDOW base rides in consts[2] (v9) — packed
                    // into the spare high bits of the `a` operand.
                    self.gen_expr(&args[0], env, f, refs)?;
                    self.gen_expr(&args[1], env, f, refs)?;
                    let hw = self.pool_idx(consts.first().copied().unwrap_or(0));
                    let hh = self.pool_idx(consts.get(1).copied().unwrap_or(0));
                    let base = consts.get(2).copied().unwrap_or(0);
                    let x_tf = self.tf_of(*aux as u8, "x")? as i32;
                    let y_tf = self.tf_of(*aux as u8, "y")? as i32;
                    self.instrs.push(Instr {
                        op: OP_COLLIDE_MASK, a: (*aux as i32) | (base << 8), b: hw as i32, c: hh as i32,
                        extra: vec![x_tf as u16, y_tf as u16],
                    });
                    Ok(VT::Int)
                }
                crate::ast::INTR_SEL => {
                    // branchless conditional move (subsystem 6): args are
                    // cond, then, else pushed in order -> OP_SEL pops else,
                    // then, cond and pushes the selected value. The result
                    // takes the THEN value's type (both branches were proven
                    // to share one value class by opt::select).
                    self.gen_expr(&args[0], env, f, refs)?;
                    self.gen_expr(&args[1], env, f, refs)?;
                    self.gen_expr(&args[2], env, f, refs)?;
                    self.instrs.push(Instr::new(OP_SEL));
                    Ok(refs.expr_type(&args[1], &env.locals, f).unwrap_or(VT::Int))
                }
                other => Err(format!("internal: unknown intrinsic kind {}", other)),
            };
        }
        let vt = refs.expr_type(e, &env.locals, f)?;
        match e {
            // optimizer intrinsics were handled (and returned) above; this arm
            // exists only for exhaustiveness — reaching it is a compiler bug.
            Expr::Intrin(..) => Err("internal: intrinsic reached late codegen".into()),
            Expr::IntLit(v) => { self.push_const(*v as i32); Ok(VT::Int) }
            Expr::FixLit(v) => { self.push_const(*v); Ok(VT::Fixed) }
            Expr::BoolLit(b) => { self.push_const(if *b { 1 } else { 0 }); Ok(VT::Bool) }
            Expr::Atom(_) => Err("internal: atom in expression".into()),
            Expr::Ident(name) => {
                if let Some(slot) = env.locals.iter().position(|(n, _)| n == name) {
                    self.instrs.push(Instr::a(OP_LD_LOCAL, slot as i32));
                    return Ok(env.locals[slot].1);
                }
                if let Some(gi) = self.gidx.get(name) {
                    let tfi = self.tf_of(GLOBAL_ENT, name)?;
                    self.instrs.push(Instr::a(OP_LD_GLBL, tfi as i32));
                    return Ok(self.ctx.globals[*gi].vt);
                }
                // v7 engine registers (checker rejects user shadowing)
                match name.as_str() {
                    "cam_x" => { self.instrs.push(Instr::new(OP_CAM_X)); return Ok(VT::Fixed); }
                    "cam_y" => { self.instrs.push(Instr::new(OP_CAM_Y)); return Ok(VT::Fixed); }
                    _ => {}
                }
                Err(format!("unknown var {}", name))
            }
            Expr::Field(base, fname) => {
                if let Expr::Ident(v) = &**base {
                    if let Some((slot, VT::Ent(ei))) = env.locals.iter().enumerate()
                        .find(|(_, (n, _))| n == v).map(|(s, (_, vt))| (s, *vt)) {
                        let tfi = self.tf_of(ei as u8, fname)?;
                        self.instrs.push(Instr::ab(OP_LD_ENT, slot as i32, tfi as i32));
                        return Ok(vt);
                    }
                }
                Err("internal: bad field access".into())
            }
            Expr::Index(base, ix) => {
                // v11 ARRAYS: resolve id, evaluate index (checker proved it
                // int + literal-in-range), then load the raw 32-bit cell.
                let key = match &**base {
                    Expr::Ident(name) => (GLOBAL_ENT, name.clone()),
                    Expr::Field(inner, fname) => {
                        let v = match &**inner {
                            Expr::Ident(v) => v.clone(),
                            _ => return Err("internal: bad array base".into()),
                        };
                        let ei = match env.locals.iter().find(|(n, _)| *n == v).map(|(_, vt)| *vt) {
                            Some(VT::Ent(ei)) => ei,
                            _ => return Err(format!("'{}' is not an entity variable", v)),
                        };
                        (ei as u8, fname.clone())
                    }
                    _ => return Err("internal: bad array base".into()),
                };
                let arr_id = *self.arrs.get(&key)
                    .ok_or_else(|| String::from("internal: array id missing"))?;
                let elem = self.ctx.arrays[arr_id].elem;
                self.gen_expr(ix, env, f, refs)?;
                match self.ctx.arrays[arr_id].ent {
                    GLOBAL_ENT => self.instrs.push(Instr::a(OP_LD_GARR, arr_id as i32)),
                    _ => {
                        let v = match &**base {
                            Expr::Field(inner, _) => match &**inner {
                                Expr::Ident(v) => v.clone(),
                                _ => unreachable!(),
                            },
                            _ => unreachable!(),
                        };
                        let slot = self.slot_of(env, &v);
                        self.instrs.push(Instr::ab(OP_LD_EARR, slot as i32, arr_id as i32));
                    }
                }
                Ok(elem)
            }
            Expr::Unary(UnOp::Neg, inner) => {
                let t = self.gen_expr(inner, env, f, refs)?;
                self.instrs.push(Instr::new(OP_NEG));
                Ok(t)
            }
            Expr::Unary(UnOp::Not, inner) => {
                self.gen_expr(inner, env, f, refs)?;
                self.instrs.push(Instr::new(OP_NOT_B));
                Ok(VT::Bool)
            }
            Expr::Binary(op, a, b) => {
                let at = self.ty_of(a, env, f, refs);
                let bt = self.ty_of(b, env, f, refs);
                use BinOp::*;
                // emit with int->fixed widening on the int side
                match (at, bt) {
                    (VT::Fixed, VT::Int) => {
                        self.gen_expr(a, env, f, refs)?;
                        self.gen_expr(b, env, f, refs)?;
                        self.instrs.push(Instr::new(OP_I2F));
                    }
                    (VT::Int, VT::Fixed) => {
                        self.gen_expr(a, env, f, refs)?;
                        self.instrs.push(Instr::new(OP_I2F));
                        self.gen_expr(b, env, f, refs)?;
                    }
                    _ => {
                        self.gen_expr(a, env, f, refs)?;
                        self.gen_expr(b, env, f, refs)?;
                    }
                }
                // operator selection
                let is_fix = at == VT::Fixed || bt == VT::Fixed;
                let opn = match op {
                    Add => if is_fix { OP_ADD_F } else { OP_ADD_I },
                    Sub => if is_fix { OP_SUB_F } else { OP_SUB_I },
                    Mul => if is_fix { OP_MUL_F } else { OP_MUL_I },
                    Div => if is_fix { OP_DIV_F } else { OP_DIV_I },
                    Mod => OP_MOD_I,
                    And => OP_AND_I, Or => OP_OR_I, Xor => OP_XOR_I,
                    Shl => OP_SHL, Shr => OP_SHR,
                    Eq => if is_fix { OP_EQ_F } else { OP_EQ_I },
                    Ne => if is_fix { OP_NE_F } else { OP_NE_I },
                    Lt => if is_fix { OP_LT_F } else { OP_LT_I },
                    Gt => if is_fix { OP_GT_F } else { OP_GT_I },
                    Le => if is_fix { OP_LE_F } else { OP_LE_I },
                    Ge => if is_fix { OP_GE_F } else { OP_GE_I },
                    LAnd => OP_AND_B, LOr => OP_OR_B,
                };
                self.instrs.push(Instr::new(opn));
                Ok(match op {
                    LAnd | LOr | Eq | Ne | Lt | Gt | Le | Ge => VT::Bool,
                    _ => if is_fix { VT::Fixed } else { at },
                })
            }
            Expr::Call(name, args) => {
                match name.as_str() {
                    "key" => {
                        if let Expr::Atom(a) = &args[0] {
                            let p = self.dom_pos(&self.ctx.keys, *a, "key")?;
                            self.instrs.push(Instr::a(OP_KEY, p as i32));
                            Ok(VT::Bool)
                        } else { Err("internal: key()".into()) }
                    }
                    "sin" | "cos" => {
                        self.gen_expr(&args[0], env, f, refs)?;
                        self.instrs.push(Instr::new(if name == "sin" { OP_SIN } else { OP_COS }));
                        Ok(VT::Fixed)
                    }
                    "ang" => {
                        self.gen_expr(&args[0], env, f, refs)?;
                        Ok(VT::Ang) // int/ang 1:1 units
                    }
                    "fixed" => {
                        let t = self.gen_expr(&args[0], env, f, refs)?;
                        if t == VT::Int { self.instrs.push(Instr::new(OP_I2F)); }
                        Ok(VT::Fixed)
                    }
                    "int" => {
                        let t = self.gen_expr(&args[0], env, f, refs)?;
                        if t == VT::Fixed { self.instrs.push(Instr::new(OP_F2I)); }
                        Ok(VT::Int)
                    }
                    "rand" => {
                        self.gen_expr(&args[0], env, f, refs)?;
                        self.instrs.push(Instr::new(OP_RAND_MAX));
                        Ok(VT::Int)
                    }
                    "anim" => {
                        self.gen_expr(&args[1], env, f, refs)?;
                        if let Expr::Atom(a) = &args[0] {
                            let anim_atoms: Vec<u16> = self.ctx.anims.iter().map(|x| x.atom).collect();
                            let p = self.dom_pos(&anim_atoms, *a, "anim")?;
                            self.instrs.push(Instr::a(OP_ANIM, p as i32));
                            Ok(VT::Fixed)
                        } else { Err("internal: anim()".into()) }
                    }
                    "count" => {
                        if let Expr::Ident(ent) = &args[0] {
                            let ei = *self.eidx.get(ent).ok_or_else(|| format!("unknown entity {}", ent))?;
                            self.instrs.push(Instr::a(OP_COUNT, ei as i32));
                            Ok(VT::Int)
                        } else { Err("internal: count()".into()) }
                    }
                    // v7: SRAM persistence (raw 32-bit slots)
                    "saved" | "savedf" => {
                        self.gen_expr(&args[0], env, f, refs)?;
                        self.instrs.push(Instr::new(OP_SAVED));
                        Ok(if name == "savedf" { VT::Fixed } else { VT::Int })
                    }
                    // v7: overflow-safe distance (i64 internals)
                    "dist" => {
                        for a in args { self.gen_num_fixed(a, env, f, refs)?; }
                        self.instrs.push(Instr::new(OP_DIST));
                        Ok(VT::Fixed)
                    }
                    // v7: screen-convention aim (integer CORDIC)
                    "atan2" => {
                        self.gen_num_fixed(&args[0], env, f, refs)?;
                        self.gen_num_fixed(&args[1], env, f, refs)?;
                        self.instrs.push(Instr::new(OP_ATAN2));
                        Ok(VT::Ang)
                    }
                    // v8 OPTIMIZER subsystem 3: swept Minkowski interval test
                    // (stack: ax ay avx avy bx by hw hh -> bool). ONE opcode,
                    // the whole continuous 1D inequality solve per axis.
                    "swept_hit" => {
                        for a in args { self.gen_num_fixed(a, env, f, refs)?; }
                        self.instrs.push(Instr::new(OP_SWEPT));
                        Ok(VT::Bool)
                    }
                    // ---------------- v11 3D + arrays ----------------
                    // proj3: pops z, y, x -> pushes scale; stash readable via
                    // projx/projy/projok (evaluation-order contract: they read
                    // the LAST proj3 in the same expression/statement).
                    "proj3" => {
                        for a in args { self.gen_num_fixed(a, env, f, refs)?; }
                        self.instrs.push(Instr::new(OP_PROJ3));
                        Ok(VT::Fixed)
                    }
                    "projx" => { self.instrs.push(Instr::new(OP_PROJ_X)); Ok(VT::Fixed) }
                    "projy" => { self.instrs.push(Instr::new(OP_PROJ_Y)); Ok(VT::Fixed) }
                    "projok" => { self.instrs.push(Instr::new(OP_PROJ_OK)); Ok(VT::Bool) }
                    // alen: the capacity is a compile-time constant — zero
                    // runtime bytes beyond the constant push.
                    "alen" => {
                        let cap = match &args[0] {
                            Expr::Ident(name) => {
                                let id = *self.arrs.get(&(GLOBAL_ENT, name.clone()))
                                    .ok_or_else(|| crate::diag::internal(&format!("alen array {}", name)))?;
                                self.ctx.arrays[id].cap as i32
                            }
                            Expr::Field(inner, fname) => {
                                let v = match &**inner {
                                    Expr::Ident(v) => v.clone(),
                                    _ => return Err("internal: alen base".into()),
                                };
                                let ei = match env.locals.iter().find(|(n, _)| *n == v).map(|(_, vt)| *vt) {
                                    Some(VT::Ent(ei)) => ei,
                                    _ => return Err(format!("'{}' is not an entity variable", v)),
                                };
                                let id = *self.arrs.get(&(ei as u8, fname.clone()))
                                    .ok_or_else(|| crate::diag::internal(&format!("alen array {}", fname)))?;
                                self.ctx.arrays[id].cap as i32
                            }
                            _ => return Err("internal: alen arg".into()),
                        };
                        self.push_const(cap);
                        Ok(VT::Int)
                    }
                    other => Err(format!("internal: unknown call {}", other)),
                }
            }
        }
    }
}

fn env_set(env: &mut Env, name: &str, vt: VT) {
    if let Some(slot) = env.locals.iter_mut().find(|(n, _)| n == name) {
        slot.1 = vt;
    } else {
        env.locals.push((name.to_string(), vt));
    }
}


fn field_var_name(target: &Expr) -> String {
    if let Expr::Field(base, _) = target {
        if let Expr::Ident(v) = &**base { return v.clone(); }
    }
    String::new()
}
