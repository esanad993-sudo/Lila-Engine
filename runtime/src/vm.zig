//! LILA bytecode VM: decoded-instruction interpreter with i32 value stack,
//! frame stack, entity-iterator stack. Runs game logic at 60Hz.

const std = @import("std");
const c = @import("core.zig");
const ent = @import("entity.zig");
const render = @import("render.zig");
const render3d = @import("render3d.zig");
const audio = @import("audio.zig");

const STACK_SIZE: u32 = 512;
const MAX_FRAMES: u32 = 32;
// v12: 32 slots (was 16 — real articulated rigs need more scratch locals).
// The compiler enforces the same limit at build time with a loud error, so
// a game that outgrows the frame is rejected, never silently miscompiled
// (slots >= the limit would drop writes and hang loops).
const MAX_LOCALS: u32 = 32;
const MAX_ITERS: u32 = 8;
const NO_IDX: u32 = 0xFFFFFFFF;

const Frame = struct {
    ret_pc: u32,
    fn_idx: u32,
    locals: [MAX_LOCALS]i32,
};

const Iter = struct { typ: u32, next_bit: u32, slot: u32, mask: u32 = 0, base: u32 = 0 };

/// The entire interpreter state. On native builds this is THREADLOCAL so the
/// v10 parallel systems can run proven-disjoint update groups on concurrent
/// workers — each worker gets its own stack/frames/iterators automatically.
/// On wasm32-freestanding (single-threaded host) it is a plain static.
const State = struct {
    stack: [STACK_SIZE]i32 = undefined,
    sp: u32 = 0,
    frames: [MAX_FRAMES]Frame = undefined,
    fp: u32 = 0, // number of live frames
    pc: u32 = 0,
    iters: [MAX_ITERS]Iter = undefined,
    itr: u32 = 0, // number of live iterators
    halted: bool = false,
};

const IS_WASM = @import("builtin").cpu.arch.isWasm();
threadlocal var st_thread: State = .{};
var st_single: State = .{};

inline fn st() *State {
    if (IS_WASM) return &st_single;
    return &st_thread;
}

/// Relative jump with wrap-safe semantics: a 16-bit sign-extended offset
/// can push the target below 0 (crafted bytecode). The old @intCast
/// panicked in ReleaseSafe (and was UB in ReleaseFast); the hardened
/// contract wraps to a huge pc, which the step() prologue treats as
/// off-the-end -> implicit RET. Deterministic, never a trap — the same
/// soft-landing contract as every other operand mask.
inline fn jumpRel(next: u32, rel: i32) u32 {
    return next +% @as(u32, @bitCast(rel));
}

pub fn resetVm() void {
    st().sp = 0;
    st().fp = 0;
    st().itr = 0;
    st().halted = false;
}

// ---------------- v10: VM state snapshots ----------------
// The interpreter state is threadlocal on native builds, so parallel-system
// workers own their state by construction — no save/restore needed. This
// API remains for hosts/tools that want a full interpreter copy.

pub const VmSnapshot = State;

pub fn snapshot() State {
    return st().*;
}

pub fn restore(s: State) void {
    st().* = s;
}

inline fn push(v: i32) void {
    if (st().sp < STACK_SIZE) {
        st().stack[st().sp] = v;
        st().sp += 1;
    }
}
inline fn pop() i32 {
    if (st().sp > 0) {
        st().sp -= 1;
        return st().stack[st().sp];
    }
    return 0;
}

/// Call a function: args land in locals[0..n].
fn callFn(fi: u32, args: []const i32) bool {
    if (st().fp >= MAX_FRAMES) return false;
    if (st().sp + 4 > STACK_SIZE) return false;
    const f = &st().frames[st().fp];
    f.ret_pc = st().pc;
    f.fn_idx = fi;
    f.locals = [_]i32{0} ** MAX_LOCALS;
    for (args, 0..) |a, i| {
        if (i < MAX_LOCALS) f.locals[i] = a;
    }
    st().fp += 1;
    st().pc = c.fnStart(fi);
    return true;
}

/// Run a function to completion (used by engine for scheduled fns).
pub fn run(fi: u32) void {
    if (st().halted) return;
    const saved_pc = st().pc;
    if (!callFn(fi, &.{})) { st().pc = saved_pc; return; }
    // sentinel: mark this base frame by ret_pc = 0xFFFFFFFE
    st().frames[st().fp - 1].ret_pc = 0xFFFFFFFE;
    // execute until that frame is popped
    while (st().fp > 0 and !st().halted) {
        step();
    }
    st().pc = saved_pc;
}

/// Run an entity fn for one entity handle.
pub fn runEntity(fi: u32, handle: i32) void {
    if (st().halted) return;
    const saved_pc = st().pc;
    if (!callFn(fi, &[1]i32{handle})) { st().pc = saved_pc; return; }
    st().frames[st().fp - 1].ret_pc = 0xFFFFFFFE;
    while (st().fp > 0 and !st().halted) {
        step();
    }
    st().pc = saved_pc;
}

/// One decoded instruction.
fn step() void {
    comptime @setEvalBranchQuota(10000); // v11: the opcode switch grew past the 1000 default
    if (st().pc >= c.instr_count) {
        // ran off the end: implicit RET
        popFrame();
        return;
    }
    const op = c.iOp(st().pc);
    const a = c.iA(st().pc);
    const b = c.iB(st().pc);
    const cc = c.iC(st().pc);
    const next = st().pc + 1;

    switch (op) {
        c.OP_NOP => {},
        c.OP_PUSH_S8 => push(a),
        c.OP_PUSH_POOL => {
            const pi = @as(u32, @bitCast(a)) & 0xFFFF;
            push(@bitCast(c.rd32(c.ADDR_POOL + pi * 4)));
        },
        c.OP_LD_LOCAL => {
            const fr = &st().frames[st().fp - 1];
            const slot: u32 = @intCast(@as(u32, @bitCast(a)) & 0xFF);
            push(if (slot < MAX_LOCALS) fr.locals[slot] else 0);
        },
        c.OP_ST_LOCAL => {
            const fr = &st().frames[st().fp - 1];
            const slot: u32 = @as(u32, @bitCast(a)) & 0xFF;
            const v = pop();
            if (slot < MAX_LOCALS) fr.locals[slot] = v;
        },
        c.OP_LD_GLBL => push(ent.globalLoad(@intCast(@as(u32, @bitCast(a))))),
        c.OP_ST_GLBL => {
            const v = pop();
            ent.globalStore(@intCast(@as(u32, @bitCast(a))), v);
        },
        c.OP_LD_ENT => {
            const fr = &st().frames[st().fp - 1];
            const slot: u32 = @as(u32, @bitCast(a)) & 0xFF;
            const tf: u32 = @as(u32, @bitCast(b)) & 0xFFFF;
            const handle: i32 = if (slot < MAX_LOCALS) fr.locals[slot] else 0;
            push(ent.fieldLoad(tf, @bitCast(handle)));
        },
        c.OP_ST_ENT => {
            const fr = &st().frames[st().fp - 1];
            const slot: u32 = @as(u32, @bitCast(a)) & 0xFF;
            const tf: u32 = @as(u32, @bitCast(b)) & 0xFFFF;
            const v = pop();
            const handle: i32 = if (slot < MAX_LOCALS) fr.locals[slot] else 0;
            ent.fieldStore(tf, @bitCast(handle), v);
        },
        c.OP_ADD_F, c.OP_ADD_I => {
            const y = pop(); const x = pop();
            push(x +% y);
        },
        c.OP_SUB_F, c.OP_SUB_I => {
            const y = pop(); const x = pop();
            push(x -% y);
        },
        c.OP_MUL_F => {
            const y = pop(); const x = pop();
            push(c.mulF(x, y));
        },
        c.OP_MUL_I => {
            const y = pop(); const x = pop();
            push(x *% y);
        },
        c.OP_DIV_F => {
            const y = pop(); const x = pop();
            push(c.divF(x, y));
        },
        c.OP_DIV_I => {
            const y = pop(); const x = pop();
            if (y == 0) { push(0); } else { push(@divTrunc(x, y)); }
        },
        c.OP_MOD_I => {
            const y = pop(); const x = pop();
            if (y == 0) { push(0); } else { push(@rem(x, y)); }
        },
        c.OP_AND_I => { const y = pop(); const x = pop(); push(x & y); },
        c.OP_OR_I => { const y = pop(); const x = pop(); push(x | y); },
        c.OP_XOR_I => { const y = pop(); const x = pop(); push(x ^ y); },
        c.OP_SHL => {
            const y = pop(); const x = pop();
            const sh: u5 = @intCast(@as(u32, @bitCast(y)) & 31);
            push(x << sh);
        },
        c.OP_SHR => {
            const y = pop(); const x = pop();
            const sh: u5 = @intCast(@as(u32, @bitCast(y)) & 31);
            push(x >> sh);
        },
        c.OP_AND_B => { const y = pop(); const x = pop(); push(@intFromBool((x != 0) and (y != 0))); },
        c.OP_OR_B => { const y = pop(); const x = pop(); push(@intFromBool((x != 0) or (y != 0))); },
        c.OP_NOT_B => { const x = pop(); push(@intFromBool(x == 0)); },
        c.OP_NEG => { const x = pop(); push(-%x); },
        c.OP_EQ_F, c.OP_EQ_I => { const y = pop(); const x = pop(); push(@intFromBool(x == y)); },
        c.OP_NE_F, c.OP_NE_I => { const y = pop(); const x = pop(); push(@intFromBool(x != y)); },
        c.OP_LT_F, c.OP_LT_I => { const y = pop(); const x = pop(); push(@intFromBool(x < y)); },
        c.OP_GT_F, c.OP_GT_I => { const y = pop(); const x = pop(); push(@intFromBool(x > y)); },
        c.OP_LE_F, c.OP_LE_I => { const y = pop(); const x = pop(); push(@intFromBool(x <= y)); },
        c.OP_GE_F, c.OP_GE_I => { const y = pop(); const x = pop(); push(@intFromBool(x >= y)); },
        c.OP_SIN => { const x = pop(); push(c.sinA(x)); },
        c.OP_COS => { const x = pop(); push(c.cosA(x)); },
        c.OP_RAND_MAX => { const m = pop(); push(c.randMax(m)); },
        c.OP_ANIM => {
            const t = pop();
            push(evalAnim(@as(u32, @bitCast(a)) & 0xFFFF, t));
        },
        c.OP_KEY => {
            const input = c.rd32(c.ADDR_INPUT);
            const k: u5 = @intCast(@as(u32, @bitCast(a)) & 31);
            push(@intFromBool((input >> k) & 1 != 0));
        },
        c.OP_JMP => { st().pc = jumpRel(next, a); return; },
        c.OP_JZ => {
            const v = pop();
            if (v == 0) { st().pc = jumpRel(next, a); return; }
        },
        c.OP_JNZ => {
            const v = pop();
            if (v != 0) { st().pc = jumpRel(next, a); return; }
        },
        c.OP_FOR_BGN => {
            const t: u32 = @as(u32, @bitCast(a)) & 0xFF;
            const slot: u32 = @as(u32, @bitCast(b)) & 0xFF;
            if (st().itr >= MAX_ITERS) {
                st().pc = jumpRel(next, cc);
                return;
            }
            const idx = ent.nextLive(t, 0);
            if (idx == NO_IDX) {
                st().pc = jumpRel(next, cc);
                return;
            }
            st().iters[st().itr] = .{ .typ = t, .next_bit = idx + 1, .slot = slot };
            st().itr += 1;
            const fr = &st().frames[st().fp - 1];
            if (slot < MAX_LOCALS) fr.locals[slot] = @bitCast((@as(u32, ent.genOf(t, idx)) << 20) | idx);
        },
        c.OP_FOR_ADV => {
            if (st().itr == 0) { st().pc = jumpRel(next, a); return; }
            const it = &st().iters[st().itr - 1];
            const idx = ent.nextLive(it.typ, it.next_bit);
            if (idx == NO_IDX) {
                st().itr -= 1; // loop done: fall through (exit)
            } else {
                it.next_bit = idx + 1;
                // find the FOR_BGN's slot operand: stored in the b of the loop head.
                // We re-derive: the local slot was set by FOR_BGN; keep a parallel
                // slot record in the iterator.
                const fr = &st().frames[st().fp - 1];
                if (it.slot < MAX_LOCALS) {
                    fr.locals[it.slot] = @bitCast((@as(u32, ent.genOf(it.typ, idx)) << 20) | idx);
                }
                st().pc = jumpRel(next, a);
                return;
            }
        },
        c.OP_SPAWN => {
            const t: u32 = @as(u32, @bitCast(a)) & 0xFF;
            const n: u32 = @as(u32, @bitCast(b)) & 0xFF;
            const handle = ent.spawn(t);
            if (handle != NO_IDX) {
                ent.applyDefaults(t, handle);
                // extra tf list at ADDR_EXTRA + extra_off*2
                const ex = c.ADDR_EXTRA + c.iExtra(st().pc) * 2;
                var j: u32 = 0;
                while (j < n) : (j += 1) {
                    const tf = c.rd16(ex + j * 2);
                    // v14: a crafted arg count could read below the stack
                    // base (u32 wrap -> OOB panic); guard reads that need it
                    const v: i32 = if (st().sp >= n) st().stack[st().sp - n + j] else 0;
                    ent.fieldStore(tf, handle, v);
                }
                st().sp -|= n;
            } else {
                st().sp -|= n;
            }
        },
        c.OP_KILL => {
            const fr = &st().frames[st().fp - 1];
            const slot: u32 = @as(u32, @bitCast(a)) & 0xFF;
            const t: u32 = @as(u32, @bitCast(b)) & 0xFF;
            const handle: i32 = if (slot < MAX_LOCALS) fr.locals[slot] else 0;
            const h: u32 = @bitCast(handle);
            ent.kill(t, h & 0xFFFFF);
        },
        c.OP_SFX => audio.triggerSfx(@as(u32, @bitCast(a)) & 0xFFFF),
        c.OP_MUSIC => audio.startMusic(@as(u32, @bitCast(a)) & 0xFFFF),
        c.OP_STOP_MUSIC => audio.stopMusic(),
        c.OP_SHAKE => {
            // bump amplitude (overlapping shakes take the max, never cancel)
            const amp: u32 = @as(u32, @bitCast(a)) & 0xFF;
            if (amp > c.rd32(c.ADDR_SHAKE)) c.wr32(c.ADDR_SHAKE, amp);
        },
        // ---------------- v7: camera / persistence / aim math / music vol ----------------
        c.OP_CAMERA => {
            // pop y, x (fixed) — clamp so the viewport stays inside the world
            const y = pop();
            const x = pop();
            const cl = c.clampCam(x, y);
            c.wr32(c.ADDR_CAM_X, @bitCast(cl.x));
            c.wr32(c.ADDR_CAM_Y, @bitCast(cl.y));
        },
        c.OP_MUSIC_VOL => audio.setMusicVol(@as(u32, @bitCast(a)) & 0xFF),
        c.OP_SAVE => {
            // pop value, then slot index — raw 32-bit store, host persists
            const v = pop();
            const ix = pop();
            const slot: u32 = @as(u32, @bitCast(ix)) % c.SRAM_SLOTS;
            c.wr32(c.ADDR_SRAM + slot * 4, @bitCast(v));
            c.sram_dirty = 1;
        },
        c.OP_SAVED => {
            const ix = pop();
            const slot: u32 = @as(u32, @bitCast(ix)) % c.SRAM_SLOTS;
            push(@bitCast(c.rd32(c.ADDR_SRAM + slot * 4)));
        },
        c.OP_DIST => {
            // st().stack: x1 y1 x2 y2 (fixed) -> pop y2 x2 y1 x1
            const y2 = pop();
            const x2 = pop();
            const y1 = pop();
            const x1 = pop();
            push(c.distF(x1, y1, x2, y2));
        },
        c.OP_ATAN2 => {
            // st().stack: dy dx (fixed) -> pop dx then dy
            const dx = pop();
            const dy = pop();
            push(c.atan2ang(dy, dx));
        },
        c.OP_CAM_X => push(@bitCast(c.rd32(c.ADDR_CAM_X))),
        c.OP_CAM_Y => push(@bitCast(c.rd32(c.ADDR_CAM_Y))),
        // ---------------- v8: optimizer-baked physics / AI ----------------
        c.OP_SWEPT => {
            // swept Minkowski interval test (subsystem 3): A moves by
            // (avx, avy) over the frame step, B static. Relative position
            // per axis is linear in t in [0,1]; the axis is hit iff the
            // segment CROSSES zero (r0·r1 < 0) or an endpoint already sits
            // inside (min(|r0|,|r1|) < h) — exact continuous 1D solve,
            // unit-tested against sub-step sampling in opt::physics.
            const hh = pop();
            const hw = pop();
            const by = pop();
            const bx = pop();
            const avy = pop();
            const avx = pop();
            const ay = pop();
            const ax = pop();
            const dx: i64 = @as(i64, bx) - @as(i64, ax);
            const dy: i64 = @as(i64, by) - @as(i64, ay);
            const r1x: i64 = dx - @as(i64, avx);
            const r1y: i64 = dy - @as(i64, avy);
            const hitx = (dx * r1x < 0) or (@min(@abs(dx), @abs(r1x)) < @as(i64, hw));
            const hity = (dy * r1y < 0) or (@min(@abs(dy), @abs(r1y)) < @as(i64, hh));
            push(@intFromBool(hitx and hity));
        },
        c.OP_COLLIDE_MASK => {
            // Galois 32-lane collision mask (subsystem 3): ONE pass over a
            // 32-slot WINDOW of the type, per-lane Chebyshev box test in pure
            // ALU — the resulting immediate bitmask feeds a ctz candidate
            // scan (OP_FOR_MASK_BGN). The exact dist() test still runs on
            // candidates; the mask is a provable SUPERSET of the disk.
            // v9 multi-window: a = type | (window_base/32 << 8); windows
            // tile capacities up to 128 slots (4 windows).
            const cy = pop();
            const cx = pop();
            const au = @as(u32, @bitCast(a));
            const t: u32 = au & 0xFF;
            const base: u32 = ((au >> 8) & 0xFF) * 32;
            const hw: i32 = @bitCast(c.rd32(c.ADDR_POOL + (@as(u32, @bitCast(b)) & 0xFFFF) * 4));
            const hh: i32 = @bitCast(c.rd32(c.ADDR_POOL + (@as(u32, @bitCast(cc)) & 0xFFFF) * 4));
            const ex = c.ADDR_EXTRA + c.iExtra(st().pc) * 2;
            const x_tf: u32 = c.rd16(ex);
            const y_tf: u32 = c.rd16(ex + 2);
            var mask: u32 = 0;
            const top = @min(base + 32, c.maxLive(t));
            var idx: u32 = base;
            while (idx < top) : (idx += 1) {
                if (!ent.isAlive(t, idx)) continue;
                const h: u32 = (@as(u32, ent.genOf(t, idx)) << 20) | idx;
                const x = ent.fieldLoad(x_tf, h);
                const y = ent.fieldLoad(y_tf, h);
                const dx = x -% cx;
                const dy = y -% cy;
                // strict |dx| < hw && |dy| < hh — the SAME L-inf guard the
                // compiler proves sound against the retained exact test
                if (dx < hw and dx > -%hw and dy < hh and dy > -%hh) {
                    mask |= @as(u32, 1) << @intCast(idx - base);
                }
            }
            push(@bitCast(mask));
        },
        c.OP_FOR_MASK_BGN => {
            // ctz-scan entry over an immediate lane mask: pop mask; first set
            // bit = the first candidate slot. mask == 0 -> exit (c is rel).
            // v9: b = type | (window_base/32 << 8) — handles offset by base.
            const slot: u32 = @as(u32, @bitCast(a)) & 0xFF;
            const bu = @as(u32, @bitCast(b));
            const t: u32 = bu & 0xFF;
            const base: u32 = ((bu >> 8) & 0xFF) * 32;
            const mask: u32 = @bitCast(pop());
            if (st().itr >= MAX_ITERS or mask == 0) {
                st().pc = jumpRel(next, cc);
                return;
            }
            const idx: u32 = base + @ctz(mask);
            st().iters[st().itr] = .{ .typ = t, .next_bit = 0, .slot = slot, .mask = mask & (mask - 1), .base = base };
            st().itr += 1;
            const fr = &st().frames[st().fp - 1];
            if (slot < MAX_LOCALS) fr.locals[slot] = @bitCast((@as(u32, ent.genOf(t, idx)) << 20) | idx);
        },
        c.OP_FOR_MASK_ADV => {
            // next candidate: lowest remaining set bit; jump back to the body
            // while any remain, otherwise pop the iterator and fall through.
            if (st().itr == 0) { st().pc = jumpRel(next, a); return; }
            const it = &st().iters[st().itr - 1];
            if (it.mask == 0) {
                st().itr -= 1; // scan complete: exit the loop
            } else {
                const idx: u32 = it.base + @ctz(it.mask);
                it.mask &= it.mask - 1;
                const fr = &st().frames[st().fp - 1];
                if (it.slot < MAX_LOCALS) {
                    fr.locals[it.slot] = @bitCast((@as(u32, ent.genOf(it.typ, idx)) << 20) | idx);
                }
                st().pc = jumpRel(next, a);
                return;
            }
        },
        c.OP_FSM_NEXT => {
            // bit-plane transition (subsystem 5): next = T[s * span + em].
            // Guard priority was PRECOMPUTED into the table by the compiler;
            // the runtime is one shift + mask + load. Stack: em pushed first,
            // s second -> pop s then em.
            const s = pop();
            const em = pop();
            const ti: u32 = @as(u32, @bitCast(a)) & 0xFFFF;
            const base = c.ADDR_FSM + ti * c.FSM_STRIDE;
            const nstates: u32 = c.mem[base + 2];
            const span: u32 = c.mem[base + 3];
            if (nstates == 0 or span == 0) { push(0); } // no table loaded: stay
            else {
                const su: u32 = @as(u32, @bitCast(s)) % nstates;
                const eu: u32 = @as(u32, @bitCast(em)) & (span - 1);
                push(c.mem[base + 4 + su * span + eu]);
            }
        },
        c.OP_SEL => {
            // branchless conditional move (subsystem 6): a DATA edge, not a
            // control edge — the PC never jumps, execution time is constant
            // whatever the condition. Stack: cond, then, else pushed in that
            // order -> pop else, then, cond. Values are raw i32 payloads, so
            // bool/int/fixed all select bit-exactly.
            const elsev = pop();
            const thenv = pop();
            const cond = pop();
            push(if (cond != 0) thenv else elsev);
        },
        // ---------------- v11: arrays ----------------
        c.OP_LD_GARR => {
            // a = array id; pop idx (wrapped by mask|mod), push the raw cell
            const id: u32 = @as(u32, @bitCast(a)) & 0xFF;
            const idx = pop();
            if (id >= c.n_arrs or c.arr_ent[id] != 0xFF) { push(0); }
            else {
                const i = c.arrIndex(id, idx);
                push(@bitCast(c.rd32(c.garrElem(id, i))));
            }
        },
        c.OP_ST_GARR => {
            const id: u32 = @as(u32, @bitCast(a)) & 0xFF;
            // stack: idx pushed, then val -> top is val
            const v = pop();
            const idx = pop();
            if (id < c.n_arrs and c.arr_ent[id] == 0xFF) {
                const i = c.arrIndex(id, idx);
                c.wr32(c.garrElem(id, i), @bitCast(v));
            }
        },
        c.OP_LD_EARR => {
            // a = local slot holding the entity handle, b = array id; pop idx.
            // Stale handles read 0 (same contract as fieldLoad).
            const slot: u32 = @as(u32, @bitCast(a)) & 0xFF;
            const id: u32 = @as(u32, @bitCast(b)) & 0xFF;
            const idx = pop();
            const fr = &st().frames[st().fp - 1];
            const handle: u32 = @bitCast(if (slot < MAX_LOCALS) fr.locals[slot] else 0);
            if (id >= c.n_arrs or c.arr_ent[id] == 0xFF) { push(0); }
            else {
                const t: u32 = c.arr_ent[id];
                const sidx = handle & 0xFFFFF;
                const gen = (handle >> 20) & 0xFFF;
                if (sidx >= c.maxLive(t) or ent.genOf(t, sidx) != gen or !ent.isAlive(t, sidx)) {
                    push(0);
                } else {
                    const i = c.arrIndex(id, idx);
                    push(@bitCast(c.rd32(c.earrRow(id, sidx) + i * 4)));
                }
            }
        },
        c.OP_ST_EARR => {
            const slot: u32 = @as(u32, @bitCast(a)) & 0xFF;
            const id: u32 = @as(u32, @bitCast(b)) & 0xFF;
            // stack: idx pushed, then val -> top is val
            const v = pop();
            const idx = pop();
            if (id < c.n_arrs and c.arr_ent[id] != 0xFF) {
                const t: u32 = c.arr_ent[id];
                const fr = &st().frames[st().fp - 1];
                const handle: u32 = @bitCast(if (slot < MAX_LOCALS) fr.locals[slot] else 0);
                const sidx = handle & 0xFFFFF;
                const gen = (handle >> 20) & 0xFFF;
                if (sidx < c.maxLive(t) and ent.genOf(t, sidx) == gen and ent.isAlive(t, sidx)) {
                    const i = c.arrIndex(id, idx);
                    c.wr32(c.earrRow(id, sidx) + i * 4, @bitCast(v));
                }
            }
        },
        c.OP_DUP => {
            // duplicate top of stack (compound array assigns keep the index
            // under the element without re-evaluating it)
            if (st().sp > 0 and st().sp < STACK_SIZE) {
                st().stack[st().sp] = st().stack[st().sp - 1];
                st().sp += 1;
            }
        },
        // ---------------- v11: 3D camera + projection ----------------
        c.OP_CAM3 => {
            // pop pitch, yaw, z, y, x -> page-0 registers (snapshot-safe)
            const pitch = pop();
            const yaw = pop();
            const z = pop();
            const y = pop();
            const x = pop();
            c.wr32(c.ADDR_CAM3_X, @bitCast(x));
            c.wr32(c.ADDR_CAM3_Y, @bitCast(y));
            c.wr32(c.ADDR_CAM3_Z, @bitCast(z));
            c.wr32(c.ADDR_CAM3_YAW, @bitCast(yaw));
            c.wr32(c.ADDR_CAM3_PITCH, @bitCast(pitch));
        },
        c.OP_PROJ3 => {
            const z = pop();
            const y = pop();
            const x = pop();
            const p = c.proj3(x, y, z);
            c.wr32(c.ADDR_PROJ_X, @bitCast(p.sx));
            c.wr32(c.ADDR_PROJ_Y, @bitCast(p.sy));
            c.wr32(c.ADDR_PROJ_OK, @intFromBool(p.ok));
            push(p.scale);
        },
        c.OP_PROJ_X => push(@bitCast(c.rd32(c.ADDR_PROJ_X))),
        c.OP_PROJ_Y => push(@bitCast(c.rd32(c.ADDR_PROJ_Y))),
        c.OP_PROJ_OK => push(@intFromBool(c.rd32(c.ADDR_PROJ_OK) != 0)),
        c.OP_DRAW3D => {
            // a = global array id (mesh); stack: n, x, y, z, yaw, rgba pushed
            // in that order -> pop rgba, yaw, z, y, x, n. render3d owns the
            // whole transform/view/project/cull/sort/raster pipeline.
            const rgba = pop();
            const yaw = pop();
            const z = pop();
            const y = pop();
            const x = pop();
            const n = pop();
            const id: u32 = @as(u32, @bitCast(a)) & 0xFF;
            render3d.draw3d(id, n, x, y, z, yaw, @bitCast(rgba));
        },
        // ---------------- v12: articulated 3D (mat4/quat over arrays) ----
        c.OP_QUAT_AA => {
            // stack: q_aid, off, ax, ay, az, ang pushed in source order
            const ang = pop();
            const az = pop();
            const ay = pop();
            const ax = pop();
            const off = pop();
            const qid: u32 = @bitCast(pop());
            render3d.quatAA(qid, off, ax, ay, az, ang);
        },
        c.OP_Q_MUL => {
            // stack: d_aid, doff, a_aid, aoff, b_aid, boff
            const boff = pop();
            const bid: u32 = @bitCast(pop());
            const aoff = pop();
            const aid: u32 = @bitCast(pop());
            const doff = pop();
            const did: u32 = @bitCast(pop());
            render3d.qMul(did, doff, aid, aoff, bid, boff);
        },
        c.OP_M4_QT => {
            // stack: m_aid, moff, q_aid, qoff, tx, ty, tz
            const tz = pop();
            const ty = pop();
            const tx = pop();
            const qoff = pop();
            const qid: u32 = @bitCast(pop());
            const moff = pop();
            const mid: u32 = @bitCast(pop());
            render3d.m4QT(mid, moff, qid, qoff, tx, ty, tz);
        },
        c.OP_M4_MUL => {
            // stack: d_aid, doff, a_aid, aoff, b_aid, boff (D = A*B)
            const boff = pop();
            const bid: u32 = @bitCast(pop());
            const aoff = pop();
            const aid: u32 = @bitCast(pop());
            const doff = pop();
            const did: u32 = @bitCast(pop());
            render3d.m4Mul(did, doff, aid, aoff, bid, boff);
        },
        c.OP_SKIN3 => {
            // stack: v_aid, voff, m_aid, moff, x, y, z -> verts[voff..+2]
            const z = pop();
            const y = pop();
            const x = pop();
            const moff = pop();
            const mid: u32 = @bitCast(pop());
            const voff = pop();
            const vid: u32 = @bitCast(pop());
            render3d.skin3(vid, voff, mid, moff, x, y, z);
        },
        c.OP_DRAW3DI => {
            // a = vertex buffer id, b = index buffer id; stack: nv, ni, x,
            // y, z, yaw, rgba -> pop rgba, yaw, z, y, x, ni, nv.
            const rgba = pop();
            const yaw = pop();
            const z = pop();
            const y = pop();
            const x = pop();
            const ni = pop();
            const nv = pop();
            const vid: u32 = @as(u32, @bitCast(a)) & 0xFF;
            const iid: u32 = @as(u32, @bitCast(b)) & 0xFF;
            render3d.draw3di(vid, iid, nv, ni, x, y, z, yaw, @bitCast(rgba));
        },
        c.OP_GOTO => {
            const scene: u32 = @as(u32, @bitCast(a)) & 0xFFFF;
            c.wr32(c.ADDR_SCENE, scene + 1);
            if (scene < c.n_scenes) {
                const fi = c.rd16(c.ADDR_SCENES + scene * 2);
                st().pc = next;
                _ = callFn(fi, &.{});
                return;
            }
        },
        c.OP_DRAW => {
            const rgba = pop();
            const scale = pop();
            const rot = pop();
            const y = pop();
            const x = pop();
            render.drawSprite(@as(u32, @bitCast(a)) & 0xFFFF, x, y, rot, scale, @bitCast(rgba));
        },
        c.OP_DRAW_TEXT => {
            const rgba = pop();
            const y = pop();
            const x = pop();
            render.drawText(@as(u32, @bitCast(a)) & 0xFFFF, x, y, @bitCast(rgba));
        },
        c.OP_DRAW_NUM => {
            const rgba = pop();
            const y = pop();
            const x = pop();
            const val = pop();
            render.drawNum(val, x, y, @bitCast(rgba));
        },
        c.OP_CALL_TBL => {
            const fr = &st().frames[st().fp - 1];
            const table: u32 = @as(u32, @bitCast(a)) & 0xFF;
            const slot: u32 = @as(u32, @bitCast(b)) & 0xFF;
            const key = pop();
            const ku: u32 = @as(u32, @bitCast(key)) & 0xFF;
            const base = c.ADDR_TABLES + table * c.TABLE_STRIDE;
            const fi = c.rd16(base + ku * 2);
            const handle: i32 = if (slot < MAX_LOCALS) fr.locals[slot] else 0;
            st().pc = next;
            _ = callFn(fi, &[1]i32{handle});
            return;
        },
        c.OP_COUNT => {
            const t: u32 = @as(u32, @bitCast(a)) & 0xFF;
            push(@intCast(ent.count(t)));
        },
        c.OP_RET => {
            popFrame();
            return; // st().pc set by popFrame — must NOT fall through to st().pc = next
        },
        c.OP_CALL_FN => {
            const fi: u32 = @as(u32, @bitCast(a)) & 0xFFFF;
            const n: u32 = @as(u32, @bitCast(b)) & 0xFF;
            // args: last n st().stack values in source order
            // v14: guard the underflow a crafted count could cause (the
            // old u32 wrap indexed ~4 billion -> OOB panic)
            var args: [4]i32 = .{ 0, 0, 0, 0 };
            var j: u32 = 0;
            while (j < n and j < 4) : (j += 1) {
                args[j] = if (st().sp >= n) st().stack[st().sp - n + j] else 0;
            }
            st().sp -|= n;
            st().pc = next;
            _ = callFn(fi, args[0..@min(n, 4)]);
            return;
        },
        c.OP_I2F => { const x = pop(); push(x *% 256); },
        c.OP_F2I => { const x = pop(); push(x >> 8); },
        else => {},
    }
    st().pc = next;
}

fn popFrame() void {
    if (st().fp == 0) {
        st().halted = true;
        return;
    }
    st().fp -= 1;
    const ret = st().frames[st().fp].ret_pc;
    if (ret == 0xFFFFFFFE) {
        st().halted = false; // base frame — caller loop stops on st().fp==0
        return;
    }
    st().pc = ret;
}

/// Cubic Hermite evaluation of an anim curve at integer frame t (wraps at
/// duration). All math in Q24.8 with i64 intermediates. O(#segs) segment scan.
pub fn evalAnim(curve: u32, t: i32) i32 {
    if (curve >= c.n_anims) return 0;
    const base = c.ADDR_ANIM + curve * c.ANIM_STRIDE;
    const dur: u32 = c.rd16(base);
    const nseg: u32 = c.mem[base + 2];
    if (nseg == 0 or dur == 0) return @bitCast(c.rd32(base + 4 + 2));
    const tt: u32 = @as(u32, @bitCast(t)) % dur;
    // find segment
    var s: u32 = 0;
    while (s < nseg) : (s += 1) {
        const off = base + 4 + s * 20;
        const t0: u32 = c.rd16(off);
        const t1: u32 = if (s + 1 < nseg) c.rd16(base + 4 + (s + 1) * 20) else dur;
        if (tt >= t0 and tt < t1) {
            const p0: i32 = @bitCast(c.rd32(off + 2));
            const m0: i32 = @bitCast(c.rd32(off + 6));
            const p1: i32 = @bitCast(c.rd32(off + 10));
            const m1: i32 = @bitCast(c.rd32(off + 14));
            const dt: i32 = @intCast(t1 - t0);
            // s in Q8: 0..256
            const sq8: i32 = @intCast(@divTrunc(@as(i64, @as(i32, @intCast(tt - t0))) << 8, @as(i64, dt)));
            const s2: i32 = @intCast(@divTrunc(@as(i64, sq8) * sq8, 256));
            const s3: i32 = @intCast(@divTrunc(@as(i64, s2) * sq8, 256));
            const h00: i32 = 256 + 2 * s3 - 3 * s2;
            const h01: i32 = 256 - h00;
            const h10: i32 = s3 - 2 * s2 + sq8;
            const h11: i32 = s3 - s2;
            const v: i64 = @as(i64, h00) * p0 + @as(i64, h10) * dt * m0 +
                @as(i64, h01) * p1 + @as(i64, h11) * dt * m1;
            return @intCast(@divTrunc(v, 256));
        }
    }
    // t == dur edge: clamp to last segment end
    const off = base + 4 + (nseg - 1) * 20;
    return @bitCast(c.rd32(off + 10));
}
