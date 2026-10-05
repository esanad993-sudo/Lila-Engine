//! LILA engine orchestration: init + 60Hz frame scheduling
//! (update0 -> per-entity updates -> draw0 -> per-entity draws).

const std = @import("std");
const c = @import("core.zig");
const loader = @import("loader.zig");
const vm = @import("vm.zig");
const ent = @import("entity.zig");
const render = @import("render.zig");
const audio = @import("audio.zig");

// ---------------- optional per-system profiler ----------------
// Unreal Insights-lite: when prof_on, every scheduled fn and the auto-
// integration pass are wall-clock timed. Overhead when off: one branch
// per call site; when on: ~20 nanoTimestamp calls per frame (game logic
// itself is unmodified, so profiled frames stay bit-identical).
pub var prof_on: bool = false;
pub var prof_fn_ns: [c.MAX_FNS]u64 = [_]u64{0} ** c.MAX_FNS;
pub var prof_fn_calls: [c.MAX_FNS]u64 = [_]u64{0} ** c.MAX_FNS;
pub var prof_integ_ns: u64 = 0;
pub var prof_integ_calls: u64 = 0;

pub inline fn nowNs() u64 {
    // wasm32-freestanding has no clock: the profiler degrades to call
    // counts there (ns columns read 0). Native keeps wall-clock precision.
    if (@import("builtin").cpu.arch.isWasm()) return 0;
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

inline fn profEnter() u64 {
    return if (prof_on) nowNs() else 0;
}
inline fn profLeave(fi: u32, t0: u64) void {
    if (prof_on) {
        prof_fn_ns[fi] +%= nowNs() -| t0;
        prof_fn_calls[fi] +%= 1;
    }
}

pub fn init(blob: []const u8) i32 {
    // v7: SRAM survives re-init (hi-scores persist across in-game restarts
    // — the "battery-backed" region). Everything else resets to zero.
    @memset(c.mem[0..c.ADDR_SRAM], 0);
    @memset(c.mem[c.ADDR_SRAM + c.SRAM_BYTES ..], 0);
    const rc = loader.load(blob);
    if (rc != c.ERR_OK) return rc;
    vm.resetVm();
    render.streamReset();
    audio.reset();
    var i: u32 = 0;
    while (i < c.n_fns) : (i += 1) {
        if (c.fnKind(i) == c.FN_INIT) vm.run(i);
    }
    // v10: persistent parallel-system workers (no-op without groups / wasm)
    Pool.start();
    return c.ERR_OK;
}

pub fn frame() void {
    render.streamReset();
    render.emitStarfield(); // engine background layer — behind all game draws

    var i: u32 = 0;
    while (i < c.n_fns) : (i += 1) {
        if (c.fnKind(i) == c.FN_UPDATE0) {
            const t0 = profEnter();
            vm.run(i);
            profLeave(i, t0);
        }
    }

    // ---------------- v10: per-entity updates ----------------
    // Without groups (or with threading disabled) this is the classic
    // sequential sweep. With compile-proven parallel groups (subsystem 10)
    // each disjoint group runs its members on concurrent workers: every
    // worker snapshots the interpreter state, runs its system's per-entity
    // loop, restores. The members touch disjoint field planes by PROOF, so
    // the merged result is bit-identical to any sequential order.
    const use_par = threads_enabled and c.n_par_groups > 0 and !@import("builtin").cpu.arch.isWasm();
    var done: [c.MAX_FNS]bool = [_]bool{false} ** c.MAX_FNS;
    var t: u32 = 0;
    while (t < c.n_entities) : (t += 1) {
        i = 0;
        while (i < c.n_fns) : (i += 1) {
            if (c.fnKind(i) == c.FN_UPDATEENT and c.fnEntity(i) == t) {
                if (done[i]) continue;
                if (use_par and runGroupParallel(i, &done)) {
                    // group executed (possibly concurrently)
                } else {
                    const t0 = profEnter();
                    runPerEntity(i, t);
                    profLeave(i, t0);
                    done[i] = true;
                }
                break; // next type
            }
        }
    }

    // auto-integration pass: pos = (pos + vel) & wrap_mask — 2 ops/axis,
    // emitted per-entity by the compiler (context-driven physics)
    const t_integ: u64 = if (prof_on) nowNs() else 0;
    t = 0;
    while (t < c.n_entities) : (t += 1) {
        const n_integ = c.integCount(t);
        if (n_integ == 0) continue;
        var idx = ent.nextLive(t, 0);
        while (idx != 0xFFFFFFFF) {
            const handle = (@as(u32, ent.genOf(t, idx)) << 20) | idx;
            var k: u32 = 0;
            while (k < n_integ) : (k += 1) {
                const pos_tf = c.integPos(t, k);
                const vel_tf = c.integVel(t, k);
                const mask = c.integMask(t, k);
                var p = ent.fieldLoad(pos_tf, handle);
                const v = ent.fieldLoad(vel_tf, handle);
                p +%= v;
                if (mask != 0) p &= mask;
                ent.fieldStore(pos_tf, handle, p);
            }
            idx = ent.nextLive(t, idx + 1);
        }
    }
    if (prof_on) {
        prof_integ_ns +%= nowNs() -| t_integ;
        prof_integ_calls +%= 1;
    }

    i = 0;
    while (i < c.n_fns) : (i += 1) {
        if (c.fnKind(i) == c.FN_DRAW0) {
            const t0 = profEnter();
            vm.run(i);
            profLeave(i, t0);
        }
    }

    t = 0;
    while (t < c.n_entities) : (t += 1) {
        i = 0;
        while (i < c.n_fns) : (i += 1) {
            if (c.fnKind(i) == c.FN_DRAWENT and c.fnEntity(i) == t) {
                const t0 = profEnter();
                runPerEntity(i, t);
                profLeave(i, t0);
            }
        }
    }

    // decay screen-shake (bumped by the SHAKE opcode during updates)
    const amp = c.rd32(c.ADDR_SHAKE);
    if (amp > 0) {
        const na = (amp * 3) / 4;
        c.wr32(c.ADDR_SHAKE, if (na < 2) 0 else na);
    }
    c.wr32(c.ADDR_FRAME, c.rd32(c.ADDR_FRAME) + 1);
}

fn runPerEntity(fi: u32, t: u32) void {
    var idx = ent.nextLive(t, 0);
    while (idx != 0xFFFFFFFF) {
        const handle = (@as(u32, ent.genOf(t, idx)) << 20) | idx;
        vm.runEntity(fi, @bitCast(handle));
        idx = ent.nextLive(t, idx + 1);
    }
}

// ---------------- v10: compile-proven parallel group execution ----------------

pub var threads_enabled: bool = true;

const builtin = @import("builtin");
const HAS_THREADS = !builtin.cpu.arch.isWasm();

/// Worker entry: each thread owns its interpreter state (threadlocal VM),
/// so a worker just resets and runs its system's per-entity loop. The
/// system was PROVEN (at build time) to touch only its own type's field
/// planes — no spawns, no engine registers — so concurrent execution
/// merges bit-identically with any sequential order.
const Job = struct {
    fi: u32,
    t: u32,
};

// ---- persistent worker pool (spawn ONCE, park on a condvar) ----
// Spawning a thread per frame costs ~100-400us/frame on some hosts — far
// more than the systems themselves. The pool pays the spawn cost once per
// PROCESS; per frame the main thread posts the group's jobs and workers
// steal them off an atomic cursor. Determinism is untouched: workers only
// execute proven-disjoint systems, so the merged memory is bit-identical
// whatever order the lanes land in.
const Pool = struct {
    // Linux futex sync primitives (Zig 0.16 dropped std.Thread.Mutex):
    //   epoch futex  — workers park until the main thread posts a batch
    //   done futex   — the last worker to finish wakes the main thread
    // All ordering is .release/.acquire through the atomics themselves.
    const linux = @import("std").os.linux;

    inline fn futexWait(addr: *std.atomic.Value(u32), expected: u32) void {
        const op = linux.FUTEX_OP{ .cmd = .WAIT, .private = true };
        _ = linux.futex_4arg(addr, op, expected, null);
        // EAGAIN (value changed) just means: re-check the loop condition
    }

    inline fn futexWakeAll(addr: *std.atomic.Value(u32)) void {
        const op = linux.FUTEX_OP{ .cmd = .WAKE, .private = true };
        _ = linux.futex_3arg(addr, op, c.PAR_MAX_MEMBERS);
    }

    inline fn futexWakeOne(addr: *std.atomic.Value(u32)) void {
        const op = linux.FUTEX_OP{ .cmd = .WAKE, .private = true };
        _ = linux.futex_3arg(addr, op, 1);
    }

    var jobs: [c.PAR_MAX_MEMBERS]Job = undefined;
    var n_jobs: u32 = 0;
    var cursor: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
    var remaining: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
    var epoch: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
    var done_flag: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
    var shutdown_flag: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
    var started: bool = false;
    var n_workers: u32 = 0;

    fn workerLoop() void {
        var my_epoch: u32 = epoch.load(.acquire);
        while (shutdown_flag.load(.acquire) == 0) {
            const cur = epoch.load(.acquire);
            if (cur == my_epoch) {
                futexWait(&epoch, cur); // parked — no work this frame
                continue;
            }
            my_epoch = cur;
            // steal jobs (fresh threadlocal VM per system) — ONE decrement
            // per stolen job, exactly mirroring the main thread's loop, so
            // `remaining` counts UNFINISHED jobs and the drain is exact
            while (true) {
                const j = cursor.fetchAdd(1, .monotonic);
                if (j >= n_jobs) break;
                vm.resetVm();
                runPerEntity(jobs[j].fi, jobs[j].t);
                if (remaining.fetchSub(1, .acq_rel) == 1) {
                    done_flag.store(1, .release);
                    futexWakeOne(&done_flag);
                }
            }
        }
    }

    fn start() void {
        if (started or !HAS_THREADS) return;
        const cpus = @import("std").Thread.getCpuCount() catch 1;
        // always at least ONE worker: containers often misreport cpu count
        // (getCpuCount failing -> 1), and --threads 1 is the explicit off switch
        const want = @max(1, @min(@as(u32, @intCast(cpus)) -| 1, c.PAR_MAX_MEMBERS - 1));
        var i: u32 = 0;
        while (i < want) : (i += 1) {
            const t = @import("std").Thread.spawn(.{}, workerLoop, .{});
            if (t) |h| {
                h.detach();
                n_workers += 1;
            } else |_| break;
        }
        started = n_workers > 0;
    }

    /// Post ALL members to the shared cursor, then the main thread STEALS
    /// from the same cursor — every lane (main + workers) takes the next
    /// unclaimed system whenever it is free, so the split self-balances
    /// whatever the per-system cost. Falls back to fully sequential when
    /// the pool could not start. Returns after ALL members complete.
    fn runGroup(members: []const Job) void {
        if (!started or members.len < 2) {
            for (members) |j| {
                const t0 = profEnter();
                runPerEntity(j.fi, j.t);
                profLeave(j.fi, t0);
            }
            return;
        }
        // post members — ordering: jobs, then remaining/done reset, then
        // the epoch bump that releases the parked workers
        n_jobs = @intCast(members.len);
        for (members, 0..) |j, i| jobs[i] = j;
        done_flag.store(0, .monotonic);
        remaining.store(n_jobs, .release);
        cursor.store(0, .monotonic);
        _ = epoch.fetchAdd(1, .release);
        futexWakeAll(&epoch);
        // main thread steals from the same cursor as the workers
        while (true) {
            const j = cursor.fetchAdd(1, .monotonic);
            if (j >= n_jobs) break;
            const t0 = profEnter();
            runPerEntity(jobs[j].fi, jobs[j].t);
            profLeave(jobs[j].fi, t0);
            _ = remaining.fetchSub(1, .acq_rel);
        }
        // drain: wait for the workers' share. NOTE: main's own final
        // decrement must NOT block here — we wait on remaining hitting 0,
        // and the last WORKER out raises done_flag + wakes us. If main
        // itself took the last job, remaining is already 0 and the drain
        // falls straight through.
        while (remaining.load(.acquire) > 0) {
            futexWait(&done_flag, 0);
        }
    }
};

fn groupOf(fi: u32) ?u32 {
    var g: u32 = 0;
    while (g < c.n_par_groups) : (g += 1) {
        var m: u32 = 0;
        while (m < c.par_group_len[g]) : (m += 1) {
            if (c.parMember(g, m) == fi) return g;
        }
    }
    return null;
}

/// Execute `fi`'s group (persistent workers + main thread). Returns true if
/// the group ran; false if `fi` is not in any group (caller runs it alone).
fn runGroupParallel(fi: u32, done: *[c.MAX_FNS]bool) bool {
    if (!HAS_THREADS) return false;
    const g = groupOf(fi) orelse return false;
    const n = c.par_group_len[g];
    // WORTH HEURISTIC: waking a parked worker costs ~20-50us (futex + CFS
    // wakeup) — pointless when the group's own work is smaller. The work is
    // estimable at runtime from the bytecode itself: sum(fnLen x maxLive)
    // over the members. Calibrated on this host: ~1.1ns per instr-entity of
    // parallel savings vs a ~31us wake round trip -> break-even ~40k units.
    // Determinism is untouched either way — the decision only picks lanes,
    // never changes results.
    var work: u64 = 0;
    var m0: u32 = 0;
    while (m0 < n) : (m0 += 1) {
        const mfi: u32 = c.parMember(g, m0);
        work += @as(u64, c.fnLen(mfi)) * @as(u64, c.maxLive(c.fnEntity(mfi)));
    }
    if (work < 40_000) {
        m0 = 0;
        while (m0 < n) : (m0 += 1) {
            const mfi: u32 = c.parMember(g, m0);
            const t0 = profEnter();
            runPerEntity(mfi, c.fnEntity(mfi));
            profLeave(mfi, t0);
            done[mfi] = true;
        }
        return true;
    }
    var members: [c.PAR_MAX_MEMBERS]Job = undefined;
    var m: u32 = 0;
    while (m < n) : (m += 1) {
        const mfi: u32 = c.parMember(g, m);
        members[m] = .{ .fi = mfi, .t = c.fnEntity(mfi) };
    }
    Pool.runGroup(members[0..n]);
    m = 0;
    while (m < n) : (m += 1) {
        done[c.parMember(g, m)] = true;
    }
    return true;
}

pub fn startWorkers() void {
    if (threads_enabled and c.n_par_groups > 0) Pool.start();
}

/// Called after a hot-swap: the new binary may carry groups the old one
/// did not (or vice versa — the pool is harmless when idle).
pub fn ensureWorkers() void {
    Pool.start();
}

/// Hot-swap support: reload a new .libyte, preserving live entity state if
/// the entity/globals schema hash matches (manual Strategy 4).
var schema_sig: u64 = 0;

/// Entity-state backup region (meta + dense + cold + coldgen) — sized from
/// the build profile (was a hardcoded 0x60000 = the v1 default). Static, not
/// stack: sized for the profile, costs no wasm file bytes (bss).
const ENT_REGION: u32 = c.ADDR_CODE - c.ADDR_META;
var hot_backup: [ENT_REGION]u8 = undefined;
/// v9: page-0 scalars (input/frame/rand/scene/globals/shake/cam) ride
/// along — live-coding keeps your score, RNG phase and camera instead of
/// silently resetting them. The TF registry that types the globals block
/// is hashed into schema_sig, so a mismatched layout re-inits anyway.
const PAGE0_BYTES: u32 = c.ADDR_META;
var hot_backup_page0: [PAGE0_BYTES]u8 = undefined;

pub fn computeSchemaSig() u64 {
    // hash of entity schema + typed field registry + globals
    var h: u64 = 1469598103934665603;
    var i: u32 = 0;
    while (i < c.n_entities) : (i += 1) {
        h = hashBytes(h, c.mem[c.ENT_SCHEMA_BASE + i * 6 .. c.ENT_SCHEMA_BASE + i * 6 + 6]);
    }
    h = hashBytes(h, c.mem[c.TF_BASE .. c.TF_BASE + c.n_tfs * c.TF_SIZE]);
    return h;
}

fn hashBytes(seed: u64, bytes: []const u8) u64 {
    var h = seed;
    for (bytes) |b| {
        h ^= b;
        h *%= 1099511628211;
    }
    return h;
}

/// Snapshot of every schema global the loader mutates — the hotSwap
/// rollback needs all of them to truly restore the previous game.
const LoadCounts = struct {
    n_tfs: u32,
    n_entities: u32,
    n_fns: u32,
    n_tables: u32,
    n_scenes: u32,
    n_sprites: u32,
    n_tracks: u32,
    n_sfx: u32,
    n_anims: u32,
    n_texts: u32,
    n_keys: u32,
    n_fsms: u32,
    n_arrs: u32,
    n_par_groups: u32,
    pool_len: u32,
    instr_count: u32,
    game_w: u32,
    game_h: u32,
    game_wrap: bool,
    game_world_w: u32,
    game_world_h: u32,

    fn save() LoadCounts {
        return .{
            .n_tfs = c.n_tfs, .n_entities = c.n_entities, .n_fns = c.n_fns,
            .n_tables = c.n_tables, .n_scenes = c.n_scenes, .n_sprites = c.n_sprites,
            .n_tracks = c.n_tracks, .n_sfx = c.n_sfx, .n_anims = c.n_anims,
            .n_texts = c.n_texts, .n_keys = c.n_keys, .n_fsms = c.n_fsms,
            .n_arrs = c.n_arrs, .n_par_groups = c.n_par_groups,
            .pool_len = c.pool_len, .instr_count = c.instr_count,
            .game_w = c.game_w, .game_h = c.game_h, .game_wrap = c.game_wrap,
            .game_world_w = c.game_world_w, .game_world_h = c.game_world_h,
        };
    }

    fn restore(s: LoadCounts) void {
        c.n_tfs = s.n_tfs; c.n_entities = s.n_entities; c.n_fns = s.n_fns;
        c.n_tables = s.n_tables; c.n_scenes = s.n_scenes; c.n_sprites = s.n_sprites;
        c.n_tracks = s.n_tracks; c.n_sfx = s.n_sfx; c.n_anims = s.n_anims;
        c.n_texts = s.n_texts; c.n_keys = s.n_keys; c.n_fsms = s.n_fsms;
        c.n_arrs = s.n_arrs; c.n_par_groups = s.n_par_groups;
        c.pool_len = s.pool_len; c.instr_count = s.instr_count;
        c.game_w = s.game_w; c.game_h = s.game_h; c.game_wrap = s.game_wrap;
        c.game_world_w = s.game_world_w; c.game_world_h = s.game_world_h;
    }
};

pub fn rememberSchema() void {
    schema_sig = computeSchemaSig();
}

/// Returns true if entity/globals state was preserved.
pub fn hotSwap(blob: []const u8) bool {
    const old_sig = schema_sig;
    const saved: usize = ENT_REGION;
    @memcpy(hot_backup[0..saved], c.mem[c.ADDR_META .. c.ADDR_META + saved]);
    @memcpy(hot_backup_page0[0..PAGE0_BYTES], c.mem[0..PAGE0_BYTES]);
    // the loader owns the schema globals; snapshot them so a FAILED swap
    // rolls the whole engine view back (counts zeroed by the transactional
    // load would otherwise freeze the previous game)
    const snap_counts = LoadCounts.save();

    const rc = loader.load(blob);
    if (rc != c.ERR_OK) {
        // v14 hardening: the failed parse may have clobbered c.mem before
        // bailing (section writes precede validation failures). A watch-mode
        // host would keep calling frame() on the half-parsed state. Roll the
        // entity region + page-0 + schema back so the PREVIOUS game keeps
        // running deterministically; the host reads lila_last_error for the cause.
        LoadCounts.restore(snap_counts);
        @memcpy(c.mem[c.ADDR_META .. c.ADDR_META + saved], hot_backup[0..saved]);
        @memcpy(c.mem[0..PAGE0_BYTES], hot_backup_page0[0..PAGE0_BYTES]);
        return false;
    }
    vm.resetVm();
    render.streamReset();
    audio.reset();
    const new_sig = computeSchemaSig();
    if (new_sig == old_sig and saved == ENT_REGION) {
        // schema unchanged: restore entity + globals state
        @memcpy(c.mem[c.ADDR_META .. c.ADDR_META + saved], hot_backup[0..saved]);
        // v9: page-0 (frame counter, RNG phase, globals block, camera,
        // shake) restores too — the swap is transparent to gameplay.
        @memcpy(c.mem[0..PAGE0_BYTES], hot_backup_page0[0..PAGE0_BYTES]);
        rememberSchema();
        return true;
    }
    // schema changed: full re-init
    rememberSchema();
    var i: u32 = 0;
    while (i < c.n_fns) : (i += 1) {
        if (c.fnKind(i) == c.FN_INIT) vm.run(i);
    }
    return false;
}
