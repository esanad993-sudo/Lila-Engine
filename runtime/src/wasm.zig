//! LILA wasm32-freestanding entry: the WebAssembly ABI.
//! Build: zig build-exe src/wasm.zig -target wasm32-freestanding -fno-entry
//!        -rdynamic -O ReleaseSmall
//! Exports: memory + lila_init/lila_frame/lila_stream_ptr/lila_stream_count/
//!          lila_audio_fill/lila_version/lila_hot_swap
//! v6 additions (build-profile introspection + diagnostics):
//!          lila_mem_size/lila_cap_ent/lila_cap_verts/lila_cap_instr/
//!          lila_stream_dropped/lila_spawn_denied/lila_last_error
//! v7 additions (ABI 3 — additive): lila_sram_ptr/lila_sram_dirty/
//!          lila_cap_types/lila_world_w/lila_world_h

const std = @import("std");
const c = @import("core.zig");
const engine = @import("engine.zig");
const audio = @import("audio.zig");

var blob_store: [256 * 1024]u8 = undefined;

export fn lila_init(ptr: [*]const u8, len: u32) i32 {
    if (len == 0 or len > blob_store.len) return c.ERR_TOO_BIG;
    @memcpy(blob_store[0..len], ptr[0..len]);
    // fresh session: reset diagnostics telemetry
    c.stream_drop_total = 0;
    c.spawn_denied_total = 0;
    c.last_error = 0;
    const rc = engine.init(blob_store[0..len]);
    if (rc == c.ERR_OK) engine.rememberSchema();
    return rc;
}

export fn lila_hot_swap(ptr: [*]const u8, len: u32) i32 {
    if (len == 0 or len > blob_store.len) return c.ERR_TOO_BIG;
    @memcpy(blob_store[0..len], ptr[0..len]);
    const preserved = engine.hotSwap(blob_store[0..len]);
    return if (preserved) 1 else 0;
}

export fn lila_frame() void {
    engine.frame();
}

export fn lila_version() i32 {
    return 3;
}

// ---------------- v7: SRAM persistence + world introspection ----------------

/// Linear address of the 256-byte SRAM block. Hosts copy saved bytes here
/// BEFORE calling lila_init (the engine's zeroing pass skips this range, so
/// saved values survive init, re-init, and hot-swap).
export fn lila_sram_ptr() i32 {
    return @intCast(c.ADDR_SRAM + @intFromPtr(&c.mem));
}

/// Returns 1 if any save() happened since the last call, and clears the
/// flag — hosts poll this each frame and persist only on change.
export fn lila_sram_dirty() i32 {
    if (c.sram_dirty != 0) {
        c.sram_dirty = 0;
        return 1;
    }
    return 0;
}

/// Entity types this build supports (profile knob max_types).
export fn lila_cap_types() i32 {
    return @intCast(c.MAX_TYPES);
}

/// The loaded game's simulation bounds (world; == screen when undeclared).
export fn lila_world_w() i32 {
    return @intCast(c.game_world_w);
}
export fn lila_world_h() i32 {
    return @intCast(c.game_world_h);
}

// ---------------- v6: build-profile introspection + diagnostics ----------------
// The host reads these to display the engine profile and surface budget
// problems — same class of telemetry a native engine logs.

/// Total flat engine memory in bytes (the configurable "2MB").
export fn lila_mem_size() i32 {
    return @intCast(c.MEM_SIZE);
}
/// Entity slots per type this build provides.
export fn lila_cap_ent() i32 {
    return @intCast(c.MAX_ENT);
}
/// Per-frame render vertex budget.
export fn lila_cap_verts() i32 {
    return @intCast(c.MAX_VERTS);
}
/// Decoded instruction capacity.
export fn lila_cap_instr() i32 {
    return @intCast(c.MAX_INSTR);
}
/// Cumulative vertices dropped because the vertex budget was exceeded.
export fn lila_stream_dropped() i32 {
    return @intCast(c.stream_drop_total);
}
/// Cumulative spawns denied because an entity pool / per-type cap was full.
export fn lila_spawn_denied() i32 {
    return @intCast(c.spawn_denied_total);
}
/// Live entities right now (all types) — host telemetry.
export fn lila_live_ents() i32 {
    const ent = @import("entity.zig");
    var n: i32 = 0;
    var t: u32 = 0;
    while (t < c.n_entities) : (t += 1) {
        n += @intCast(ent.count(t));
    }
    return n;
}
/// Loader return code of the most recent load attempt (0 = ok).
export fn lila_last_error() i32 {
    return c.last_error;
}

/// Address of the staging buffer (JS copies fetched .libyte bytes here).
export fn lila_blob_ptr() i32 {
    return @intCast(@intFromPtr(&blob_store));
}

/// Linear address of the engine's flat 2MB buffer (its base is NOT 0 in
/// wasm: Zig places statics after the stack). All engine-space addresses
/// (input 0x00, stream, audio scratch) are offsets from THIS base.
export fn lila_mem_ptr() i32 {
    return @intCast(@intFromPtr(&c.mem));
}

export fn lila_stream_ptr() i32 {
    return @intCast(c.ADDR_STREAM + 4);
}

export fn lila_stream_count() i32 {
    return @intCast(c.rd32(c.ADDR_STREAM));
}

/// Fill `len` i16 PCM samples at `dst` (a wasm-memory address).
/// Returns samples written. Called by the host audio path.
export fn lila_audio_fill(dst: i32, len: i32) i32 {
    if (len <= 0) return 0;
    const n: usize = @intCast(@min(len, 8192));
    // dst is ENGINE-space; convert to a wasm linear pointer.
    // v14: clamp the window to the 2MB flat buffer — a bad host-side dst
    // used to write straight through wasm memory (host-triggerable OOB).
    const base: usize = @intFromPtr(&c.mem);
    const mem_size: usize = c.mem.len;
    const off: usize = @as(usize, @intCast(@as(u32, @bitCast(dst)))) & ~@as(usize, 1);
    const avail: usize = if (off >= mem_size) 0 else (mem_size - off) / 2;
    if (avail == 0) return 0;
    const count: usize = @min(n, avail);
    const slice: []i16 = @as([*]i16, @ptrFromInt(base + off))[0..count];
    const written = audio.pull(slice);
    return @intCast(written);
}

/// Read/write the 4-byte input bitmask at address 0 (manual section 5).
export fn lila_input_set(mask: u32) void {
    c.wr32(c.ADDR_INPUT, mask);
}
