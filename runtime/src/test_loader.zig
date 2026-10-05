//! v14 loader robustness tests: the .libyte parser must treat every input
//! as hostile — truncated, mutated, or crafted files return TYPED error
//! codes and never panic (the ReleaseSafe trap contract).
//!
//! The fixture is REAL compiler output (the demo game, embedded), so these
//! tests exercise the exact bytes a game ships. Run: zig test src/test_loader.zig

const std = @import("std");
const c = @import("core.zig");
const loader = @import("loader.zig");
const engine = @import("engine.zig");

const FIXTURE = @embedFile("test_fixture.libyte");

/// Deterministic xorshift64 PRNG (mirrors the Rust disasm fuzz — the same
/// philosophy: no external deps, stable across runs and platforms).
const Rng = struct {
    s: u64,
    fn next(r: *Rng) u64 {
        var x = r.s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        r.s = x;
        return x;
    }
    fn byte(r: *Rng) u8 {
        return @truncate(r.next() >> 32);
    }
};

test "fixture: real compiler output loads clean" {
    // engine.init (not bare loader.load) so the full decode + entity setup
    // path runs; the demo game must come up without a single error
    const rc = engine.init(FIXTURE);
    try std.testing.expectEqual(@as(i32, c.ERR_OK), rc);
    try std.testing.expect(c.n_fns > 0);
    try std.testing.expect(c.n_entities == 3);
    try std.testing.expect(c.instr_count > 0);
}

test "truncation at every prefix length returns a typed error" {
    // every strict prefix of the fixture either fails loudly with a code
    // or (short prefixes) fails the magic check — never a panic
    var len: usize = 0;
    while (len < FIXTURE.len) : (len += 1) {
        const rc = loader.load(FIXTURE[0..len]);
        try std.testing.expect(rc != c.ERR_OK);
        // the failure must be a KNOWN code, not garbage
        try std.testing.expect(rc >= -23);
    }
}

test "single-byte mutations never panic (2000 rounds)" {
    var rng = Rng{ .s = 0x5EED_1A4E };
    var ok: u32 = 0;
    var err: u32 = 0;
    var buf: [4096]u8 = undefined;
    var round: u32 = 0;
    while (round < 2000) : (round += 1) {
        @memcpy(buf[0..FIXTURE.len], FIXTURE);
        var blob: []u8 = buf[0..FIXTURE.len];
        const n_mut = 1 + rng.next() % 8;
        var m: u32 = 0;
        while (m < n_mut) : (m += 1) {
            const pos = rng.next() % blob.len;
            switch (rng.next() % 3) {
                0 => blob[pos] ^= @as(u8, 1) << @intCast(rng.byte() % 8),
                1 => blob[pos] = rng.byte(),
                else => blob[pos] +%= rng.byte(),
            }
        }
        // occasionally truncate or extend
        switch (round % 4) {
            0 => {
                const cut = rng.next() % (blob.len + 1);
                blob = blob[0..cut];
            },
            1 => {
                blob.len += 1; // within the 4096 buffer
                blob[blob.len - 1] = rng.byte();
            },
            else => {},
        }
        const rc = loader.load(blob);
        if (rc == c.ERR_OK) ok += 1 else err += 1;
        // after a failed load the counts are zeroed (transactional contract)
        if (rc != c.ERR_OK) {
            try std.testing.expectEqual(@as(u32, 0), c.n_fns);
            try std.testing.expectEqual(@as(u32, 0), c.n_entities);
        }
    }
    try std.testing.expectEqual(@as(u32, 2000), ok + err);
}

test "bad magic and version are typed errors" {
    var buf: [4096]u8 = undefined;
    @memcpy(buf[0..FIXTURE.len], FIXTURE);
    var blob: []u8 = buf[0..FIXTURE.len];

    blob[0] = 'X'; // break magic
    try std.testing.expectEqual(c.ERR_MAGIC, loader.load(blob));

    @memcpy(blob[0..FIXTURE.len], FIXTURE);
    blob[4] = 9; // version 9 (LE low byte)
    blob[5] = 0;
    try std.testing.expectEqual(c.ERR_VERSION, loader.load(blob));
}

test "over-limit header counts are loud limit errors" {
    var buf: [4096]u8 = undefined;
    @memcpy(buf[0..FIXTURE.len], FIXTURE);
    var blob: []u8 = buf[0..FIXTURE.len];

    // n_entities byte at offset 12: claim 200 types (build has 16)
    blob[12] = 200;
    try std.testing.expectEqual(c.ERR_LIMIT_TYPES, loader.load(blob));

    @memcpy(blob[0..FIXTURE.len], FIXTURE);
    // n_fns u16 at 13: claim 9999 fns (build has 128)
    blob[13] = 0x0F;
    blob[14] = 0x27; // 9999 LE
    try std.testing.expectEqual(c.ERR_LIMIT_FNS, loader.load(blob));
}

test "fn table kind/range validation fires" {
    var buf: [4096]u8 = undefined;
    @memcpy(buf[0..FIXTURE.len], FIXTURE);
    var blob: []u8 = buf[0..FIXTURE.len];

    // find a fn record: [kind, ent=0xFF, nparams, nlocals, start u32, len u32]
    // (the demo's init fn is global: ent byte 0xFF anchors it)
    var at: ?usize = null;
    var i: usize = 40;
    while (i + 12 <= blob.len) : (i += 1) {
        if (blob[i] == 0 and blob[i + 1] == 0xFF and blob[i + 2] <= 3 and blob[i + 3] <= 32) {
            const start = std.mem.readInt(u32, blob[i + 4 ..][0..4], .little);
            const len = std.mem.readInt(u32, blob[i + 8 ..][0..4], .little);
            if (start <= 4096 and len >= 1 and len <= 4096 and start + len <= 4096) {
                at = i;
                break;
            }
        }
    }
    try std.testing.expect(at != null);
    const rec = at.?;

    // crafted kind byte 99 -> ERR_BAD_HEADER
    blob[rec] = 99;
    try std.testing.expectEqual(c.ERR_BAD_HEADER, loader.load(blob));

    // crafted start beyond MAX_INSTR -> ERR_LIMIT_INSTR
    @memcpy(blob[0..FIXTURE.len], FIXTURE);
    std.mem.writeInt(u32, blob[rec + 4 ..][0..4], 900_000, .little);
    try std.testing.expectEqual(c.ERR_LIMIT_INSTR, loader.load(blob));

    // crafted len escaping the instruction array -> ERR_LIMIT_INSTR
    @memcpy(blob[0..FIXTURE.len], FIXTURE);
    std.mem.writeInt(u32, blob[rec + 8 ..][0..4], 900_000, .little);
    try std.testing.expectEqual(c.ERR_LIMIT_INSTR, loader.load(blob));
}

test "huffman length bytes > 8 are neutralized" {
    var buf: [4096]u8 = undefined;
    @memcpy(buf[0..FIXTURE.len], FIXTURE);
    var blob: []u8 = buf[0..FIXTURE.len];

    // find the opcode huffman table: after pool. Locate (sym, len) pairs by
    // scanning for the table marker region — patch every len byte after the
    // pool count to 0xFF; the loader must zero them and still fail LOUDLY
    // (or load fine) — never trap. Conservative: patch bytes in the last
    // 25% of the file (code+huffman region).
    var i: usize = blob.len * 3 / 4;
    while (i < blob.len - 4) : (i += 2) {
        if (blob[i + 1] > 8 and blob[i + 1] != 0xFF) blob[i + 1] = 0xFF;
    }
    const rc = loader.load(blob);
    try std.testing.expect(rc == c.ERR_OK or rc == c.ERR_HUFF or rc == c.ERR_TRUNCATED or rc == c.ERR_BAD_HEADER);
}

test "failed load leaves the engine counts zeroed (transactional)" {
    var buf: [4096]u8 = undefined;
    @memcpy(buf[0..FIXTURE.len], FIXTURE);
    const good = loader.load(buf[0..FIXTURE.len]);
    try std.testing.expectEqual(c.ERR_OK, good);
    const fns_before = c.n_fns;
    try std.testing.expect(fns_before > 0);

    // now a corrupted blob: header n_entities out of range
    var bad: [4096]u8 = undefined;
    @memcpy(bad[0..FIXTURE.len], FIXTURE);
    bad[12] = 200;
    const rc = loader.load(bad[0..FIXTURE.len]);
    try std.testing.expectEqual(c.ERR_LIMIT_TYPES, rc);
    try std.testing.expectEqual(@as(u32, 0), c.n_fns);
    try std.testing.expectEqual(@as(u32, 0), c.n_entities);
    try std.testing.expectEqual(@as(u32, 0), c.instr_count);
}

test "hotSwap rollback: a failed swap keeps the previous game running" {
    const ent = @import("entity.zig");
    const vm = @import("vm.zig");

    // boot the good game and run one frame
    try std.testing.expectEqual(@as(i32, c.ERR_OK), engine.init(FIXTURE));
    c.wr32(c.ADDR_INPUT, 0);
    engine.frame();
    const frames_before = c.rd32(c.ADDR_FRAME);
    try std.testing.expect(frames_before == 1);
    engine.rememberSchema();

    // swap in garbage: must fail AND roll back
    var garbage: [64]u8 = undefined;
    @memset(&garbage, 0xAA);
    const preserved = engine.hotSwap(&garbage);
    try std.testing.expect(!preserved);
    // the previous game's schema survived: counts intact, frame continues
    try std.testing.expect(c.n_fns > 0);
    engine.frame();
    try std.testing.expectEqual(frames_before + 1, c.rd32(c.ADDR_FRAME));

    // entity iteration still works on the rolled-back schema
    var t: u32 = 0;
    while (t < c.n_entities) : (t += 1) {
        try std.testing.expect(ent.count(t) >= 0); // iterates without trapping
    }
    _ = vm; // (vm state is per-worker; frame() already exercises it)
}
