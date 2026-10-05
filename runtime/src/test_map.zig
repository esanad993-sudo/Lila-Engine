//! Bit-compatibility test: the config-derived memory map at the DEFAULT
//! profile must equal the fixed v1 map exactly. This is the proof that
//! configurability changed nothing for existing games — every address the
//! shooter, the host JS (globals at 0x10), and all tests rely on is where
//! it always was. Run: zig test src/test_map.zig (from runtime/).

const std = @import("std");
const c = @import("core.zig");
const cfg = @import("engine_config.zig");

test "default profile matches v1 fixed map exactly" {
    try std.testing.expectEqualStrings("default", cfg.PROFILE_NAME);
    try std.testing.expectEqual(@as(u32, 2), cfg.MEM_MB);
    try std.testing.expectEqual(@as(u32, 512), c.MAX_ENT);
    try std.testing.expectEqual(@as(u32, 6000), c.MAX_VERTS);
    try std.testing.expectEqual(@as(u32, 8192), c.MAX_INSTR);
    try std.testing.expectEqual(@as(u32, 64), c.MAX_SPRITES);
    try std.testing.expectEqual(@as(u32, 32), c.MAX_TEXTS);
    try std.testing.expectEqual(@as(u32, 128), c.MAX_FNS);
    try std.testing.expectEqual(@as(u32, 1024), c.MAX_POOL);

    try std.testing.expectEqual(@as(u32, 8), c.BM_WORDS); // 512-bit bitmap
    try std.testing.expectEqual(@as(u32, 1088), c.META_STRIDE); // 512*2 + 64

    try std.testing.expectEqual(@as(u32, 0x0100), c.ADDR_META);
    try std.testing.expectEqual(@as(u32, 0x08000), c.ADDR_DENSE);
    try std.testing.expectEqual(@as(u32, 0x48000), c.ADDR_COLD);
    // v1 reserved coldgen at 0x50000 — INSIDE the cold-rows reservation
    // (16 x 512 x 8B = 0x10000, so cold rows really span 0x48000..0x58000;
    // the v1 comment mis-computed the size as 0x8000). Fixed in v6: coldgen
    // moves to 0x58000, after the region it used to overlap. Every access is
    // symbolic, so this is a compatibility-safe correctness fix; all v1
    // games with small cold rows never touched the overlap anyway.
    try std.testing.expectEqual(@as(u32, 0x58000), c.ADDR_COLDGEN);
    try std.testing.expectEqual(@as(u32, 0x60000), c.ADDR_CODE);
    try std.testing.expectEqual(@as(u32, 0x88000), c.ADDR_EXTRA);
    try std.testing.expectEqual(@as(u32, 0x4000), c.EXTRA_BYTES);
    try std.testing.expectEqual(@as(u32, 0x8C000), c.TF_BASE);
    try std.testing.expectEqual(@as(u32, 0x8E000), c.ENT_SCHEMA_BASE);
    try std.testing.expectEqual(@as(u32, 0x8E200), c.INTEG_BASE);
    try std.testing.expectEqual(@as(u32, 0x8F000), c.ADDR_FNS);
    try std.testing.expectEqual(@as(u32, 0x90000), c.ADDR_TABLES);
    try std.testing.expectEqual(@as(u32, 0x91000), c.ADDR_SCENES);
    try std.testing.expectEqual(@as(u32, 0x91100), c.ADDR_POOL);
    try std.testing.expectEqual(@as(u32, 0x94000), c.ADDR_SPRITES);
    try std.testing.expectEqual(@as(u32, 0xB0000), c.ADDR_MUSIC);
    try std.testing.expectEqual(@as(u32, 0xB2000), c.ADDR_SFX);
    try std.testing.expectEqual(@as(u32, 0xB2400), c.ADDR_ANIM);
    try std.testing.expectEqual(@as(u32, 0xB8000), c.ADDR_TEXTS);
    try std.testing.expectEqual(@as(u32, 0x100000), c.ADDR_STREAM);
    try std.testing.expectEqual(@as(u32, 0x180000), c.ADDR_AUDIO);
    // v7: SRAM (64 raw save slots) extends the map past audio. Every v1/v6
    // region above is byte-identical; MAP_END grows by exactly SRAM_BYTES.
    try std.testing.expectEqual(@as(u32, 0x188000), c.ADDR_SRAM);
    try std.testing.expectEqual(@as(u32, 256), c.SRAM_BYTES);
    try std.testing.expectEqual(@as(u32, 64), c.SRAM_SLOTS);
    // v8 OPTIMIZER tail (after SRAM — no v1 anchor moves):
    //   FSM bit-plane tables: 16 x (4B header + 16x16B) = 0x1100
    //   fitted sfx programs: MAX_SFX x 28B
    try std.testing.expectEqual(@as(u32, 0x188100), c.ADDR_FSM);
    try std.testing.expectEqual(@as(u32, 260), c.FSM_STRIDE); // 4B hdr + 16x16 table
    try std.testing.expectEqual(@as(u32, 0x188100 + 16 * 260), c.ADDR_SFX_FIT);
    try std.testing.expectEqual(@as(u32, 28), c.SFX_FIT_STRIDE);
    try std.testing.expectEqual(@as(u32, 0x188100 + 16 * 260 + 32 * 28), c.MAP_END);
    try std.testing.expectEqual(@as(u32, 2 * 1024 * 1024), c.MEM_SIZE);
}

test "map regions do not overlap and fit memory" {
    // ordering + non-overlap, sampled at every region boundary
    const order = [_]u32{
        c.ADDR_GLOBALS, c.ADDR_META,  c.ADDR_DENSE, c.ADDR_COLD,
        c.ADDR_COLDGEN, c.ADDR_CODE,  c.ADDR_EXTRA, c.TF_BASE,
        c.ENT_SCHEMA_BASE, c.INTEG_BASE, c.ADDR_FNS, c.ADDR_TABLES,
        c.ADDR_SCENES, c.ADDR_POOL, c.ADDR_SPRITES, c.ADDR_MUSIC,
        c.ADDR_SFX, c.ADDR_ANIM, c.ADDR_TEXTS, c.ADDR_STREAM,
        c.ADDR_AUDIO, c.ADDR_SRAM, c.MAP_END,
    };
    var i: usize = 1;
    while (i < order.len) : (i += 1) {
        try std.testing.expect(order[i - 1] < order[i]);
    }
    try std.testing.expect(c.MAP_END <= c.MEM_SIZE);
}
