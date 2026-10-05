//! Prints the engine build profile + derived memory map (for build_engine.sh
//! and docs). Imports core.zig directly — can never drift from the real map.
const std = @import("std");
const c = @import("core.zig");
const cfg = @import("engine_config.zig");

pub fn main() void {
    std.debug.print("profile: {s}\n", .{cfg.PROFILE_NAME});
    std.debug.print("mem: {d} MB ({d} bytes, map uses {d} = {d}%)\n", .{
        cfg.MEM_MB, c.MEM_SIZE, c.MAP_END, c.MAP_END * 100 / c.MEM_SIZE,
    });
    std.debug.print("caps: ent {d}/type | verts {d}/frame | instr {d} | sprites {d} | texts {d} | anims {d} | fns {d} | pool {d} | tracks {d} | sfx {d}\n", .{
        c.MAX_ENT, c.MAX_VERTS, c.MAX_INSTR, c.MAX_SPRITES,
        c.MAX_TEXTS, c.MAX_ANIMS, c.MAX_FNS, c.MAX_POOL, c.MAX_TRACKS, c.MAX_SFX,
    });
    std.debug.print("map:\n", .{});
    std.debug.print("  0x{X:0>6}  input/frame/rand/scene/shake\n", .{@as(u32, 0)});
    std.debug.print("  0x{X:0>6}  globals ({d}B)\n", .{ c.ADDR_GLOBALS, c.GLOBALS_BYTES });
    std.debug.print("  0x{X:0>6}  entity meta    {d}B ({d} types x {d})\n", .{ c.ADDR_META, c.MAX_TYPES * c.META_STRIDE, c.MAX_TYPES, c.META_STRIDE });
    std.debug.print("  0x{X:0>6}  dense rows     {d}B\n", .{ c.ADDR_DENSE, c.MAX_TYPES * c.DENSE_TYPE_STRIDE });
    std.debug.print("  0x{X:0>6}  cold rows      {d}B\n", .{ c.ADDR_COLD, c.MAX_TYPES * c.COLD_TYPE_STRIDE });
    std.debug.print("  0x{X:0>6}  cold gens      {d}B\n", .{ c.ADDR_COLDGEN, c.MAX_TYPES * c.MAX_ENT * 2 });
    std.debug.print("  0x{X:0>6}  code           {d}B ({d} instrs x {d})\n", .{ c.ADDR_CODE, c.MAX_INSTR * c.INSTR_SIZE, c.MAX_INSTR, c.INSTR_SIZE });
    std.debug.print("  0x{X:0>6}  spawn extra    {d}B\n", .{ c.ADDR_EXTRA, c.EXTRA_BYTES });
    std.debug.print("  0x{X:0>6}  typed fields   {d}B ({d} x {d})\n", .{ c.TF_BASE, c.MAX_TF * c.TF_SIZE, c.MAX_TF, c.TF_SIZE });
    std.debug.print("  0x{X:0>6}  ent schemas    + integ pairs\n", .{c.ENT_SCHEMA_BASE});
    std.debug.print("  0x{X:0>6}  fn records     {d}B ({d} x {d})\n", .{ c.ADDR_FNS, c.MAX_FNS * c.FN_SIZE, c.MAX_FNS, c.FN_SIZE });
    std.debug.print("  0x{X:0>6}  tables         {d}B\n", .{ c.ADDR_TABLES, c.MAX_TABLES * c.TABLE_STRIDE });
    std.debug.print("  0x{X:0>6}  scenes+pool    {d}B\n", .{ c.ADDR_SCENES, c.MAX_SCENES * 2 + c.MAX_POOL * 4 });
    std.debug.print("  0x{X:0>6}  sprites        {d}B ({d} x {d}B)\n", .{ c.ADDR_SPRITES, c.MAX_SPRITES * c.SPRITE_STRIDE, c.MAX_SPRITES, c.SPRITE_STRIDE });
    std.debug.print("  0x{X:0>6}  music+sfx      {d}B\n", .{ c.ADDR_MUSIC, c.MAX_TRACKS * c.MUSIC_STRIDE + c.MAX_SFX * 8 });
    std.debug.print("  0x{X:0>6}  anims          {d}B\n", .{ c.ADDR_ANIM, c.MAX_ANIMS * c.ANIM_STRIDE });
    std.debug.print("  0x{X:0>6}  texts          {d}B\n", .{ c.ADDR_TEXTS, c.MAX_TEXTS * c.TEXT_STRIDE });
    std.debug.print("  0x{X:0>6}  vertex stream  {d}B ({d} verts x 14)\n", .{ c.ADDR_STREAM, 4 + c.MAX_VERTS * 14, c.MAX_VERTS });
    std.debug.print("  0x{X:0>6}  audio          {d}B\n", .{ c.ADDR_AUDIO, c.AUDIO_BYTES });
    std.debug.print("  0x{X:0>6}  MAP_END (reserved tail: {d}B)\n", .{ c.MAP_END, c.MEM_SIZE - c.MAP_END });
}
