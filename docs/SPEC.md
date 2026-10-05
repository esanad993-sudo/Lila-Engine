# LILA Language Specification (v1, frozen)

LILA = **L**ightweight **I**nteger **L**anguage for **A**rcade.
A compile-time-compiled DSL for deterministic 2D arcade games. Strict Q24.8
fixed-point; no runtime strings; `#atom` resource symbols; bit-packed entities.

This is the internal working spec. User-facing doc: `docs/LANGUAGE.md`.

## Lexical
- Comments: `//` to end of line.
- Identifiers: `[A-Za-z_][A-Za-z0-9_]*`.
- Atoms: `#` + identifier, e.g. `#shoot`. Atoms are compile-time symbols
  (interned per domain table: sound, key, text, scene, sprite, music, anim, sfx).
  Raw string literals are ONLY allowed in `text {}` declarations — the language
  bans runtime strings everywhere else.
- Integer literals: decimal `42`, hex `0x3AF` (fits u32).
- Fixed literals: `0.25`, `-6.5` (compiled to Q24.8 i32; 8 fractional bits,
  2 decimals max).
- Keywords: game sound key text scene sprite music sfx anim entity global
  table fn let if else while for spawn kill sfx music stop_music goto draw
  draw_text draw_num call rand sin cos ang fixed call_count true false cold.
  (Note: `sfx`/`music` are both decl keywords and statement keywords —
  context disambiguates.)
- Operators: `+ - * / % & | ^ << >>` `== != < > <= >=` `&& || !` `= += -= *=`
  `( ) { } [ ] , ; : ?`
- No `?` ternary (keep parser small). Boolean ops are `&& || !`.

## Top-level declarations (in order)
1. `game { title: "..." (the only string outside text{}), width: 512, height: 512, wrap: true }`
   width/height must be power-of-two ints if wrap:true (compiler emits `& (w<<8 - 1)` masks).
2. `sound { #shoot, #explode, ... }` — declares sound-effect atoms.
3. `key { #left, #right, ... }` — input atoms → bits of the 4-byte input bitmask.
   First 32 keys map to bits 0..31 of the u32 at address 0.
4. `text { #score_label: "SCORE", #over: "GAME OVER" }` — the ONLY strings.
5. `scene { #play: start_play, #over: start_over }` — atom → 0-arg fn.
6. `sprite #ship (16) { poly [...] fill: 0x33AAFF; ... }` — vector sprites.
   `(16)` = hit radius in px (for compiler-side collision checks/documentation).
   `poly [ x0,y0, x1,y1, ... ]` = closed polygon, integer pixel coords,
   origin at sprite center. Modifiers: `fill: 0xRRGGBB`, optional
   `grad: 0xRRGGBB` (per-vertex color lerp), `add: true` (additive blend token),
   `pat: flat|noise` (uber-shader pattern token bits).
   Compiler: polygon → triangle fan → 14-byte quantized vertices
   (i16 pos, u16 uv, u32 rgba, u16 token).
7. `music #track1 { bpm: 150, square: [C4 - E4 - G4 ...], tri: [C2 ...], noise: [x . x .] }`
   16 steps per voice. Notes: `C4`..`B8` (semitone index), `-` = hold,
   `.` = silence, `x` = noise hit. Optional `vol: 0..15` per voice.
8. `sfx #shoot { wave: square, freq: 880, sweep: -400, decay: 12, vol: 12 }`
   wave ∈ square|tri|noise|saw. freq Hz, sweep Hz/sec, decay in 1/60s units, vol 0..15.
9. `anim #wave { keys: [ (0,0), (30,40), (60,10), (90,40) ] }`
   Compiler compresses keyframes → cubic Hermite control points (RDP simplify +
   Catmull-Rom tangents). Runtime O(1) evaluation. Values are fixed.
10. `entity Enemy { x: fixed, y: fixed, vx: fixed, vy: fixed, hp: u = 5, ... }`
    Field types: `fixed` (Q24.8 i32), `uN`/`sN` (N=1..16 bit ints), `bool`,
    `ang` (u16, 0..4095 = 0..360 degrees), plain `u`/`s` = width inferred from
    initializer (minimum 4 bits; no default → u8).
    `cold field: ...` marks seldom-used fields → sparse side-table storage.
    Default values after `=` (int or fixed literal; `true`/`false` for bool).
    Plain `u`/`s` width = inferred from initializer, minimum 4 bits
    (e.g. `hp: u = 3` → u4; range 0..15). Field names are globally unique
    across entities (enables direct field-id ops).
11. `global { score: u16 = 0, wave: u = 1, ... }` — same field types, no `cold`.
    Globals live at comptime-known offsets in the flat buffer (live-patchable
    from JS/native — "0-cost runtime tweaking").
12. `table behaviors { 0: enemy_drift, 1: enemy_sine, 2: enemy_chase }`
    u8-keyed fn dispatch (the Wasm-Table-style behavior registry).
13. `fn` declarations — kinds by shape (auto-run vs helper):
    - `fn init()` — runs once at startup (alias: `fn start()`).
    - `fn update()` — runs every frame, FIRST (global logic).
    - `fn update(e: T)` — runs once per live entity of T, logic pass,
      entity-declaration order.
    - `fn draw()` — runs every frame, LAST (HUD).
    - `fn draw(e: T)` — per live entity, render pass, declaration order.
      Draw fns may ONLY contain draw*/sfx/music statements (checker-enforced).
    - Any other name = helper: `call name;` / `call name(a1, a2);` 0-2 params
      of any simple type (fixed/int/bool/ang) or an entity ref.
    Scene fns (referenced by `scene {}`) are 0-arg helpers invoked by `goto`.
    Table fns are entity-helpers invoked via `call table[k](e);`.

## Statements
- `let name = expr;` — local (fixed/int/bool/ang/entity-ref typed by checker).
- `if cond { } else { }` / `else if` chains.
- `while cond { }`
- `for (b: Bullet) { }` — iterate live entities (bitmap ctz order = ascending index).
- `spawn Enemy { x: e.x, y: 16, vx: 0, vy: 0.5 };` or `spawn Enemy;` (defaults).
  Field init exprs evaluated in written order.
- `kill e;` — kill entity referenced by local/param (gen++ on its slot).
- `sfx(#shoot);` `music(#track1);` `stop_music();`
- `goto(#play);` — switch scene + call its fn.
- `draw(#ship, e.x, e.y, rot, scale, 0xRRGGBBAA);` — rot: ang expr, scale: fixed.
- `draw_text(#score_label, x, y, 0xRRGGBBAA);` `draw_num(expr, x, y, 0xRRGGBBAA);`
  (draw_num renders the integer part of a fixed value, or an int value).
- `call name;` / `call name(expr, expr);` — helper call (0-2 args).
- `call behaviors[e.behavior](e);` — table dispatch. Only indirect call form.
- `return;` — early return from current fn.
- Assignment: `target = expr;` `+=` `-=` `*=` for fixed/int targets.

## Expressions
- Literals: int (u32 range), fixed (Q24.8), `true`, `false`.
- Atoms in expression position: only as args to `key(#x)`, `sfx(#x)`, ...,
  `draw(#x,...)`, `anim(#x, t)`, `music(#x)`, `draw_text(#x,...)`, `goto(#x)`.
- `key(#left)` → bool (bitmask test).
- `sin(a)`, `cos(a)` — arg is `ang` (u16 0..4095); returns fixed. LUT-based.
- `ang(x)` — cast fixed/int → ang (scales: 360deg == 4096).
- `fixed(x)` / `int(x)` — explicit casts.
- `rand(N)` → u32 in 0..N-1 (deterministic LCG, seed in globals area).
- `anim(#id, t)` → fixed (t int frame counter; t wraps mod curve duration) —
  Hermite eval. RDP-simplified keys → control points.
- `count(Enemy)` → u16 number of live entities (int type).
- `self` is NOT a keyword; the entity param name is arbitrary (`e` by convention).
- Field access on entity locals/params: `e.hp`. On globals: bare name.
- Precedence (low→high): `||` < `&&` < comparisons < `& | ^` < `<< >>` <
  `+ -` < `* / %` < unary `- !` < postfix `.field` / calls / atoms.
- Type rules (checker): fixed⊕fixed→fixed; int⊕int→int (i32 internally,
  range-checked on store to bit-packed fields); **int widens to fixed
  implicitly in mixed arithmetic/comparison/assignment** (C int→double
  analogy); bool only with `&& || !` and comparisons; ang accepts int
  literals in +/- and passes to sin/cos/draw via explicit `ang()` cast or
  ang-typed exprs; comparisons yield bool; Q24.8 constants fold at compile time.
- Screen convention: rot 0 = facing UP; `vx = sin(rot)`, `vy = -cos(rot)`.

## Semantics / determinism
- All spatial state Q24.8 i32. World wrap = `& (dim<<8 - 1)` (auto-applied on
  stores to `fixed` fields when game.wrap — compiler injects mask).
- i32 overflow wraps (u32/i32 two's complement) — CPU-integer-native.
- Entity handles: u32 = (gen:12 << 20) | index:20. Kill → gen++.
- Iteration order: ascending slot index via 1-bit existence bitmap ctz scan.
- rand: LCG `x = x*1664525 + 1013904223 (mod 2^32)`.
- 60 fixed steps/sec; audio pulled separately (sample-accurate tracker).

## .libyte binary format (little-endian)
```
"LIBY" u8[4], version u16 (1 or 2), flags u16 (bit0: wrap)
game: width u16, height u16
[v2 only, present iff the game declared a capacity OR world dims:]
    capacity_entities u32,
    world u32  (v7: world_w | world_h<<16; 0/absent = world == screen)
counts: nsprites u16, nmusic u16, nsfx u16, nanim u16, nentities u8,
        nglobals u8, nfields u16, nfns u16, ntables u8, natoms_total u16
atom tables: for each domain (sound,key,text,scene,sprite,music,anim,sfx):
    count u16, then per atom: id u16 (+ for text: len u8 + utf8 bytes)
entity schemas: per entity: name_hash u32, nfields u8, max_live u16;
    per field: name_hash u32, ty u8 (0=fix,1=uN,2=sN,3=bool,4=ang),
    bits u8 (fixed=32,bool=1,ang=16), cold u8, default u32
globals: same field encoding (all dense)
sprites: per sprite: nverts u16, token_base u16; per vertex: 14B
    (i16 x, i16 y, u16 u, u16 v, u32 rgba, u16 tok)
music: per track: bpm u16, per voice(4): nsteps u8, vol u8, rows u16[]
    (note<<4 | vol nibble; 0xFFF note=off, 0xFFE=hold)
sfx: per program: wave u8, freq u16, sweep i16, decay u8, vol u8
anim: per curve: nkeys u8, per key: (t u16, p0 i32, m0 i32, p1 i32, m1 i32)
code section:
    fn table: per fn: entity_param u16 (0xFFFF=none), nlocals u8,
              code_offset u32, code_len u32
    table dispatch: per table: n u8, entries: (u8 key, u16 fn_idx)
    scene map: per scene atom: fn_idx u16
    huffman table: nleaf u16, 256 entries of (symbol u8, len u8) canonical
    code stream: huffman-packed opcodes + fixed-width operands, bit-LSB-first
data pool: u32 constants
```
Opcode stream: each instruction = Huffman-coded opcode byte (canonical,
max code len 8 → single 256-entry decode LUT) followed by fixed-width operand
bytes (u8/u16/u32 immediates as declared per opcode). NOT byte-aligned —
operands are bit-packed LSB-first too (true bitstream compaction).

## Opcodes (u8 symbol space; typed ops)
```
0x00 NOP
0x01 PUSH_S8   s8          0x02 PUSH_S32  pool_idx u16
0x03 LD_LOCAL  u8          0x04 ST_LOCAL  u8
0x05 LD_GLBL   field u16   0x06 ST_GLBL   field u16
0x07 LD_ENT    slot u8, field u16   0x08 ST_ENT slot u8, field u16
0x09 ADD_F 0x0A SUB_F 0x0B MUL_F 0x0C DIV_F     (Q24.8)
0x0D ADD_I 0x0E SUB_I 0x0F MUL_I 0x10 DIV_I 0x11 MOD_I
0x12 AND_I 0x13 OR_I 0x14 XOR_I 0x15 SHL 0x16 SHR
0x17 AND_B 0x18 OR_B 0x19 NOT_B 0x1A NEG
0x1B EQ_F 0x1C NE_F 0x1D LT_F 0x1E GT_F 0x1F LE_F 0x20 GE_F
0x21 EQ_I 0x22 NE_I 0x23 LT_I 0x24 GT_I 0x25 LE_I 0x26 GE_I
0x27 SIN 0x28 COS                     (ang→fixed, LUT)
0x29 RAND_MAX s32                     (0..max-1)
0x2A ANIM curve u16                   (t on stack)
0x2B KEY atom u16                     (→bool)
0x2C JMP rel_s16  0x2D JZ rel_s16  0x2E JNZ rel_s16
0x2F FOR_BGN type u8, slot u8  0x30 FOR_ADV rel_s16
0x31 SPAWN type u8, ninit u8, (field u16 × ninit)  (inits popped R→L)
0x32 KILL slot u8
0x33 SFX atom u16   0x34 MUSIC atom u16   0x35 STOP_MUSIC
0x36 GOTO atom u16
0x37 DRAW spr u16   (x,y,rot,scale,rgba on stack)
0x38 DRAW_TEXT atom u16 (x,y,rgba)   0x39 DRAW_NUM (val,x,y,rgba)
0x3A CALL_TBL table u8, slot u8      (key from slot's field on stack)
0x3B COUNT type u8
0x3C RET
0x3D CALL_FN fn u16, nargs u8        (args popped into callee locals L-R)
0x40 SHAKE amp u8                    (world-layer offset, decays)
-- v7: camera / persistence / aim math / music volume --
0x41 CAMERA          (y,x on stack, fixed: viewport center, engine-clamped)
0x42 MUSIC_VOL vol u8 (0..16 master tracker volume, 16 = unity)
0x43 SAVE            (v,ix on stack: SRAM slot ix&63 <- raw 32-bit; sets dirty)
0x44 SAVED           (ix on stack -> push slot value, raw i32)
0x45 DIST            (x1,y1,x2,y2 fixed on stack -> Q24.8 distance, i64+isqrt)
0x46 ATAN2           (dy,dx fixed on stack -> screen-convention ang, CORDIC)
0x47 CAM_X           (-> viewport top-left x, fixed)   0x48 CAM_Y
```

## Runtime memory map (flat, profile-sized — default 2MB; wasm + native identical)
```
0x0000_0000  u32 input bitmask (JS writes keydown/keyup bits here)
0x0000_0004  u32 control: frame_no | audio_occupancy etc.
0x0000_0008  u32 rand_state
0x0000_000C  u32 scene_id
0x0000_0050  u32 screen-shake amplitude (decays ~25%/frame)
0x0000_0054  i32 camera top-left x (Q24.8, v7; 0 = v1 behavior)
0x0000_0058  i32 camera top-left y (Q24.8, v7)
0x0000_0010  globals block (bit-packed per schema)
0x0000_0100  entity meta: per type: gen[], bitmap u64[]  (dense)
0x0001_0000  dense SoA rows per entity type (row = slot index)
0x0007_0000  sparse cold storage (open-address u32->row)
0x0008_0000  vertex stream: u32 count + 14B vertices (max 6000)
0x0010_0000  audio: voices, tracker state, 256B ADPCM ring, decode staging
0x0011_0000  code+data loaded from .libyte (immutable after init)
0x0018_8000  SRAM: 64 x u32 save slots (v7; init's zeroing pass SKIPS it)
0x0020_0000  end (2MB)
```
Native-only: framebuffer 512×512×4 = separate static (not in 2MB budget,
web path renders the vertex stream via WebGL instead).

## Wasm ABI (wasm32-freestanding, exports)
```
memory                 (exported via -rdynamic recipe)
lila_init(ptr,len) -> i32        0 = ok, negative = typed error (below)
lila_frame() -> void             one 60Hz tick: systems + stream build
lila_version() -> i32            3 (v7 ABI — additive exports only)
lila_stream_ptr() -> i32         vertex stream address
lila_stream_count() -> i32
lila_audio_fill(dst,len) -> i32  decode ADPCM ring → i16 PCM, ret samples
lila_random_seed(s) -> void      (optional determinism control)
-- v6: build-profile introspection + diagnostics --
lila_mem_size() -> i32           total flat engine memory (bytes)
lila_cap_ent() -> i32            entity slots per type (build profile)
lila_cap_verts() -> i32          per-frame vertex budget
lila_cap_instr() -> i32          decoded instruction capacity
lila_live_ents() -> i32          live entities right now (all types)
lila_stream_dropped() -> i32     cumulative verts dropped (budget)
lila_spawn_denied() -> i32       cumulative spawns denied (pool full)
lila_last_error() -> i32         loader rc of the most recent load
-- v7: SRAM persistence + world introspection --
lila_sram_ptr() -> i32           SRAM block address (copy saved bytes here
                                 BEFORE lila_init; init's zeroing skips it)
lila_sram_dirty() -> i32         1 iff save() since last call (read-clear)
lila_cap_types() -> i32          entity types this build supports
lila_world_w() -> i32            loaded game's world width (px)
lila_world_h() -> i32            loaded game's world height (px)
```

### Loader return codes
```
 0 ok            -1 bad magic    -2 bad version  -3 blob too big
-4 truncated     -5 bad huffman  -6 bad header
-7 LIMIT entities (game capacity > build MAX_ENT)   -8 LIMIT fns
-9 LIMIT tables  -10 LIMIT scenes -11 LIMIT texts -12 LIMIT sprites
-13 LIMIT sprite verts (>128) -14 LIMIT music -15 LIMIT sfx
-16 LIMIT anims  -17 LIMIT typed fields -18 LIMIT pool -19 LIMIT instrs
-20 LIMIT spawn tf-list pool -21 LIMIT entity types (> build max_types)
```
Every limit error is LOUD: the runner prints a readable name and exits
nonzero; the web host maps the code to an on-page message. v1 silently
clamped most of these — that behavior is gone by design.

### Engine build profiles (v6/v7)

The flat memory map is derived at comptime from
`runtime/src/engine_config.zig` (generated by `scripts/build_engine.sh`):
profiles `default` (2MB / 16 types / 512 ent / 6000 verts / 8192 instrs —
the v1 values), `big` (4MB / 32 types / 2048 / 12000 / 12288 …), `huge`
(8MB / 64 types / 4096 / …), or custom `key=value` overrides (knobs:
`mem_mb max_types max_ent max_verts max_instr max_sprites max_texts
max_anims max_fns max_pool max_tracks max_sfx`). A profile whose map
outgrows its declared memory fails at compile time. At the default profile
the derived map is byte-identical to the v1 fixed map (asserted in
`runtime/src/test_map.zig`), except ADDR_COLDGEN: 0x50000 → 0x58000 (v1
reserved it inside the cold-rows region by a size miscalculation; the fix
is symbolic-access-safe) and the v7 SRAM extension past audio
(MAP_END 0x188000 → 0x188100 — additive, no prior region moved).

## Demo game (wrap-shooter) acceptance criteria
- 512×512 wrap world; player ship rotates/thrusts (sin/cos LUT), fires bullets.
- Enemy waves (3 behaviors via table dispatch), anim-curve sine motion.
- Particles on kills; powerups (+score); lives; score draw via draw_num.
- Sounds: shoot/explosion/pickup (procedural sfx); music loop (tracker).
- Scenes: #play / #over; goto transitions; restart on fire.
- Determinism: same input script → identical PNG frames + WAV bytes.
```

FILE: /home/z/my-project/lila-engine/docs/SPEC.md (internal working spec)
