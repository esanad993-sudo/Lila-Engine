# The LILA Language — Reference (v7)

LILA (**L**ightweight **I**nteger **L**anguage for **A**rcade) is a strict,
compile-time-compiled DSL for deterministic 2D arcade games. A game is one
`.lila` file; the compiler (`lilac`) turns it into one `.libyte` binary that
runs identically on the native (Zig) and WebAssembly runtimes.

## Philosophy

- **Integers only.** All spatial math is Q24.8 fixed-point (`i32`, 8
  fractional bits). `1.5` compiles to the integer `384`. This buys
  bit-perfect cross-platform determinism and branch-free wrapping.
- **Names are compile-time.** Resources are `#atoms` — interned, indexed,
  and gone after compilation. The only strings in a game are `text {}`
  labels.
- **Data layout is the compiler's job.** You declare fields with widths
  (`hp: u4`); the compiler computes bit offsets, dense/cold placement, and
  emits the integration code.

## Source layout

Declarations may appear in any order (fns must be defined before referenced
by `scene`/`table` blocks at check time — order in the file is otherwise
free):

```lila
game { title: "MY GAME"  width: 512  height: 512  wrap: true
       world_w: 4096  world_h: 512          // v7: scrolling world (optional)
       capacity { entities: 2048 } }
key     { #left, #right, #up, #fire }
sound   { #shoot, #boom }
text    { #score_label: "SCORE", #over: "GAME OVER" }
scene   { #play: start_play, #over: start_over }
sprite  #ship (12) { poly [ 0,-12, 8,10, 0,5, -8,10 ] fill: 0x66CCFF grad: 0x3366AA }
music   #track { bpm: 150
                 square: [ C4 . E4 . G4 . E4 . ]
                 tri:    [ C2 . . . G1 . . . ]
                 noise:  [ . . x . . . x . ] }
sfx     #shoot { wave: square  freq: 880  sweep: -1200  decay: 10  vol: 10 }
anim    #bob   { keys: [ (0,0), (32,80), (64,0) ] }
global  { score: u16 = 0  lives: u = 3  px: fixed = 256 }
entity  Bullet { x: fixed  y: fixed  vx: fixed  vy: fixed  ttl: u8 = 60
                 cold kind: u2 = 0 }
table   behaviors { 0: drift, 1: sine, 2: chase }
fn init() { spawn Player; goto(#play); }
```

### `capacity` (v6): declare what the game needs

`game.capacity.entities` is the number of live-entity slots per type the
game is built for (default 512, range 16..65535). It is written into a v2
`.libyte` header and checked at load: an engine build whose profile has
fewer slots rejects the file with a typed error instead of silently
clamping spawns. Build the engine to match with
`scripts/build_engine.sh big` (2048 slots) / `huge` (4096) or custom
`max_ent=...` overrides — see the repo README, *Engine profiles*.

### `world_w` / `world_h` (v7): scrolling worlds

The screen (`width`/`height`) is the VIEWPORT; the world is the simulation
bounds. Declare a world larger than the screen and drive the viewport with
`camera(x, y)` — positions, wrap masks, spawns, everything stays in world
coordinates; the renderer subtracts the camera and the starfield parallaxes
against it. Defaults: world == screen (a fixed-camera game — byte-compatible
with v1/v6 files). Limits: 16..32768 per axis; power-of-two per axis when
`wrap: true`.

### `world_w` / `world_h` (v7): scrolling worlds

The screen (`width`/`height`) is the VIEWPORT; the world is the simulation
bounds. Declare a world larger than the screen and drive the viewport with
`camera(x, y)` — positions, wrap masks, spawns, everything stays in world
coordinates; the renderer subtracts the camera and the starfield parallaxes
against it. Defaults: world == screen (a fixed-camera game — byte-compatible
with v1/v6 files). Limits: 16..32768 per axis; power-of-two per axis when
`wrap: true`.

## Types

| Type | Meaning | Storage |
|---|---|---|
| `fixed` | Q24.8 fixed-point (i32, `>> 8`) | 32 bits |
| `uN` / `sN` (N = 1..16) | bit-packed integer | N bits |
| `u` / `s` | width inferred from initializer (min 4 bits) | inferred |
| `bool` | flag | 1 bit |
| `ang` | angle: u16, 4096 units = 360° | 16 bits |

Rules: `fixed ⊕ fixed → fixed`; `int ⊕ int → int`; **int widens to fixed**
implicitly in mixed arithmetic, comparisons, and assignments (like C's
`int → double`). `bool` only combines via `&& || !` and comparisons. `ang`
converts with `ang(x)`; `sin`/`cos` take `ang` and return `fixed`.

Literals: `42`, `0x3AF` (u32 ints); `1.5`, `-0.25` (fixed, max 2 decimals);
`true` / `false`. Hex colors are `0xRRGGBB` in sprite fills (alpha is baked
opaque) and `0xRRGGBBAA` in draw calls.

## #Atoms

`#name` is a compile-time resource symbol. Each domain keeps its own index
table; compiled code references positions, never strings:

| Domain | Declared in | Used by |
|---|---|---|
| key | `key {}` | `key(#x)` |
| sound | `sound {}` | `sfx(#x)` |
| text | `text {}` | `draw_text(#x, …)` |
| scene | `scene {}` | `goto(#x)` |
| sprite | `sprite #x …` | `draw(#x, …)` |
| music | `music #x …` | `music(#x)` |
| anim | `anim #x …` | `anim(#x, t)` |

Using an atom in the wrong domain, or an undeclared one, is a compile error.

## Entities & globals

```lila
entity Enemy {
    x: fixed = 0          // dense row, 32 bits
    y: fixed = 0
    hp: u = 3             // inferred u4 (fits 0..3, min 4 bits)
    behavior: u8 = 0
    rot: ang = 0
    cold seed: u8 = 0     // seldom-read: cold side storage
}
```

- Field names are globally unique across all entities (they become global
  field IDs).
- Entities with `x`/`y` **and** `vx`/`vy` fixed fields get compiler-emitted
  auto-integration: `pos = (pos + vel) & wrap_mask` once per frame (the mask
  is `(dim<<8)-1` when `game.wrap`, else unmasked). You never write the
  movement loop yourself.
- Globals work the same way but live in the flat buffer at a
  compile-time-known offset — hosts can live-patch them at runtime.
- Handles are `u32`: bits 0–19 slot index, bits 20–31 generation. Stale
  references fail their generation check and read as zeros — no crashes,
  no null checks.

## Functions

```lila
fn init() { ... }            // runs once at startup
fn update() { ... }          // runs every frame (global logic)
fn update(e: Enemy) { ... }  // runs once per live Enemy (declaration order)
fn draw() { ... }            // runs every frame AFTER entities (HUD)
fn draw(e: Enemy) { ... }    // per live entity, render pass
fn helper(x: fixed, y: fixed) { ... }   // call via `call helper(x, y);`
```

- `update`/`draw` may **overload by entity type** — one `update` per entity.
- Draw fns may only contain `draw*`, `sfx`, `music`, `stop_music`
  statements (checker-enforced).
- Scene fns (referenced in `scene {}`) are 0-arg helpers fired by `goto`.
- Table fns take exactly one entity param; dispatched via
  `call behaviors[e.behavior](e);`.
- `return;` exits early. Locals rebind via `let` (Python-style).

## Statements

```
let n = 1 + wave;                    // local
target = expr;  target += e;  -=  *= // assignment (fields, globals, locals)
if cond { } else if cond { } else { }
while cond { }
for (b: Bullet) { ... }              // iterate live entities (slot order)
spawn Enemy { x: 256, y: 32 };       // fields evaluated in written order
kill e;                              // frees the slot (gen++)
sfx(#shoot);   music(#track);   stop_music();
music_vol(8);                        // v7: master music volume 0..16 (16 = full)
shake(5);                             // screen shake, amplitude 0..255 px (literal)
camera(px + 110, py);                // v7: aim the viewport center at a world point
save(0, best);                       // v7: write SRAM slot 0..63 (raw 32-bit)
goto(#play);                         // switch scene + run its fn
draw(#ship, e.x, e.y, e.rot, 1.0, 0xFFFFFFFF);
draw_text(#score_label, 8, 8, 0x88CCFFFF);
draw_num(score, 64, 8, 0xFFFFFFFF);
call boom(ex, ey);                   // helper (0-2 args)
call behaviors[e.behavior](e);       // table dispatch
return;
```

Screen convention: rotation `0` faces **up**; `vx = sin(rot)`,
`vy = -cos(rot)`. Draw x/y/scale take fixed-point values; rot takes `ang`.

**Camera semantics (v7):** `camera(x, y)` sets the viewport CENTER in world
coords; the engine clamps it so the viewport never leaves the world (a
world == screen game pins it to offset 0 — identical to no camera at all).
`cam_x` / `cam_y` read back the viewport TOP-LEFT (world coords, fixed) —
use them for spawn-relative-to-view math (`spawn Cell { x: cam_x + 700 }`).
They are engine registers: declaring a variable of those names is an error.

## Engine services

Services the runtime provides without game code:

- **Starfield** — 60 hashed stars in 3 parallax layers (twinkling,
  additive, scrolling) rendered behind all game draws. No setup; it is
  always on.
- **Screen shake** — `shake(n)` bumps a decaying amplitude register
  (~25% decay per frame, overlapping shakes take the max). Only the world
  layer offsets — HUD text and the starfield stay fixed.
- **Camera (v7)** — `camera(x, y)` clamps the viewport inside the world;
  world-space draws are translated at emit time so every host (WebGL2,
  WebGL1, software rasterizer) renders scrolling worlds unchanged. The
  starfield layers scroll against the camera at depths 1/8, 1/4, 1/2.
- **SRAM (v7)** — 64 raw 32-bit save slots that survive re-init and
  restarts (battery-backed semantics); hosts persist them to localStorage
  (web) or files (native `--sram`).

## Expressions

```
key(#left)            // -> bool (input bitmask test)
sin(a)  cos(a)        // ang -> fixed (LUT)
atan2(dy, dx)         // v7: -> ang, screen convention (0 = up, clockwise) — CORDIC
dist(x1, y1, x2, y2)  // v7: -> fixed px, i64 internals (cannot overflow)
saved(ix)  savedf(ix) // v7: SRAM slot as int / as fixed (raw 32-bit bits)
ang(x)  fixed(x)  int(x)   // casts
rand(N)               // -> int in 0..N-1 (deterministic LCG)
anim(#bob, t)         // -> fixed (Hermite curve at frame t, wraps)
count(Enemy)          // -> int (live entity count)
cam_x  cam_y          // v7: viewport top-left (fixed, world coords)
```

Operator precedence (low → high): `||` < `&&` < comparisons <
`& | ^` < `<< >>` < `+ -` < `* / %` < unary `- !` < `.` field access.
Integer overflow wraps (two's complement) — the wrap-around world relies on
it.

## Sprites

```lila
sprite #ship (12) {                  // (12) = hit radius in px
    poly [ 0,-12, 8,10, 0,5, -8,10 ] // closed polygon, pixel coords,
                                     // origin = sprite center
        fill: 0x66CCFF               // flat RGB
        grad: 0x3366AA               // per-vertex color lerp
        add: true                    // additive blend token
        pat: noise;                  // noise dither token (or flat)
}
```

The compiler fans the polygon into triangles and quantizes each vertex to
14 bytes. Sprites render with infinite mathematical crispness at any scale —
they are geometry, not bitmaps.

## Audio

**Music** — a tracker: up to 4 voices × 16 steps. Notes `C1`..`B8`, `-`
hold, `.` silence, `x` noise hit; optional `vol: 0..15` per voice.
**SFX** — one-line synth programs: `wave` (square/tri/noise/saw), `freq`
(Hz), `sweep` (Hz/s), `decay` (in 1/60 s), `vol`. The mixer runs pure
integer adds through a 256-byte IMA-ADPCM ring (4:1 vs 16-bit PCM).

## Anim curves

```lila
anim #bob { keys: [ (0,0), (32,80), (64,0) ] }
```

Keyframes are `(frame, fixed-value)`. The compiler simplifies the polyline
(RDP) and bakes cubic Hermite control points; runtime evaluates in O(#segs)
integer math and wraps at the last key time.

## v11/v12 — Arrays, 3D math, indexed meshes

### Arrays (v11)

```lila
global { arr fixed[1800] terr  arr u8[64] vb }   // game-wide arrays
entity Orb { arr fixed[8] trail }                 // per-instance arrays
```

Fixed-capacity, zero-alloc i32-cell aggregates. Literal indices are
bounds-**proven** at compile time; dynamic indices wrap (power-of-two caps
mask in one AND, others modulo) — never a trap. `alen(name)` returns the
compile-time capacity. Cells are raw values: `fixed`/`u8`/… type the
element for checking; storage is one flat plane.

### 3D camera + projection (v11)

```lila
cam3(x, y, z, yaw, pitch);              // position the 3D camera (page-0 regs)
let s = proj3(x, y, z);                 // -> scale (focal / view depth)
draw(#sprite, projx(), projy(), 0, s * 0.1, rgba);   // billboard through cam3
if (projok()) { ... }                   // last proj3 was in front
```

Convention: +X right, +Y forward, +Z up; yaw 0 looks along +Y; positive
pitch looks up; fov 90. All Q24.8 integer math + the shared ang LUT —
bit-exact on every runtime, snapshot/rewind-safe (registers live in page 0).

### mat4 / quat builtins over arrays (v12)

A **mat4** is 16 consecutive cells of a global array, column-major
(`m[c*4 + r]`, translation in cells 12..14). A **quat** is 4 consecutive
cells `(x, y, z, w)`, right-handed: positive angle rotates counter-clockwise
viewed from the +axis. Each statement is ONE native opcode — the game
orchestrates the rig, the engine crunches the math:

```lila
quat_aa(lq, 0, 0.0, 1.0, 0.0, ang(int(sin(t) * 60)));  // axis-angle quat (axis normalized here)
qmul(qd, 0, qa, 0, qb, 0);                             // d = a*b (Hamilton, alias-safe)
m4qt(wm, 0, lq, 0, tx, ty, tz);                        // M = T(t) * R(q) in one op
m4mul(wm, 16, wm, 0, lm, 0);                           // D = A*B (world = parent * local)
skinv(rv, v * 3, wm, bone * 16, bx, by, bz);           // verts = M * (x, y, z, 1)
```

Rules: all arrays are GLOBAL (rig storage is shared); offsets are int
expressions (literal offsets are bounds-proven: `+4` for quats, `+16` for
mats); products accumulate in i64 and truncate `>> 8` once. Gameplay-side
only — rejected in draw fns.

### Indexed mesh pass (v12)

```lila
draw3di(verts, nv, idx, ni, x, y, z, yaw, rgba);   // draw-pass only
```

The GPU vertex/index-buffer vocabulary: `verts` holds 3 fixed cells per
UNIQUE vertex (animated in place by `skinv`), `idx` holds vertex indices,
3 per triangle. The engine transforms/views/projects/backface-culls/
near-clips/depth-sorts/rasterizes exactly like `draw3d` — triangles address
shared vertices, so an articulated model animates without touching its
topology. Budgets: nv <= 768, ni <= 768 (256 tris), literal counts proven
against the array capacities; out-of-range runtime indices drop the tri.

### Z-buffer profile knob (v12)

The 3D rasterizer has two variants: the v11 **painter's sort** (default,
zero extra memory) and a per-pixel **Z-buffer** (correct triangle-triangle
occlusion for interpenetrating geometry, +512KB depth buffer). Engine
config, not game state:

- `scripts/build_engine.sh zbuffer3d=1` — bake the z-buffer as the build's
  default profile (like max_ent);
- `runner game.libyte --zbuffer` / `--no-zbuffer` — per-run override.

In z mode each 3D tri ships a 12-bit view depth in the vertex token
(bit 3 = flag, bits 4..15 = depth); the rasterizer depth-tests per pixel,
additive tris test but never write (glows don't occlude), and 2D content
keeps painter's semantics. Each mode is fully deterministic.

### Frame locals limit (v12)

A fn may use 32 local slots (params + `let` bindings). Needing more is a
BUILD ERROR ("needs N local slots") — the old silent behavior (writes past
the frame dropped, reads returned 0) hung loops instead of failing loudly.

## Determinism

Given the same input script, the native runner and the wasm host produce
identical frame streams and audio samples: LCG rand seeded in the flat
buffer, Q24.8 math, LUT trig, integer audio. The 60 Hz fixed-step web loop
is display-rate independent (accumulator-based, max 3 steps/frame).

## Toolchain

```
lilac build  <in.lila> [-o out.libyte]   # compile
lilac disasm <in.libyte>                 # dump + validate the binary
lilac watch  <in.lila> [-o out.libyte]   # rebuild on save (hot reload)
```

Runtime CLIs: `runner <game.libyte> [--frames N] [--watch] [--sram FILE]`
(native — `--sram` persists save slots to a file), `wasm.zig` → `lila.wasm`
+ `web/index.html` (browser — SRAM persists to localStorage).
