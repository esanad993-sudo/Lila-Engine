//! ============================================================================
//! LILA OPTIMIZER PIPELINE — the compile-time metaprogramming core.
//! ============================================================================
//!
//! EVERYTHING IN THIS TREE RUNS WHEN THE DEVELOPER WRITES/COMPILES THE CODE
//! (`lilac build` / `lilac shader` / `lilac wavfit`). The player's runtime is
//! a pre-baked executor: it performs ZERO analysis, ZERO fitting, ZERO layout
//! — it only executes the finished math the compiler baked into the .libyte
//! and the emitted artifacts.
//!
//! Pipeline order (driven by main.rs::compile_pipeline):
//!
//!   [parse] ──> [checker] ──> OPT PASSES ──> [codegen] ──> [huffman+libyte]
//!                                  │
//!    opt::ranges   subsystem 1a    global value-range analysis ->
//!                                  minimum bit-widths (u8 declared, u2 needed)
//!    opt::soa      subsystem 1b    cache-line SoA synthesis: zero-hole
//!                                  packing, hot fields into the first 16
//!                                  bytes (one 128-bit register window),
//!                                  physics+render data share the cache line;
//!                                  emits the comptime Zig manifest.
//!    opt::physics  subsystem 3     inequality transformation: Chebyshev
//!                                  guards on dist(), Galois 32-lane collision
//!                                  bitmasks, swept interval tests.
//!    opt::fsm      subsystem 5     AI bit-plane synthesis: state machines ->
//!                                  transition tables evaluated with one
//!                                  shift+mask per entity, branchless.
//!    opt::audio    subsystem 4     (asset phase, before checker) .wav PCM ->
//!                                  O(1) f(t) synthesis programs at build time.
//!    opt::sdf      subsystem 2     (shader phase, separate CLI verb) CSG ->
//!                                  fold + smin-consolidate + autodiff ->
//!                                  branchless WGSL sphere tracer.

pub mod sdf;
pub mod ranges;
pub mod soa;
pub mod physics;
pub mod fsm;
pub mod audio;
pub mod wgsl;
pub mod select;
pub mod arena;

/// Optimizer build report — printed by `lilac build` so the dev SEES what the
/// compiler did to their "naive" code (the unoptimized -> elite story).
#[derive(Default)]
#[derive(Debug)]
pub struct OptReport {
    /// subsystem 1: (field, declared width, synthesized width) narrowings
    pub narrowed: Vec<(String, u8, u8)>,
    /// subsystem 1: bytes saved per entity row by zero-hole repacking
    pub row_bytes_saved: Vec<(String, u16, u16)>, // (entity, before, after)
    /// subsystem 1: hot-register-segment description per entity
    pub hot_segments: Vec<(String, u32, Vec<String>)>, // (entity, bytes, hot fields)
    /// subsystem 3: physics rewrites (site, what)
    pub physics: Vec<(String, String)>,
    /// subsystem 6: branchless SEL rewrites (conditional diamonds + cond-moves)
    pub sel_moves: usize,
    /// subsystem 7: static spawn budget (entity, proven, capacity, has-dynamic)
    pub arena: Vec<(String, u32, u32, bool)>,
    /// subsystem 7: exact entity-region bytes (Σ live·(row+cold))
    pub arena_bytes: u32,
    /// subsystem 8: schedule proof (systems, RAW edges served, WAR orderings)
    pub schedule: Option<(usize, usize, usize)>,
    /// subsystem 10: compile-proven parallel systems (members, groups)
    pub parallel: Option<(usize, usize)>,
    /// subsystem 11: language synthesis (prefabs + particles) — (kind, detail)
    pub synth: Vec<(String, String)>,
    /// subsystem 9: dictionary synthesis on the text payload (raw, packed)
    pub text_pack: Option<(usize, usize)>,
    /// subsystem 5: fsm synthesis (name, states, table bytes, minimization note)
    pub fsms: Vec<(String, usize, usize, String)>,
    /// subsystem 4: audio fits (atom, model, residual %)
    pub audio: Vec<(String, String, f32)>,
    /// subsystem 2: shader synthesis notes (emitted by `lilac shader`)
    pub shaders: Vec<(String, String)>,
}

impl OptReport {
    pub fn print(&self) {
        if self.narrowed.is_empty() && self.physics.is_empty() && self.fsms.is_empty()
            && self.audio.is_empty() && self.row_bytes_saved.is_empty()
        {
            return; // legacy game: optimizer found nothing to do — silence is golden
        }
        println!("  optimizer:");
        if !self.narrowed.is_empty() {
            println!("    [1] bit-width synthesis:");
            for (f, d, w) in &self.narrowed {
                println!("        {} : u{} -> u{}", f, d, w);
            }
        }
        for (e, before, after) in &self.row_bytes_saved {
            println!("    [1] SoA row {}: {}B -> {}B (zero-hole, hot/cold split)", e, before, after);
        }
        for (e, bytes, fields) in &self.hot_segments {
            println!("    [1] {} register window: {}B [{}]", e, bytes, fields.join(", "));
        }
        for (site, what) in &self.physics {
            println!("    [3] {} -> {}", site, what);
        }
        if self.sel_moves > 0 {
            println!("    [6] superopt: {} conditional assignments -> branchless SEL (no JZ/JMP)", self.sel_moves);
        }
        for (name, proven, cap, dynamic) in &self.arena {
            let tail = if *dynamic { " + dynamic respawns (runtime-guarded)" } else { "" };
            println!("    [7] arena: {} spawn budget {}/{} proven{} — 0 runtime denials possible", name, proven, cap, tail);
        }
        if self.arena_bytes > 0 {
            println!("    [7] arena: entity region {}B of 2MB flat — allocation-free by construction", self.arena_bytes);
        }
        if let Some((systems, raw, war)) = &self.schedule {
            println!("    [8] schedule: {} systems, {} RAW dataflow edges + {} WAR orderings — single-worker topological order, 0 locks/0 barriers", systems, raw, war);
        }
        if let Some((members, groups)) = &self.parallel {
            println!("    [10] parallel: {} update systems in {} proven-disjoint group(s) — concurrent workers, 0 scheduler cost (R/W contracts proven at build time)", members, groups);
        }
        for (kind, detail) in &self.synth {
            println!("    [11] synth {}: {}", kind, detail);
        }
        if let Some((raw, packed)) = &self.text_pack {
            let pct = 100.0 - (*packed as f64 / *raw as f64 * 100.0);
            println!("    [9] dictionary: texts {}B -> {}B entropy-coded ({:.1}% smaller, per-game canonical Huffman)", raw, packed, pct);
        }
        for (name, states, bytes, note) in &self.fsms {
            println!("    [5] fsm {}: {} states, {}B bit-plane table — {}", name, states, bytes, note);
        }
        for (atom, model, res) in &self.audio {
            println!("    [4] sfx #{}: fitted {} (residual {:.1}%)", atom, model, res * 100.0);
        }
        for (name, what) in &self.shaders {
            println!("    [2] shader {}: {}", name, what);
        }
    }
}
