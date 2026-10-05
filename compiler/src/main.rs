//! lilac — the LILA compiler CLI.
//! subcommands: build, disasm, watch, shader, wavfit
//!
//! v8 OPTIMIZER PIPELINE (all of it DEV-TIME — the player never runs any of
//! this; the runtime executes only the finished math baked into the .libyte):
//!
//!   [lex/parse] -> [AUDIO: .wav -> O(1) f(t) fits] -> [checker]
//!     -> opt::ranges  (minimum bit-width synthesis, `optimize: true`)
//!     -> opt::soa     (cache-line SoA repack + Zig comptime manifest)
//!     -> opt::physics (Chebyshev guards + Galois 32-lane collision masks)
//!     -> opt::fsm     (bit-plane transition tables + branchless expansion)
//!     -> [codegen] -> [huffman + .libyte v3]
//!
//! LEGACY CONTRACT: a game that uses NONE of the new features (no
//! `optimize: true`, no `fsm` blocks, no `sfx ... from:`) takes NO passes and
//! produces byte-identical .libyte output to the pre-optimizer compiler.

mod lexer;
mod ast;
mod parser;
mod checker;
mod codegen;
mod huffman;
mod libyte;
mod disasm;
mod diag;
mod opt;
mod synth;

use std::time::Instant;

fn read_source(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("cannot read '{}': {}", path, e))
}

/// Value of a `flag value` CLI argument. The old code indexed
/// `args[i + 1]` unchecked — `lilac build x.lila -o` as the last argument
/// PANICKED with "index out of bounds". Now it's a usage error.
fn flag_value(args: &[String], flag: &str) -> Result<Option<String>, String> {
    match args.iter().position(|a| a == flag) {
        Some(i) => {
            if i + 1 >= args.len() {
                return Err(format!("flag {} needs a value (it was the last argument)", flag));
            }
            Ok(Some(args[i + 1].clone()))
        }
        None => Ok(None),
    }
}

#[derive(Debug)]
pub struct BuildReport {
    pub source_bytes: usize,
    pub libyte_bytes: usize,
    pub code_uncompressed: usize,
    pub code_huffman: usize,
    pub opt: opt::OptReport,
    /// file format version emitted (1, 2 or 3)
    pub version: u16,
}

/// One full compile: source text -> .libyte bytes. The optimizer passes run
/// here, gated on the game's opt-in flags (see the module doc).
/// `base_dir` resolves asset paths (`sfx ... from:`) relative to the SOURCE
/// file, so builds work from any working directory.
fn compile_to_libyte(src: &str, zig_opt_path: Option<&str>, base_dir: Option<&str>) -> Result<(Vec<u8>, BuildReport), String> {
    let (toks, atom_names) = lexer::Lexer::new(src).tokenize()?;
    let mut p = parser::Parser::new(toks, atom_names);
    let mut prog = p.parse_program()?;
    let atom_names = std::mem::take(&mut p.atom_names);

    // ---- v10 LANGUAGE SYNTHESIS (before checking) — prefabs + particles
    // lower to plain entities/fns; the rest of the pipeline never knows.
    let mut synth_rows: Vec<(String, String)> = Vec::new();
    if !prog.entities.iter().any(|e| e.from.is_some()) || !prog.particles.is_empty() {
        synth_rows = synth::expand(&mut prog, &atom_names)?;
    }

    // ---- SUBSYSTEM 4 (audio): fit every `sfx #x { from: "y.wav" }` NOW, at
    // build time. The .wav is a dev asset: it never ships, and the runtime
    // only ever sees the fitted O(1) synthesis parameters in the bytecode.
    let mut fitted_audio: std::collections::HashMap<u16, opt::audio::FittedSfx> =
        std::collections::HashMap::new();
    let mut audio_rows: Vec<(String, String, f32)> = Vec::new();
    for sd in &prog.sfx {
        if let Some(path) = &sd.from {
            let name = atom_names.get(sd.atom as usize)
                .map(|s| s.as_str()).unwrap_or("sfx").to_string();
            // asset paths are relative to the SOURCE file (then CWD)
            let resolved: String = {
                let cand = match base_dir {
                    Some(d) if !d.is_empty() => format!("{}/{}", d, path),
                    _ => path.clone(),
                };
                if std::path::Path::new(&cand).exists() { cand } else { path.clone() }
            };
            let bytes = std::fs::read(&resolved)
                .map_err(|e| format!("sfx #{}: cannot read '{}': {}", name, resolved, e))?;
            let pcm = opt::audio::parse_wav(&bytes)
                .map_err(|e| format!("sfx #{} ({}): {}", name, resolved, e))?;
            let fit = opt::audio::fit(&name, &pcm)
                .map_err(|e| format!("sfx #{} ({}): {}", name, resolved, e))?;
            if fit.residual > 0.6 {
                eprintln!("warning: sfx #{}: fit residual {:.1}% is high — the baked f(t) will be a loose approximation of '{}'",
                    name, fit.residual * 100.0, resolved);
            }
            audio_rows.push((name, fit.equation.clone(), fit.residual));
            fitted_audio.insert(sd.atom, fit);
        }
    }

    let mut ctx = checker::check(&prog, atom_names, &fitted_audio)?;

    // ---- OPT PASSES ----
    // Gate: legacy games (no optimize flag, no fsms) run NOTHING and keep
    // byte-identical output. `optimize: true` enables the schema/physics
    // passes; `fsm` blocks always run their pass (the bytecode would be
    // unexecutable otherwise — FsmStep must be expanded before codegen).
    let mut report = opt::OptReport::default();
    let mut fsm_tables: Vec<Vec<u8>> = Vec::new();

    if ctx.game.optimize {
        // [1a] minimum bit-width synthesis over every assignment site
        let r = opt::ranges::narrow(&mut ctx);
        report.narrowed = r.narrowed;

        // [1b] access-density SoA reorder + zero-hole repack + comptime manifest
        let r = opt::soa::synthesize(&mut ctx, zig_opt_path);
        report.row_bytes_saved = r.row_bytes_saved;
        report.hot_segments = r.hot_segments;

        // [3] physics: Chebyshev guards + Galois 32-lane collision masks
        let r = opt::physics::rewrite(&mut ctx);
        report.physics = r.rewrites;

        // [6] superoptimization: conditional assignment diamonds and
        // cond-moves -> single branchless SEL data edges (purity- and
        // type-checked by the pass; see opt::select docs)
        report.sel_moves = opt::select::rewrite(&mut ctx);

        // [7] static arena & lifetime synthesis: prove the spawn budget per
        // type (constant-trip loop inference on the boot path) and compute
        // the exact entity-region byte envelope. A proven overflow FAILS
        // the build here instead of denying spawns at runtime.
        match opt::arena::synthesize(&mut ctx) {
            Ok(a) => {
                report.arena = a.rows.iter().map(|r| {
                    (r.entity.clone(), r.proven, r.capacity, r.dynamic)
                }).collect();
                report.arena_bytes = a.entity_region_bytes;
            }
            Err(e) => return Err(e),
        }

        // [8] graph-theory schedule proof: the fixed single-worker system
        // order IS a topological order of the program's RAW dataflow DAG —
        // count the edges it serves so the claim is quantified, not asserted.
        let s = opt::arena::schedule_proof(&ctx);
        report.schedule = Some((s.systems, s.raw_edges, s.war_pairs));
    }

    if !ctx.fsms.is_empty() {
        // [5] AI: bake bit-plane transition tables + expand fsm_step into the
        // branchless event-mask form (runs whenever fsms exist — the runtime
        // has no FsmStep opcode; the pass IS the feature).
        let r = opt::fsm::expand(&mut ctx);
        report.fsms = r.report;
        fsm_tables = r.tables;
    }

    // [10] compile-proven parallel systems (subsystem 10): group the pure,
    // data-disjoint update systems so the runtime can run each group on
    // concurrent workers with ZERO scheduler cost. Optimize-gated: legacy
    // binaries keep byte-identical output (no groups section, single worker).
    let mut par_plan: Vec<Vec<u16>> = Vec::new();
    if ctx.game.optimize {
        let plan = opt::arena::parallel_groups(&ctx);
        let members: usize = plan.groups.iter().map(|g| g.len()).sum();
        if !plan.groups.is_empty() {
            report.parallel = Some((members, plan.groups.len()));
            par_plan = plan.groups.iter()
                .map(|g| g.iter().map(|&fi| fi as u16).collect())
                .collect();
        }
    }
    report.audio = audio_rows;
    report.synth = synth_rows;

    let code = codegen::gen(&ctx)?;
    let res = libyte::emit(&ctx, &code, &fsm_tables, &par_plan);
    let n = res.bytes.len();
    Ok((res.bytes, BuildReport {
        source_bytes: src.len(),
        libyte_bytes: n,
        code_uncompressed: res.code_uncompressed,
        code_huffman: res.code_packed,
        opt: {
            let mut r = report;
            if res.text_uncompressed > 0 {
                r.text_pack = Some((res.text_uncompressed, res.text_packed));
            }
            r
        },
        version: libyte::emitted_version(&ctx, &fsm_tables, !par_plan.is_empty()),
    }))
}

fn print_build_report(report: &BuildReport) {
    let ratio = if report.code_uncompressed > 0 {
        100.0 - (report.code_huffman as f64 / report.code_uncompressed as f64 * 100.0)
    } else { 0.0 };
    println!("  code: {} B raw -> {} B huffman ({:.1}% smaller)",
        report.code_uncompressed, report.code_huffman, ratio);
    report.opt.print();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: lilac <build|disasm|watch|shader|wavfit> [args]");
        std::process::exit(2);
    }
    match args[1].as_str() {
        "build" => {
            if args.len() < 3 {
                eprintln!("usage: lilac build <input.lila> [-o output.libyte] [--emit-zig-opt path.zig]");
                std::process::exit(2);
            }
            let in_path = &args[2];
            let out_path = flag_value(&args, "-o").unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(2);
            }).unwrap_or_else(|| {
                    let stem = in_path.trim_end_matches(".lila");
                    format!("{}.libyte", stem)
                });
            let zig_opt = flag_value(&args, "--emit-zig-opt").unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(2);
            });
            let t0 = Instant::now();
            let src = match read_source(in_path) {
                Ok(s) => s,
                Err(e) => { eprintln!("error: {}", e); std::process::exit(1); }
            };
            let base_dir = std::path::Path::new(in_path).parent()
                .map(|p| p.to_string_lossy().into_owned());
            match compile_to_libyte(&src, zig_opt.as_deref(), base_dir.as_deref()) {
                Ok((bytes, report)) => {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    if let Err(e) = std::fs::write(&out_path, &bytes) {
                        eprintln!("error: cannot write '{}': {}", out_path, e);
                        std::process::exit(1);
                    }
                    println!("{} -> {} ({} bytes, v{}, {:.2} ms)",
                        in_path, out_path, bytes.len(), report.version, ms);
                    print_build_report(&report);
                }
                Err(e) => {
                    eprintln!("error: {}", diag::render(&src, &e));
                    std::process::exit(1);
                }
            }
        }
        "watch" => {
            if args.len() < 3 {
                eprintln!("usage: lilac watch <input.lila> [-o output.libyte]");
                std::process::exit(2);
            }
            let in_path = args[2].clone();
            let out_path = flag_value(&args, "-o").unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(2);
            }).unwrap_or_else(|| {
                    let stem = in_path.trim_end_matches(".lila");
                    format!("{}.libyte", stem)
                });
            let mut last_sig: u64 = 0;
            println!("lilac watch: {} -> {} (Ctrl-C to stop)", in_path, out_path);
            loop {
                let Ok(src) = std::fs::read_to_string(&in_path) else {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                };
                // fnv-1a change detect
                let mut h: u64 = 1469598103934665603;
                for b in src.bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(1099511628211);
                }
                if h != last_sig {
                    if last_sig != 0 {
                        let t0 = Instant::now();
                        match compile_to_libyte(&src, None, None) {
                            Ok((bytes, _report)) => {
                                if let Err(e) = std::fs::write(&out_path, &bytes) {
                                    eprintln!("error: cannot write '{}': {}", out_path, e);
                                } else {
                                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                                    println!("rebuilt {} ({}B, {:.2} ms) -> {}",
                                        in_path, bytes.len(), ms, out_path);
                                }
                            }
                            Err(e) => eprintln!("error: {} (keeping last good build)", diag::render(&src, &e)),
                        }
                    } else {
                        // initial build
                        let t0 = Instant::now();
                        match compile_to_libyte(&src, None, None) {
                            Ok((bytes, _)) => {
                                let _ = std::fs::write(&out_path, &bytes);
                                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                                println!("initial build {} ({}B, {:.2} ms)",
                                    in_path, bytes.len(), ms);
                            }
                            Err(e) => eprintln!("error: {}", diag::render(&src, &e)),
                        }
                    }
                    last_sig = h;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
        "disasm" => {
            if args.len() < 3 {
                eprintln!("usage: lilac disasm <file.libyte>");
                std::process::exit(2);
            }
            let bytes = match std::fs::read(&args[2]) {
                Ok(b) => b,
                Err(e) => { eprintln!("error: cannot read '{}': {}", args[2], e); std::process::exit(1); }
            };
            match disasm::disasm(&bytes) {
                Ok(text) => print!("{}", text),
                Err(e) => { eprintln!("error: {}", e); std::process::exit(1); }
            }
        }
        // ---- SUBSYSTEM 2 CLI: SDF -> branchless WGSL sphere tracers ----
        "shader" => {
            if args.len() < 3 {
                eprintln!("usage: lilac shader <input.lila> [-o outdir]");
                std::process::exit(2);
            }
            let in_path = &args[2];
            let out_dir = flag_value(&args, "-o").unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(2);
            }).unwrap_or_else(|| ".".to_string());
            let src = match read_source(in_path) {
                Ok(s) => s,
                Err(e) => { eprintln!("error: {}", e); std::process::exit(1); }
            };
            match shader_verb(&src, &out_dir) {
                Ok(n) => println!("{}: {} shader(s) synthesized + naga-validated -> {}", in_path, n, out_dir),
                Err(e) => { eprintln!("error: {}", e); std::process::exit(1); }
            }
        }
        // ---- SUBSYSTEM 4 CLI: fit one .wav and print the recovered program ----
        "wavfit" => {
            if args.len() < 3 {
                eprintln!("usage: lilac wavfit <file.wav>");
                std::process::exit(2);
            }
            let bytes = match std::fs::read(&args[2]) {
                Ok(b) => b,
                Err(e) => { eprintln!("error: cannot read '{}': {}", args[2], e); std::process::exit(1); }
            };
            match opt::audio::parse_wav(&bytes).and_then(|pcm| opt::audio::fit(&args[2], &pcm)) {
                Ok(fit) => {
                    let (kind, desc) = match &fit.model {
                        opt::audio::FittedModel::Fm(p) => ("FM", format!(
                            "f0={}Hz sweep={:+}Hz/s ratio={:.3} index={:.2} decay={} vol={} noise={}",
                            p.f0, p.sweep, p.ratio_q4 as f32 / 16.0, p.index_q4 as f32 / 16.0,
                            p.decay, p.vol, p.noise)),
                        opt::audio::FittedModel::Add(p) => ("ADD", format!(
                            "f0={}Hz decay={} vol={} partials={}",
                            p.f0, p.decay, p.vol,
                            p.partials.iter().map(|(r, a, _)| format!("{:.3}x{}", *r as f32 / 64.0, a))
                                .collect::<Vec<_>>().join(","))),
                    };
                    println!("{}: {} voice (residual {:.1}%)", args[2], kind, fit.residual * 100.0);
                    println!("  equation: {}", fit.equation);
                    println!("  params:   {}", desc);
                }
                Err(e) => { eprintln!("error: {}", e); std::process::exit(1); }
            }
        }
        other => {
            eprintln!("unknown subcommand '{}'", other);
            std::process::exit(2);
        }
    }
}

/// The `shader` verb: parse, fold + consolidate every `sdf` block, emit the
/// branchless WGSL tracers, naga-validate each one, write the artifacts.
/// A broken shader is a BROKEN BUILD — validation happens before success.
fn shader_verb(src: &str, out_dir: &str) -> Result<usize, String> {
    let (toks, atom_names) = lexer::Lexer::new(src).tokenize()?;
    let mut p = parser::Parser::new(toks, atom_names);
    let prog = p.parse_program()?;
    if prog.sdfs.is_empty() {
        return Err("no `sdf #name { ... }` blocks in this source".into());
    }
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("cannot create '{}': {}", out_dir, e))?;
    let mut count = 0usize;
    for sd in &prog.sdfs {
        // passes 2-3: symbolic fold + smin consolidation (see opt/sdf.rs docs)
        let root = opt::sdf::fold(sd.root.clone());
        let root = opt::sdf::prune(root);
        let (root, blends_saved) = opt::sdf::consolidate(root);
        let decl = opt::sdf::SdfDecl { root, ..sd.clone() };
        let out = opt::wgsl::emit(&decl);

        // pass 6: validate BEFORE reporting success
        opt::wgsl::validate_wgsl(&out.main_wgsl, &decl.name)?;
        if let Some(hud) = &out.hud_wgsl {
            opt::wgsl::validate_wgsl(hud, &format!("{}_hud", decl.name))?;
        }

        let main_path = format!("{}/{}.wgsl", out_dir, decl.name);
        std::fs::write(&main_path, &out.main_wgsl)
            .map_err(|e| format!("cannot write '{}': {}", main_path, e))?;
        println!("  sdf #{}: {} nodes, {} blends consolidated -> {}",
            decl.name, opt::sdf::node_count(&decl.root), blends_saved, main_path);
        if let Some(hud) = &out.hud_wgsl {
            let hud_path = format!("{}/{}_hud.wgsl", out_dir, decl.name);
            std::fs::write(&hud_path, hud)
                .map_err(|e| format!("cannot write '{}': {}", hud_path, e))?;
            println!("  sdf #{}_hud: specialized screen-space variant (no march loop)", decl.name);
        }
        count += 1;
    }
    Ok(count)
}

// ============================ v11 TESTS ============================

#[cfg(test)]
mod v11_tests {
    use super::*;

    /// Global + entity arrays, indexed reads/writes (incl. compound), alen,
    /// cam3/draw3d/proj3 — the whole v11 surface compiles end to end, emits
    /// a v3 file with the arrays flag, and disasm round-trips the new ops.
    #[test]
    fn arrays_and_3d_compile_end_to_end() {
        let src = r#"
game { title: "t"  width: 256  height: 256  wrap: false }
global {
    tick: u = 0
    arr fixed[1800] terr
    arr fixed[108] cube
}
entity Orb {
    x: fixed
    ti: u8
    arr fixed[8] tx
}
fn init() {
    let i = 0;
    while (i < 10) {
        terr[i * 3] = fixed(i);
        terr[i * 3 + 1] = fixed(i) * 0.5;
        terr[i * 3 + 2] = sin(ang(int(fixed(i)) * 4)) * 1.2;
        i += 1;
    }
    cube[0] = -1.0;
    cube[1] = 0.5 + 0.5;
    cube[1] += 0.25;
    let h = terr[0] + cube[0];
    cube[2] = h;
    spawn Orb { ti: 2 };
}
fn update() {
    tick += 1;
    cam3(0.0, -10.0, 4.0, 0, -130);
}
fn update(o: Orb) {
    o.x = sin(ang(tick)) * 2.0;
    o.tx[o.ti] = o.x;
    o.ti += 1;
    o.x = o.tx[0] + o.tx[o.ti];
}
fn draw() {
    draw3d(terr, alen(terr) / 3 - 300 + 300, 0.0, 0.0, 0.0, 0, 0x2E7D4FCC);
    draw3d(cube, 36, 0.0, 0.0, 3.0, ang(tick * 5), 0xE07030D8);
    for (o: Orb) {
        let s = proj3(o.x, 0.0, 1.0);
        if (projok()) {
            draw(#p, projx(), projy(), 0, s * 0.16, 0xFFE0B0FF);
        }
    }
    draw_text(#t1, 8, 8, 0xFFFFFFCC);
}
scene { #play: idle_scene }
fn idle_scene() {
}
sprite #p (4) {
    poly [ -4, 0, 0, -4, 4, 0, 0, 4 ] fill: 0xFFD080 add: true
}
text { #t1: "V11" }
"#;
        let (bytes, report) = compile_to_libyte(src, None, None).expect("compiles");
        assert_eq!(report.version, 3, "arrays force v3");
        // flags bit3 (arrays) must be set: header = "LIBY"(4) + ver u16 + flags u16
        let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        assert_eq!(flags & 8, 8, "arrays flag set");
        assert!(bytes.len() > 100);
        // disasm round-trips the new opcodes
        let text = crate::disasm::disasm(&bytes).expect("disasm");
        for op in ["LD_GARR", "ST_GARR", "LD_EARR", "ST_EARR", "DUP", "CAM3", "PROJ3", "PROJ_X", "DRAW3D"] {
            assert!(text.contains(op), "disasm must name {op}");
        }
    }

    /// The compile-time bounds PROOF: literal indices outside 0..cap are
    /// build errors — the Lila replacement for runtime bound traps.
    #[test]
    fn array_literal_out_of_bounds_is_a_build_error() {
        let src = r#"
game { title: "t"  width: 64  height: 64 }
global { arr u8[4] g }
fn init() { g[4] = 1; }
"#;
        let err = compile_to_libyte(src, None, None).unwrap_err();
        assert!(err.contains("out of bounds"), "got: {err}");
    }

    /// Dynamic indices are legal (the runtime wraps them — mask|mod, the
    /// same trap-free contract as SRAM slots); a NEGATIVE literal is folded
    /// by the parser and caught by the compile-time proof.
    #[test]
    fn array_negative_literal_index_rejected() {
        let src = r#"
game { title: "t"  width: 64  height: 64 }
global { arr u8[4] g }
fn init() { let i = 0; g[i] = 1; g[-1] = 2; }
"#;
        let err = compile_to_libyte(src, None, None).unwrap_err();
        assert!(err.contains("out of bounds"), "got: {err}");
    }

    /// draw3d is a draw-pass statement and its budget/capacity are proven.
    #[test]
    fn draw3d_placement_and_budget_enforced() {
        let base = |body: &str| format!(r#"
game {{ title: "t"  width: 64  height: 64 }}
global {{ arr fixed[36] mesh }}
fn draw() {{ {body} }}
"#);
        // >768 verts
        let err = compile_to_libyte(&base("draw3d(mesh, 900, 0.0, 0.0, 0.0, 0, 0xFF0000FF);"), None, None).unwrap_err();
        assert!(err.contains("768"), "got: {err}");
        // verts*3 > cells
        let err = compile_to_libyte(&base("draw3d(mesh, 13, 0.0, 0.0, 0.0, 0, 0xFF0000FF);"), None, None).unwrap_err();
        assert!(err.contains("x 3 cells"), "got: {err}");
        // update-pass placement
        let src = r#"
game { title: "t"  width: 64  height: 64 }
global { arr fixed[36] mesh }
fn update() { draw3d(mesh, 12, 0.0, 0.0, 0.0, 0, 0xFF0000FF); }
"#;
        let err = compile_to_libyte(src, None, None).unwrap_err();
        assert!(err.contains("draw-pass"), "got: {err}");
    }

    /// Entity arrays need the handle prefix; bare names stay global-only.
    #[test]
    fn entity_array_scoping() {
        let ok = r#"
game { title: "t"  width: 64  height: 64 }
global { arr u8[4] g }
entity E { v: u8  arr u8[8] t }
fn update(e: E) { e.t[e.v] = g[0]; e.v = e.t[1]; }
"#;
        assert!(compile_to_libyte(ok, None, None).is_ok());
        let bad = r#"
game { title: "t"  width: 64  height: 64 }
entity E { v: u8  arr u8[8] t }
fn update(e: E) { t[0] = 1; }
"#;
        let err = compile_to_libyte(bad, None, None).unwrap_err();
        assert!(err.contains("unknown global array"), "got: {err}");
    }

    /// The LEGACY CONTRACT at the language level: games using none of the
    /// new surface still emit v1 (and v2 for capacity/world), never v3.
    #[test]
    fn legacy_games_keep_their_version() {
        let v1 = r#"
game { title: "t"  width: 64  height: 64 }
entity E { x: fixed  y: fixed }
fn init() { spawn E { }; }
fn update() { }
fn draw() { }
scene { #s: idle }
fn idle() { }
"#;
        let (_, rep) = compile_to_libyte(v1, None, None).expect("compiles");
        assert_eq!(rep.version, 1);
        let v2 = v1.replace("height: 64 }", "height: 64  capacity { entities: 32 } }");
        let (_, rep) = compile_to_libyte(&v2, None, None).expect("compiles");
        assert_eq!(rep.version, 2);
    }

    /// Regression: the fixed<->int scale chain that overflowed Interval::mul
    /// (i64 wrap) in opt::ranges must compile under optimize:true.
    #[test]
    fn ranges_interval_mul_handles_widened_chains() {
        let src = r#"
game { title: "t"  width: 512  height: 512  wrap: false  optimize: true }
global { arr fixed[64] m }
fn init() {
    let i = 0;
    while (i < 8) {
        let x = fixed(i * 64 - 256);
        m[i * 3] = x;
        m[i * 3 + 1] = sin(ang(int(x) * 4)) * 1.2 + cos(ang(int(x) * 4)) * 0.9;
        m[i * 3 + 2] = fixed(i) * -2.0;
        i += 1;
    }
}
fn update() { cam3(0.0, -10.0, 4.0, 0, -130); }
fn draw() { draw3d(m, 8, 0.0, 0.0, 0.0, 0, 0x2E7D4FCC); }
"#;
        let (_, _) = compile_to_libyte(src, None, None).expect("optimize:true path must not panic");
    }

}

// ============================ v12 TESTS ============================

#[cfg(test)]
mod v12_tests {
    use super::*;

    /// ARTICULATED 3D end to end: quat_aa/qmul/m4qt/m4mul/skinv pose a
    /// 2-bone rig in update(), draw3di submits an indexed mesh in draw().
    /// Emits a v3 file and disasm round-trips every new opcode.
    #[test]
    fn articulated_3d_compiles_end_to_end() {
        let src = r#"
game { title: "t"  width: 256  height: 256  wrap: false }
global {
    tick: u = 0
    arr fixed[24] verts
    arr u8[24] idx
    arr fixed[16] wm0
    arr fixed[16] wm1
    arr fixed[16] lm
    arr fixed[4] lq
    arr fixed[8] bind
}
fn init() {
    let v = 0;
    while (v < 8) {
        bind[v * 3] = fixed(v) * 0.1;
        bind[v * 3 + 1] = 0.0;
        bind[v * 3 + 2] = fixed(v) * 0.2;
        v += 1;
    }
    idx[0] = 0; idx[1] = 1; idx[2] = 2;
    idx[3] = 0; idx[4] = 2; idx[5] = 3;
    idx[6] = 4; idx[7] = 5; idx[8] = 6;
    idx[9] = 4; idx[10] = 6; idx[11] = 7;
    goto(#play);
}
scene { #play: idle }
fn idle() { }
fn update() {
    tick += 1;
    cam3(0.0, -8.0, 3.0, 0, -220);
    quat_aa(lq, 0, 0.0, 0.0, 1.0, ang(tick * 4));
    m4qt(wm0, 0, lq, 0, 0.0, 0.0, 0.0);
    quat_aa(lq, 0, 1.0, 0.0, 0.0, ang(tick * 7));
    m4qt(lm, 0, lq, 0, 0.0, 0.0, 1.5);
    m4mul(wm1, 0, wm0, 0, lm, 0);
    let v = 0;
    while (v < 4) {
        skinv(verts, v * 3, wm0, 0, bind[v * 3], bind[v * 3 + 1], bind[v * 3 + 2]);
        v += 1;
    }
    while (v < 8) {
        skinv(verts, v * 3, wm1, 0, bind[v * 3], bind[v * 3 + 1], bind[v * 3 + 2]);
        v += 1;
    }
}
fn draw() {
    draw3di(verts, 8, idx, 12, 0.0, 0.0, 0.0, 0, 0xE07030D8);
}
"#;
        let (bytes, rep) = compile_to_libyte(src, None, None).expect("compiles");
        assert_eq!(rep.version, 3, "arrays force v3");
        let text = crate::disasm::disasm(&bytes).expect("disasm");
        for op in ["QUAT_AA", "Q_MUL", "M4_QT", "M4_MUL", "SKIN3", "DRAW3DI"] {
            if op == "Q_MUL" { continue; } // not in this game
            assert!(text.contains(op), "disasm must name {op}");
        }
        // compose a qmul too
        let src2 = src.replace("m4mul(wm1, 0, wm0, 0, lm, 0);",
            "qmul(lq, 0, lq, 0, lq, 0); m4mul(wm1, 0, wm0, 0, lm, 0);");
        let (bytes2, _) = compile_to_libyte(&src2, None, None).expect("compiles");
        let text2 = crate::disasm::disasm(&bytes2).expect("disasm");
        assert!(text2.contains("Q_MUL"), "disasm must name Q_MUL");
    }

    /// Articulation statements are GAMEPLAY state: banned in draw fns.
    /// draw3di is the inverse: draw-pass only.
    #[test]
    fn articulation_placement_enforced() {
        let base = |body: &str| format!(r#"
game {{ title: "t"  width: 64  height: 64 }}
global {{
    arr fixed[16] m
    arr fixed[4] q
    arr fixed[24] v
    arr u8[12] ix
}}
fn init() {{ }}
fn update() {{ cam3(0.0, -8.0, 3.0, 0, -220); }}
fn draw() {{ {body} }}
"#);
        let err = compile_to_libyte(&base("skinv(v, 0, m, 0, 0.0, 0.0, 0.0);"), None, None).unwrap_err();
        assert!(err.contains("not allowed in draw fns"), "got: {err}");
        let err = compile_to_libyte(&base("quat_aa(q, 0, 0.0, 0.0, 1.0, 128);"), None, None).unwrap_err();
        assert!(err.contains("not allowed in draw fns"), "got: {err}");
        // draw3di in update = rejected
        let upd = r#"
game { title: "t"  width: 64  height: 64 }
global { arr fixed[24] v  arr u8[12] ix }
fn init() { }
fn update() { draw3di(v, 8, ix, 12, 0.0, 0.0, 0.0, 0, 0xFF0000FF); }
"#;
        let err = compile_to_libyte(upd, None, None).unwrap_err();
        assert!(err.contains("draw-pass statement"), "got: {err}");
    }

    /// Negative: entity arrays rejected for rig storage; literal offsets
    /// bounds-proven; index counts must be triples; the two buffers must
    /// differ; budget enforcement matches draw3d.
    #[test]
    fn articulation_type_and_budget_errors() {
        let ent_arr = r#"
game { title: "t"  width: 64  height: 64 }
entity W { arr fixed[4] q }
global { arr fixed[16] m }
fn init() { }
fn update(w: W) { quat_aa(w.q, 0, 0.0, 0.0, 1.0, 128); }
"#;
        let err = compile_to_libyte(ent_arr, None, None).unwrap_err();
        assert!(err.contains("not a global array"), "got: {err}");

        let off_over = r#"
game { title: "t"  width: 64  height: 64 }
global { arr fixed[8] m  arr fixed[4] q }
fn init() { }
fn update() { m4qt(m, 0, q, 2, 0.0, 0.0, 0.0); }
"#;
        let err = compile_to_libyte(off_over, None, None).unwrap_err();
        assert!(err.contains("holds only"), "got: {err}");

        let off_over2 = r#"
game { title: "t"  width: 64  height: 64 }
global { arr fixed[20] m }
fn init() { }
fn update() { m4mul(m, 16, m, 0, m, 0); }
"#;
        let err = compile_to_libyte(off_over2, None, None).unwrap_err();
        assert!(err.contains("holds only"), "got: {err}");

        let bad_ni = r#"
game { title: "t"  width: 64  height: 64 }
global { arr fixed[24] v  arr u8[7] ix }
fn init() { }
fn update() { cam3(0.0, -8.0, 3.0, 0, -220); }
fn draw() { draw3di(v, 8, ix, 7, 0.0, 0.0, 0.0, 0, 0xFF0000FF); }
"#;
        let err = compile_to_libyte(bad_ni, None, None).unwrap_err();
        assert!(err.contains("multiple of 3"), "got: {err}");

        let same_buf = r#"
game { title: "t"  width: 64  height: 64 }
global { arr fixed[24] v }
fn init() { }
fn update() { cam3(0.0, -8.0, 3.0, 0, -220); }
fn draw() { draw3di(v, 8, v, 12, 0.0, 0.0, 0.0, 0, 0xFF0000FF); }
"#;
        let err = compile_to_libyte(same_buf, None, None).unwrap_err();
        assert!(err.contains("different arrays"), "got: {err}");

        let over_budget = r#"
game { title: "t"  width: 64  height: 64 }
global { arr fixed[2400] v  arr u8[12] ix }
fn init() { }
fn update() { cam3(0.0, -8.0, 3.0, 0, -220); }
fn draw() { draw3di(v, 900, ix, 12, 0.0, 0.0, 0.0, 0, 0xFF0000FF); }
"#;
        let err = compile_to_libyte(over_budget, None, None).unwrap_err();
        assert!(err.contains("budget"), "got: {err}");
    }

    /// optimize:true pipeline must not panic on the articulation surface
    /// (the arena r/w pass now tracks array traffic through Index too).
    #[test]
    fn articulation_under_optimize() {
        let src = r#"
game { title: "t"  width: 512  height: 512  wrap: false  optimize: true }
global {
    arr fixed[24] v
    arr u8[12] ix
    arr fixed[32] wm
    arr fixed[4] lq
}
fn init() { }
fn update() {
    cam3(0.0, -8.0, 3.0, 0, -220);
    quat_aa(lq, 0, 0.0, 0.0, 1.0, ang(17));
    m4qt(wm, 0, lq, 0, 0.0, 0.0, 1.0);
    let k = 0;
    while (k < 8) {
        skinv(v, k * 3, wm, 0, fixed(k), 0.0, 0.0);
        k += 1;
    }
}
fn draw() { draw3di(v, 8, ix, 12, 0.0, 0.0, 0.0, 0, 0xE07030D8); }
"#;
        let (_, _) = compile_to_libyte(src, None, None).expect("optimize:true articulation must compile");
    }

    /// v12 LOUD LIMIT (regression): a fn needing more local slots than the
    /// VM frame holds is a BUILD ERROR, not a silent miscompile — stores
    /// past the limit used to drop, hanging loops at runtime.
    #[test]
    fn locals_over_frame_limit_is_a_build_error() {
        let mut decls = String::new();
        let mut body = String::new();
        for i in 0..40 {
            decls.push_str(&format!("    let v{i} = {i};\n"));
        }
        // keep a loop alive after the locals so the old bug would hang it
        body.push_str(&decls);
        body.push_str("    let k = 0;\n    while (k < 4) { k += 1; }\n");
        let src = format!(
            "game {{ title: \"t\"  width: 64  height: 64 }}\nfn init() {{\n{body}}}\n"
        );
        let err = compile_to_libyte(&src, None, None).unwrap_err();
        assert!(err.contains("local slots"), "got: {err}");
        // 24 locals is inside the 32-slot frame and compiles fine
        let mut body2 = String::new();
        for i in 0..24 {
            body2.push_str(&format!("    let w{i} = {i};\n"));
        }
        body2.push_str("    let k = 0;\n    while (k < 4) { k += 1; }\n");
        let src2 = format!(
            "game {{ title: \"t\"  width: 64  height: 64 }}\nfn init() {{\n{body2}}}\n"
        );
        let (_, _) = compile_to_libyte(&src2, None, None).expect("24 locals must compile");
    }

    /// v13 RANGE SOUNDNESS (regression): a compound increment on global
    /// state persists across frames, so its domain is unbounded within the
    /// declared width. The fixpoint walk used to "prove" max=16 (16 unroll
    /// rounds) and narrowed a u16 frame counter to 5 bits — found by LILA
    /// DRIVE where the day/night cycle wrapped every 32 frames.
    #[test]
    fn compound_increment_never_narrows_global_width() {
        let src = r#"
game { title: "t"  width: 64  height: 64  optimize: true }
global { tick: u16 = 0  score: u16 = 0 }
fn init() { goto(#play); }
scene { #play: idle }
fn idle() { }
fn update() {
    tick += 1;
    if (tick % 60 == 0) { score += 10; }
}
fn draw() { }
"#;
        let (_, report) = compile_to_libyte(src, None, None).expect("must compile");
        for (name, _from, to) in &report.opt.narrowed {
            assert!(name != "global.tick", "tick += 1 must never narrow (got u{})", to);
        }
    }

    /// v13 GLOBALS BLOCK GUARD (regression): globals live in the 128-byte
    /// register-free page-0 gap 0x80..0x100. Before the guard, a game with
    /// more than 1024 bits of globals silently overran the shake/cam3/proj
    /// registers and corrupted the engine (zeroed shake decay -> overflow
    /// panic). The build must fail loudly with the numbers.
    #[test]
    fn globals_block_overflow_is_a_build_error() {
        // 48 fixed globals = 1536 bits > 1024 — over the block
        let mut decls = String::new();
        for i in 0..48 {
            decls.push_str(&format!("    g{i}: fixed = 0.0
"));
        }
        let src = format!(
            "game {{ title: \"t\"  width: 64  height: 64 }}\nglobal {{\n{decls}}}\nfn init() {{ }}\n"
        );
        let err = compile_to_libyte(&src, None, None).unwrap_err();
        assert!(err.contains("globals block overflow"), "got: {err}");
        // 30 fixed globals = 960 bits — fits, compiles fine
        let mut decls2 = String::new();
        for i in 0..30 {
            decls2.push_str(&format!("    h{i}: fixed = 0.0\n"));
        }
        let src2 = format!(
            "game {{ title: \"t\"  width: 64  height: 64 }}\nglobal {{\n{decls2}}}\nfn init() {{ }}\n"
        );
        let (_, _) = compile_to_libyte(&src2, None, None).expect("960 bits of globals must fit");
    }
}
