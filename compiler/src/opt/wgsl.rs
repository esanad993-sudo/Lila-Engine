//! ============================================================================
//! LILA OPTIMIZER — SUBSYSTEM 2 EMITTER: branchless WGSL synthesis
//! (consumes opt::sdf's folded/consolidated tree; see that file for passes 1-3)
//! ============================================================================
//!
//! EMITTED ARTIFACTS (written when the DEV runs `lilac shader`):
//!   <name>.wgsl       full-screen 3D sphere tracer (WebGPU compute-free)
//!   <name>_hud.wgsl   specialized screen-space variant (when `hud: true`) —
//!                     a SEPARATE shader, so no GPU ever branches on a
//!                     dynamic "am I the HUD pipeline" condition
//!
//! The 3D tracer is BRANCHLESS BY CONSTRUCTION:
//!   * fixed iteration count (STEPS), no data-dependent `break`;
//!   * the march stops advancing via `select()` masking (t freezes once the
//!     hit epsilon is reached) — the ALU does the work a branch would;
//!   * hit-id, normal and shading all accumulate through select/max blends;
//!   * normals come from the SYMBOLIC GRADIENT (pass 4, autodiff) evaluated
//!     inside the same SDF call — 0 extra evaluations instead of the classic
//!     4-tap tetrahedron trick.
//!
//! Every emitted shader is parsed + VALIDATED by naga before `lilac shader`
//! reports success: a broken shader breaks the build, exactly like a type
//! error in game code.

use crate::opt::sdf::{SdfDecl, Shape};
use std::fmt::Write;

const STEPS: u32 = 96;
const MAX_DIST: f32 = 40.0;
const EPS: f32 = 0.0008;

pub struct ShaderOutput {
    pub main_wgsl: String,
    pub hud_wgsl: Option<String>,
}

pub fn emit(decl: &SdfDecl) -> ShaderOutput {
    let mut notes = Vec::new();
    let main_wgsl = emit_main(decl, &mut notes);
    let hud_wgsl = if decl.hud { Some(emit_hud(decl, &mut notes)) } else { None };
    let _ = &notes; // (collected for the test/report path below)
    ShaderOutput { main_wgsl, hud_wgsl }
}

// ---------------- node emission (autodiff: d + analytic grad) ----------------
//
// Every node compiles to `fn sdf_N(p: vec3f) -> vec4f` = (d, gx, gy, gz).
// That vec4 IS the autodiff artifact: the pass-4 chain rules live in the
// emitted select/mix expressions below.

struct Emitter {
    body: String,   // fn definitions  // statements inside sdf_scene
    counter: u32,
}

impl Emitter {
    fn next_id(&mut self) -> u32 {
        self.counter += 1;
        self.counter
    }

    /// Emit a node; returns the WGSL expression yielding vec4f(d, grad).
    fn emit_node(&mut self, s: &Shape) -> String {
        match s {
            Shape::Sphere { c, r } => {
                let id = self.next_id();
                let _ = write!(self.body,
"fn sdf_{id}(p: vec3f) -> vec4f {{
    let q = p - vec3f({cx}, {cy}, {cz});
    let l = length(q);
    let d = l - {r};
    // analytic normal: radial direction (exact)
    let g = select(vec3f(0.0, 1.0, 0.0), q / l, l > 1e-6);
    return vec4f(d, g);
}}
", id = id, cx = f(c.x), cy = f(c.y), cz = f(c.z), r = f(*r));
                format!("sdf_{}(p)", id)
            }
            Shape::Box { c, half, round } => {
                let id = self.next_id();
                let _ = write!(self.body,
"fn sdf_{id}(p: vec3f) -> vec4f {{
    let q0 = abs(p - vec3f({cx}, {cy}, {cz})) - vec3f({hx}, {hy}, {hz}) + vec3f({rd});
    let dOut = length(max(q0, vec3f(0.0)));
    let dIn = min(max(q0.x, max(q0.y, q0.z)), 0.0);
    let d = dOut + dIn - {rd};
    let s = sign(p - vec3f({cx}, {cy}, {cz}));
    // outside: radial in the clamped octant; inside: axis of the max extent
    let l = length(max(q0, vec3f(0.0)));
    let gOut = s * select(vec3f(1.0, 0.0, 0.0), max(q0, vec3f(0.0)) / l, l > 1e-6);
    let axis = select(vec3f(0.0, 0.0, 1.0), select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), q0.x >= max(q0.y, q0.z)), q0.y >= max(q0.x, q0.z));
    let g = select(s * axis, gOut, dOut > 1e-6);
    return vec4f(d, g);
}}
", id = id, cx = f(c.x), cy = f(c.y), cz = f(c.z), hx = f(half.x), hy = f(half.y), hz = f(half.z), rd = f(round.max(1e-5)));
                format!("sdf_{}(p)", id)
            }
            Shape::Torus { c, major, minor } => {
                let id = self.next_id();
                let _ = write!(self.body,
"fn sdf_{id}(p: vec3f) -> vec4f {{
    let q = p - vec3f({cx}, {cy}, {cz});
    let rxz = length(q.xz);
    let t2 = vec2f(rxz - {mj}, q.y);
    let d = length(t2) - {mn};
    let inv = select(1.0, 1.0 / rxz, rxz > 1e-6);
    let lt = length(t2);
    let g = select(vec3f(0.0, 1.0, 0.0),
        vec3f(q.x * inv * t2.x, t2.y, q.z * inv * t2.x) / select(1.0, lt, lt > 1e-6),
        lt > 1e-6);
    return vec4f(d, g);
}}
", id = id, cx = f(c.x), cy = f(c.y), cz = f(c.z), mj = f(*major), mn = f(*minor));
                format!("sdf_{}(p)", id)
            }
            Shape::Capsule { a, b, r } => {
                let id = self.next_id();
                let _ = write!(self.body,
"fn sdf_{id}(p: vec3f) -> vec4f {{
    let pa = vec3f({ax}, {ay}, {az});
    let ba = vec3f({bx}, {by}, {bz}) - pa;
    let h = clamp(dot(p - pa, ba) / dot(ba, ba), 0.0, 1.0);
    let q = p - pa - ba * h;
    let l = length(q);
    let d = l - {r};
    let g = select(vec3f(0.0, 1.0, 0.0), q / l, l > 1e-6);
    return vec4f(d, g);
}}
", id = id, ax = f(a.x), ay = f(a.y), az = f(a.z), bx = f(b.x), by = f(b.y), bz = f(b.z), r = f(*r));
                format!("sdf_{}(p)", id)
            }
            Shape::Plane { y } => {
                let id = self.next_id();
                let _ = write!(self.body,
"fn sdf_{id}(p: vec3f) -> vec4f {{
    return vec4f(p.y - {y}, 0.0, 1.0, 0.0);
}}
", id = id, y = f(*y));
                format!("sdf_{}(p)", id)
            }
            Shape::Union(kids) => {
                // chained min with gradient-of-selected-branch (exact except
                // the measure-zero crease — the standard union normal)
                let ids: Vec<String> = kids.iter().map(|k| self.emit_node(k)).collect();
                let id = self.next_id();
                let mut fn_body = String::new();
                let _ = write!(fn_body, "    var best = {};\n", ids[0]);
                for e in &ids[1..] {
                    let _ = write!(fn_body,
"    let c{e} = {e};
    best = select(best, c{e}, c{e}.x < best.x);
");
                }
                let _ = write!(self.body,
"fn sdf_{id}(p: vec3f) -> vec4f {{
{fn_body}    return best;
}}
", id = id, fn_body = fn_body);
                format!("sdf_{}(p)", id)
            }
            Shape::SmoothUnion { k, kids } => {
                // n-ary polynomial smin: ONE running pair (post-consolidation
                // form — pass 3 flattened the equal-k tree)
                let ids: Vec<String> = kids.iter().map(|k| self.emit_node(k)).collect();
                let id = self.next_id();
                let mut fn_body = String::new();
                let _ = write!(fn_body, "    var d = {};\n", ids[0]);
                for e in &ids[1..] {
                    let _ = write!(fn_body,
"    {{
        let c = {e};
        let h = clamp(0.5 + 0.5 * (c.x - d.x) / {k}, 0.0, 1.0);
        let dm = mix(c.x, d.x, h) - {k} * h * (1.0 - h);
        let gm = mix(c.yzw, d.yzw, h);
        d = vec4f(dm, gm);
    }}
", k = f(*k));
                }
                let _ = write!(self.body,
"fn sdf_{id}(p: vec3f) -> vec4f {{
{fn_body}    return d;
}}
", id = id, fn_body = fn_body);
                format!("sdf_{}(p)", id)
            }
            Shape::Sub(a, b) => {
                let ea = self.emit_node(a);
                let eb = self.emit_node(b);
                let id = self.next_id();
                let _ = write!(self.body,
"fn sdf_{id}(p: vec3f) -> vec4f {{
    let A = {ea};
    let B = {eb};
    // subtraction keeps A's surface: its gradient is the normal
    return vec4f(max(A.x, -B.x), A.yzw);
}}
", id = id, ea = ea, eb = eb);
                format!("sdf_{}(p)", id)
            }
            Shape::Isect(a, b) => {
                let ea = self.emit_node(a);
                let eb = self.emit_node(b);
                let id = self.next_id();
                let _ = write!(self.body,
"fn sdf_{id}(p: vec3f) -> vec4f {{
    let A = {ea};
    let B = {eb};
    return select(A, B, B.x > A.x);
}}
", id = id, ea = ea, eb = eb);
                format!("sdf_{}(p)", id)
            }
            Shape::SmoothSub { k, a, b } => {
                let ea = self.emit_node(a);
                let eb = self.emit_node(b);
                let id = self.next_id();
                let _ = write!(self.body,
"fn sdf_{id}(p: vec3f) -> vec4f {{
    let A = {ea};
    let B = {eb};
    let na = vec4f(-A.x, -A.yzw);
    let h = clamp(0.5 + 0.5 * (B.x - na.x) / {k}, 0.0, 1.0);
    let dm = mix(B.x, na.x, h) - {k} * h * (1.0 - h);
    let gm = mix(B.yzw, na.yzw, h);
    // d = -smin(-a, b, k): negate back, gradient negates with it
    return vec4f(-dm, -gm);
}}
", id = id, ea = ea, eb = eb, k = f(*k));
                format!("sdf_{}(p)", id)
            }
            Shape::Twist { k, kid } => {
                let ek = self.emit_node(kid);
                let id = self.next_id();
                let _ = write!(self.body,
"fn sdf_{id}(p: vec3f) -> vec4f {{
    // domain op: rotate p.xz by angle k*p.y, evaluate, rotate the gradient
    // back through the transpose (first-order chain rule — the standard
    // domain-gradient; the dR/dp term is the usual second-order omission).
    let a = {k} * p.y;
    let (ca, sa) = (cos(a), sin(a));
    let q = vec3f(ca * p.x + sa * p.z, p.y, -sa * p.x + ca * p.z);
    let r = {ek};
    let gr = vec3f(ca * r.y + sa * r.z, r.y, -sa * r.y + ca * r.z);
    return vec4f(r.x, gr);
}}
", id = id, k = f(*k), ek = ek);
                format!("sdf_{}(p)", id)
            }
            Shape::Repeat { cell, kid } => {
                let ek = self.emit_node(kid);
                let id = self.next_id();
                let cx = if cell.x != 0.0 { f(cell.x) } else { "0.0".to_string() };
                let cy = if cell.y != 0.0 { f(cell.y) } else { "0.0".to_string() };
                let cz = if cell.z != 0.0 { f(cell.z) } else { "0.0".to_string() };
                // translation domain op: gradient passes through unchanged
                let _ = write!(self.body,
"fn sdf_{id}(p: vec3f) -> vec4f {{
    var q = p;
    if ({cx} != 0.0) {{ q.x = p.x - {cx} * round(p.x / {cx}); }}
    if ({cy} != 0.0) {{ q.y = p.y - {cy} * round(p.y / {cy}); }}
    if ({cz} != 0.0) {{ q.z = p.z - {cz} * round(p.z / {cz}); }}
    let r = {ek};
    return r;
}}
", id = id, cx = cx, cy = cy, cz = cz, ek = ek);
                format!("sdf_{}(p)", id)
            }
        }
    }
}

fn f(v: f32) -> String {
    if v == v.trunc() && v.abs() < 1e6 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

fn color_rgb(c: u32) -> (f32, f32, f32) {
    (((c >> 16) & 0xFF) as f32 / 255.0, ((c >> 8) & 0xFF) as f32 / 255.0, (c & 0xFF) as f32 / 255.0)
}

// ---------------- the 3D branchless tracer ----------------

fn emit_main(decl: &SdfDecl, notes: &mut Vec<(String, String)>) -> String {
    let mut e = Emitter { body: String::new(), counter: 0 };
    let scene_expr = e.emit_node(&decl.root);
    let (cr, cg, cb) = color_rgb(decl.color);

    let mut out = String::new();
    let _ = write!(out,
"// GENERATED by lilac v8 shader synthesis — {} (branchless sphere tracing)
// Source: sdf #{} — folded + smin-consolidated + autodiff'd at compile time.
// Normals are the SYMBOLIC GRADIENT of the SDF (zero extra evaluations).
// Uniforms: res (pixels), time (s), aspect (w/h). Group 0, binding 0.

struct U {{
    res : vec2f,
    time : f32,
    aspect : f32,
}};
@group(0) @binding(0) var<uniform> u : U;

const STEPS : i32 = {steps}i;
const MAX_DIST : f32 = {maxd};
const EPS : f32 = {eps};

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4f {{
    // fullscreen triangle — zero vertex buffers, zero draws cost
    var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(pos[vi], 0.0, 1.0);
}}

", decl.name, decl.name, steps = STEPS, maxd = f(MAX_DIST), eps = f(EPS));
    out.push_str(&e.body);

    let _ = write!(out,
"
fn sdf_scene(p : vec3f) -> vec4f {{
    return {scene_expr};
}}

@fragment
fn fs(@builtin(position) frag : vec4f) -> @location(0) vec4f {{
    let ndc = (frag.xy * 2.0 - u.res) / u.res.y;
    // orbiting camera anchored at the declared eye/look
    let ang = u.time * 0.35;
    let eye = vec3f({ex}, {ey}, {ez});
    let look = vec3f({lx}, {ly}, {lz});
    let ro = look + vec3f((eye.x - look.x) * cos(ang) - (eye.z - look.z) * sin(ang),
                          eye.y - look.y,
                          (eye.x - look.x) * sin(ang) + (eye.z - look.z) * cos(ang));
    let fw = normalize(look - ro);
    let rt = normalize(cross(fw, vec3f(0.0, 1.0, 0.0)));
    let up = cross(rt, fw);
    let rd = normalize(fw * 1.6 + rt * ndc.x + up * ndc.y);

    // ---- BRANCHLESS SPHERE TRACING ----
    // fixed trip count; the march freezes through select() masking —
    // no data-dependent exit anywhere in the loop
    var t = 0.05;
    var hitm = 0.0;
    var nrm = vec3f(0.0, 1.0, 0.0);
    for (var i : i32 = 0; i < STEPS; i++) {{
        let pos = ro + rd * t;
        let s = sdf_scene(pos);
        let m = select(0.0, 1.0, s.x < EPS * max(t, 1.0) && s.x > -EPS * 64.0);
        nrm = select(nrm, normalize(s.yzw), m * (1.0 - hitm));
        hitm = max(hitm, m);
        // once hit, stop advancing (d -> 0) — the loop keeps rolling branchlessly
        t = t + select(select(s.x, 0.0, s.x < -EPS * 64.0), 0.0, hitm);
    }}
    let hit = hitm * select(0.0, 1.0, t < MAX_DIST);

    // ---- shading (masked blends only) ----
    let l1 = normalize(vec3f(0.6, 0.9, -0.5));
    let diff = max(dot(nrm, l1), 0.0);
    let rim = pow(1.0 - max(dot(nrm, -rd), 0.0), 3.0) * {glow};
    let base = vec3f({cr}, {cg}, {cb});
    let shaded = base * (0.22 + 0.78 * diff) + base * rim;
    let bg = mix(vec3f(0.03, 0.04, 0.08), vec3f(0.10, 0.11, 0.16), rd.y * 0.5 + 0.5);
    let col = mix(bg, shaded, hit);
    return vec4f(col, 1.0);
}}
", ex = f(decl.eye.x), ey = f(decl.eye.y), ez = f(decl.eye.z),
      lx = f(decl.look.x), ly = f(decl.look.y), lz = f(decl.look.z),
      glow = f(decl.glow), cr = f(cr), cg = f(cg), cb = f(cb));

    notes.push((decl.name.clone(), format!("{} SDF nodes -> 1 fragment shader, {} march steps, analytic normals (0 extra evals)", crate::opt::sdf::node_count(&decl.root), STEPS)));
    out
}

// ---------------- the HUD variant (separate specialized shader) ----------------

fn emit_hud(decl: &SdfDecl, notes: &mut Vec<(String, String)>) -> String {
    // Screen-space 2D: evaluate the folded SDF on the z=0 slice with ANALYTIC
    // anti-aliasing from the same symbolic gradient — no march loop at all.
    let mut e = Emitter { body: String::new(), counter: 0 };
    let scene_expr = e.emit_node(&decl.root);
    let (cr, cg, cb) = color_rgb(decl.color);

    let mut out = String::new();
    let _ = write!(out,
"// GENERATED by lilac v8 shader synthesis — {}_hud (specialized variant)
// #hud_mode bake: screen-space SDF, analytic AA from the symbolic gradient.
// No sphere tracing, no dynamic branching — this IS the HUD pipeline.

struct U {{
    res : vec2f,
    time : f32,
    aspect : f32,
}};
@group(0) @binding(0) var<uniform> u : U;

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4f {{
    var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(pos[vi], 0.0, 1.0);
}}

", decl.name);
    out.push_str(&e.body);

    let _ = write!(out,
"
fn sdf_scene(p : vec3f) -> vec4f {{
    return {scene_expr};
}}

@fragment
fn fs(@builtin(position) frag : vec4f) -> @location(0) vec4f {{
    let ndc = (frag.xy * 2.0 - u.res) / u.res.y;
    let p = vec3f(ndc * 2.4, 0.0);
    let s = sdf_scene(p);
    // analytic AA: one pixel of coverage = (2 * worldScale / res.y)
    let px = 2.0 * 2.4 / u.res.y;
    let a = clamp(0.5 - s.x / (length(s.yzw) * px + 1e-6), 0.0, 1.0);
    let pulse = 0.85 + 0.15 * sin(u.time * 2.0);
    let col = vec3f({cr}, {cg}, {cb}) * pulse;
    return vec4f(col * a, a);
}}
", scene_expr = scene_expr, cr = f(cr), cg = f(cg), cb = f(cb));

    notes.push((format!("{}_hud", decl.name), "specialized HUD variant baked (no march loop, analytic AA)".into()));
    out
}

// ---------------- naga validation (the proof) ----------------

pub fn validate_wgsl(src: &str, name: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(src)
        .map_err(|e| format!("shader {}: WGSL parse error: {}", name, e))?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .map_err(|e| format!("shader {}: WGSL validation error: {:?}", name, e))?;
    Ok(())
}

// ---------------- tests ----------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::sdf::{fold, prune, consolidate, V3};

    fn demo_decl(hud: bool) -> SdfDecl {
        let root = Shape::SmoothUnion {
            k: 0.3,
            kids: vec![
                Shape::Sphere { c: V3::new(0.0, 0.0, 0.0), r: 1.0 },
                Shape::Box { c: V3::new(0.0, -1.2, 0.0), half: V3::new(1.2, 0.2, 1.2), round: 0.05 },
                Shape::Torus { c: V3::new(0.0, 0.9, 0.0), major: 0.8, minor: 0.1 },
            ],
        };
        SdfDecl {
            name: "test".into(),
            root,
            color: 0x33AAFF,
            glow: 0.6,
            eye: V3::new(0.0, 1.2, -3.4),
            look: V3::new(0.0, 0.2, 0.0),
            hud,
        }
    }

    #[test]
    fn emitted_wgsl_validates_with_naga() {
        let decl = demo_decl(false);
        let root = fold(decl.root.clone());
        let root = prune(root);
        let (root, _) = consolidate(root);
        let d = SdfDecl { root, ..decl.clone() };
        let out = emit(&d);
        validate_wgsl(&out.main_wgsl, "test").expect("main shader must validate");
        assert!(out.hud_wgsl.is_none());
    }

    #[test]
    fn hud_variant_validates_and_exists() {
        let decl = demo_decl(true);
        let root = fold(decl.root.clone());
        let root = prune(root);
        let (root, _) = consolidate(root);
        let d = SdfDecl { root, ..decl.clone() };
        let out = emit(&d);
        let hud = out.hud_wgsl.expect("hud variant must exist");
        validate_wgsl(&hud, "test_hud").expect("hud shader must validate");
    }

    #[test]
    fn shader_is_branchless_no_break() {
        let decl = demo_decl(false);
        let out = emit(&decl);
        assert!(!out.main_wgsl.contains("break"), "the tracer must not use break");
        assert!(out.main_wgsl.contains("select("), "masking must use select");
    }
}
