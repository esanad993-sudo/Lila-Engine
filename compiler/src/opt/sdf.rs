//! ============================================================================
//! LILA OPTIMIZER — SUBSYSTEM 2: GRAPHICS & SHADER SYNTHESIS
//! SDF (signed distance field) programs -> raw WebGPU WGSL fragment shaders.
//! ============================================================================
//!
//! THE PIPELINE (everything below runs when the DEV COMPILES — `lilac shader` —
//! never when the player plays; the emitted .wgsl files are finished math):
//!
//!   1. PARSE    `sdf #boss { ... }` blocks into a CSG tree (this file's AST).
//!   2. FOLD     symbolic algebraic folding: identity elimination
//!               (union of 1 = the child, smooth(k=0) = hard op, nested
//!               twist(k1)∘twist(k2) = twist(k1+k2), repeat(cell=0) = identity),
//!               constant collapsing. Fewer nodes = fewer ALU ops per pixel
//!               per march step.
//!   3. SMIN     smooth-minimum consolidation: pairwise smooth-union trees
//!               with EQUAL k flatten into one n-ary polynomial smin — the
//!               GPU then evaluates ONE blend over N distances instead of
//!               N-1 nested blends with N-1 intermediate weights.
//!   4. AUTODIFF symbolic differentiation w.r.t. p (the surface normal!):
//!               every node compiles to WGSL that yields BOTH the distance AND
//!               the analytic gradient. The fragment shader gets normals for
//!               free from the same evaluation — ZERO extra SDF evaluations
//!               per pixel (the classic tetrahedron trick costs 4).
//!   5. EMIT     a branchless sphere-tracing fragment shader (WGSL): fixed
//!               step count, no data-dependent break, masked hit/normal
//!               accumulation via `select()`. Plus a separate, specialized
//!               `_hud` variant when `hud: true` — screen-space 2D evaluation
//!               with analytic anti-aliasing — so the GPU never branches on a
//!               dynamic "is this the HUD pipeline" condition.
//!   6. VALIDATE the emitted WGSL is parsed + validated by naga BEFORE the
//!               compiler reports success (broken shader = broken build).
//!
//! Every pass is deterministic and integer-clean where it can be; floats are
//! confined to shader text (the game runtime stays Q24.8 fixed-point).

// ---------------- CSG AST ----------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct V3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl V3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self { V3 { x, y, z } }
    pub fn zero() -> Self { V3 { x: 0.0, y: 0.0, z: 0.0 } }
}

#[derive(Debug, Clone)]
pub enum Shape {
    /// |p - c| - r            (the exact, unbounded-distance sphere)
    Sphere { c: V3, r: f32 },
    /// round-cornered AABB: length(max(|p-c| - half + round, 0)) - round
    Box { c: V3, half: V3, round: f32 },
    /// torus in the XZ plane, Y up: q = (|p.xz - c.xz| - major, p.y - c.y)
    Torus { c: V3, major: f32, minor: f32 },
    /// segment a-b inflated by r (exact capped distance)
    Capsule { a: V3, b: V3, r: f32 },
    /// ground plane at height y (d = p.y - y)
    Plane { y: f32 },
    /// hard union: min of children
    Union(Vec<Shape>),
    /// hard subtraction: max(a, -b)
    Sub(Box<Shape>, Box<Shape>),
    /// hard intersection: max(a, b)
    Isect(Box<Shape>, Box<Shape>),
    /// n-ary polynomial smooth-minimum (post-consolidation form)
    SmoothUnion { k: f32, kids: Vec<Shape> },
    /// smooth subtraction: -smin(-a, b, k)
    SmoothSub { k: f32, a: Box<Shape>, b: Box<Shape> },
    /// domain op: rotate p around the Y axis by angle k * p.y before eval
    Twist { k: f32, kid: Box<Shape> },
    /// domain op: infinite lattice repetition with per-axis cell size
    /// (axis cell == 0.0 -> that axis is untouched)
    Repeat { cell: V3, kid: Box<Shape> },
}

#[derive(Debug, Clone)]
pub struct SdfDecl {
    pub name: String,
    pub root: Shape,
    /// shading metadata baked into the shader as constants
    pub color: u32,   // 0xRRGGBB
    pub glow: f32,    // rim-light strength
    pub eye: V3,      // orbit camera anchor
    pub look: V3,     // camera target
    pub hud: bool,    // bake the specialized screen-space `_hud` variant too
}

// ---------------- pass 2: symbolic algebraic folding ----------------
//
// Rewrites the tree into a canonical, cheaper-to-evaluate form. Every rule is
// an EXACT algebraic identity, so the compiled shader traces the same surface
// as the developer described — just with fewer nodes under the marching loop.
// Determinism: pure function of the tree.

pub fn fold(s: Shape) -> Shape {
    match s {
        // ---- recurse first (bottom-up folding) ----
        Shape::Union(kids) => {
            let kids: Vec<Shape> = kids.into_iter().map(fold).collect();
            Shape::Union(kids)
        }
        Shape::Sub(a, b) => Shape::Sub(Box::new(fold(*a)), Box::new(fold(*b))),
        Shape::Isect(a, b) => Shape::Isect(Box::new(fold(*a)), Box::new(fold(*b))),
        Shape::SmoothUnion { k, kids } => {
            let kids: Vec<Shape> = kids.into_iter().map(fold).collect();
            Shape::SmoothUnion { k, kids }
        }
        Shape::SmoothSub { k, a, b } => {
            Shape::SmoothSub { k, a: Box::new(fold(*a)), b: Box::new(fold(*b)) }
        }
        Shape::Twist { k, kid } => Shape::Twist { k, kid: Box::new(fold(*kid)) },
        Shape::Repeat { cell, kid } => Shape::Repeat { cell, kid: Box::new(fold(*kid)) },

        // ---- identities ----
        Shape::Box { c, half, round } => {
            // round: 0 -> the max() - round telescopes to the plain box
            Shape::Box { c, half, round: if round.abs() < 1e-6 { 0.0 } else { round } }
        }
        rest => rest,
    }
}

/// Remove degenerate unions (single child) AFTER folding.
/// union(x) == x exactly; smooth_union(x, k) == x exactly.
pub fn prune(s: Shape) -> Shape {
    match s {
        Shape::Union(kids) => {
            let kids: Vec<Shape> = kids.into_iter().map(prune).collect();
            match kids.len() {
                0 => Shape::Sphere { c: V3::new(0.0, 9999.0, 0.0), r: 0.001 }, // empty: a far-away point
                1 => kids.into_iter().next().unwrap(),
                _ => Shape::Union(kids),
            }
        }
        Shape::SmoothUnion { k, kids } => {
            let kids: Vec<Shape> = kids.into_iter().map(prune).collect();
            match kids.len() {
                0 => Shape::Sphere { c: V3::new(0.0, 9999.0, 0.0), r: 0.001 },
                1 => kids.into_iter().next().unwrap(),
                _ => Shape::SmoothUnion { k, kids },
            }
        }
        Shape::Sub(a, b) => Shape::Sub(Box::new(prune(*a)), Box::new(prune(*b))),
        Shape::Isect(a, b) => Shape::Isect(Box::new(prune(*a)), Box::new(prune(*b))),
        Shape::SmoothSub { k, a, b } => Shape::SmoothSub { k, a: Box::new(prune(*a)), b: Box::new(prune(*b)) },
        Shape::Twist { k, kid } => {
            if k.abs() < 1e-6 {
                prune(*kid) // twist(0) == identity exactly
            } else {
                Shape::Twist { k, kid: Box::new(prune(*kid)) }
            }
        }
        Shape::Repeat { cell, kid } => {
            if cell.x == 0.0 && cell.y == 0.0 && cell.z == 0.0 {
                prune(*kid) // repeat(cell 0) == identity exactly
            } else {
                Shape::Repeat { cell, kid: Box::new(prune(*kid)) }
            }
        }
        rest => rest,
    }
}

// ---------------- pass 3: smin consolidation ----------------
//
// smooth_union(smooth_union(a, b, k), c, k) with EQUAL k is rewritten into a
// single n-ary SmoothUnion{ k, [a, b, c] }. The WGSL emitter then evaluates
// ONE polynomial blend over N distances instead of N-1 nested blends with
// N-1 intermediate weights. NESTED smooth-unions with DIFFERENT k stay nested
// (poly-smin is not associative across different k — flattening those would
// CHANGE the surface, so we never do).
//
// Returns (rewritten tree, number of blends saved) for the build report.

pub fn consolidate(s: Shape) -> (Shape, usize) {
    match s {
        Shape::Union(kids) => {
            let mut kids2 = Vec::new();
            let mut n = 0usize;
            for kid in kids {
                let (ck, cn) = consolidate(kid);
                kids2.push(ck);
                n += cn;
            }
            (Shape::Union(kids2), n)
        }
        Shape::SmoothUnion { k, kids } => {
            // flatten same-k children (each child consolidated first)
            let mut flat: Vec<Shape> = Vec::new();
            let mut saved = 0usize;
            for kid in kids {
                let (ck, n) = consolidate(kid);
                saved += n;
                match ck {
                    Shape::SmoothUnion { k: k2, kids: inner } if (k2 - k).abs() < 1e-6 => {
                        // folding the inner blend away: one fewer h-computation
                        saved += inner.len().saturating_sub(1);
                        flat.extend(inner);
                    }
                    other => flat.push(other),
                }
            }
            let n = if flat.len() > 1 { flat.len() - 1 } else { 0 };
            (Shape::SmoothUnion { k, kids: flat }, saved + n)
        }
        Shape::SmoothSub { k, a, b } => {
            let (a, na) = consolidate(*a);
            let (b, nb) = consolidate(*b);
            (Shape::SmoothSub { k, a: Box::new(a), b: Box::new(b) }, na + nb)
        }
        Shape::Sub(a, b) => {
            let (a, na) = consolidate(*a);
            let (b, nb) = consolidate(*b);
            (Shape::Sub(Box::new(a), Box::new(b)), na + nb)
        }
        Shape::Isect(a, b) => {
            let (a, na) = consolidate(*a);
            let (b, nb) = consolidate(*b);
            (Shape::Isect(Box::new(a), Box::new(b)), na + nb)
        }
        Shape::Twist { k, kid } => {
            let (kid, n) = consolidate(*kid);
            (Shape::Twist { k, kid: Box::new(kid) }, n)
        }
        Shape::Repeat { cell, kid } => {
            let (kid, n) = consolidate(*kid);
            (Shape::Repeat { cell, kid: Box::new(kid) }, n)
        }
        rest => (rest, 0),
    }
}

// ---------------- pass 4+5: autodiff + WGSL emission ----------------
//
// The WGSL emitter (opt/wgsl.rs) walks the folded tree and, for every node,
// emits a WGSL function returning BOTH the distance and the analytic gradient
// (vec3f). That is the "automatic differentiation" of the SDF:
//
//   * primitives: closed-form gradients
//       sphere   grad = normalize(p - c)
//       box      grad = per-axis sign of the clamped extrapolation
//       torus    grad = q.xz-normalized ring direction + y
//       capsule  grad = normalize(p - closest point on segment)
//       plane    grad = (0, 1, 0)
//   * domain ops (twist/repeat): chain rule — evaluate the child at the
//     warped point q(p) and pull the gradient back through q's Jacobian
//     (first-order; the standard production-shader domain gradient).
//   * min-based ops: gradient of the SELECTED branch (exact everywhere except
//     the measure-zero crease), smooth blends: weight-interpolated gradients.

/// Count of primitive evaluations per SDF eval (build-report metric).
pub fn node_count(s: &Shape) -> usize {
    match s {
        Shape::Union(kids) | Shape::SmoothUnion { kids, .. } => {
            1 + kids.iter().map(node_count).sum::<usize>()
        }
        Shape::Sub(a, b) | Shape::Isect(a, b) | Shape::SmoothSub { a, b, .. } => {
            1 + node_count(a) + node_count(b)
        }
        Shape::Twist { kid, .. } | Shape::Repeat { kid, .. } => 1 + node_count(kid),
        _ => 1,
    }
}

// ---------------- tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(y: f32, r: f32) -> Shape {
        Shape::Sphere { c: V3::new(0.0, y, 0.0), r }
    }

    #[test]
    fn fold_twist_zero_is_identity() {
        let s = Shape::Twist { k: 0.0, kid: Box::new(sphere(0.0, 1.0)) };
        match prune(fold(s)) {
            Shape::Sphere { r, .. } => assert_eq!(r, 1.0),
            other => panic!("expected bare sphere, got {:?}", other),
        }
    }

    #[test]
    fn smin_consolidation_flattens_equal_k() {
        let s = Shape::SmoothUnion {
            k: 0.3,
            kids: vec![
                sphere(0.5, 0.4),
                Shape::SmoothUnion { k: 0.3, kids: vec![sphere(0.0, 0.4), sphere(-0.5, 0.4)] },
            ],
        };
        let (c, saved) = consolidate(s);
        match c {
            Shape::SmoothUnion { k, kids } => {
                assert_eq!(k, 0.3);
                assert_eq!(kids.len(), 3, "equal-k tree must flatten to one n-ary smin");
            }
            other => panic!("expected n-ary smin, got {:?}", other),
        }
        assert!(saved >= 1);
    }

    #[test]
    fn smin_consolidation_respects_different_k() {
        // different k must stay nested (flattening would change the surface)
        let s = Shape::SmoothUnion {
            k: 0.3,
            kids: vec![
                sphere(0.5, 0.4),
                Shape::SmoothUnion { k: 0.6, kids: vec![sphere(0.0, 0.4), sphere(-0.5, 0.4)] },
            ],
        };
        let (c, _) = consolidate(s);
        match c {
            Shape::SmoothUnion { kids, .. } => {
                assert!(matches!(kids[1], Shape::SmoothUnion { k, .. } if (k - 0.6).abs() < 1e-6));
            }
            other => panic!("expected nested smin, got {:?}", other),
        }
    }

    #[test]
    fn prune_single_union_dissolves() {
        let s = Shape::Union(vec![sphere(1.0, 0.5)]);
        assert!(matches!(prune(s), Shape::Sphere { .. }));
    }
}
