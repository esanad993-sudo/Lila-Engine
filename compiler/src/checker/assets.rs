//! Asset baking: sprite polygon -> triangle fan -> 14B quantized
//! vertices, and anim curves: RDP simplify + Hermite control points.
//! (split out of the checker monolith)

use super::*;

// ---------------- sprite baking ----------------

pub(crate) fn bake_sprite(sd: &SpriteDecl) -> Result<SpriteInfo, String> {
    let mut verts = Vec::new();
    for poly in &sd.polys {
        let n = poly.pts.len();
        // triangle fan: (v0, vi, vi+1)
        for i in 1..n - 1 {
            let tri = [poly.pts[0], poly.pts[i], poly.pts[i + 1]];
            for (vi, &(x, y)) in tri.iter().enumerate() {
                let frac = if n > 1 { vi as f32 / 2.0 } else { 0.0 }; // 0,0.5,1 across tri roles
                // sprite fill/grad are 24-bit RGB literals per spec: shift into
                // RRGGBB position and bake OPAQUE alpha. Per-draw rgba
                // (0xRRGGBBAA) modulates at runtime.
                let rgba = if let Some(g) = poly.grad {
                    lerp_rgba((poly.fill << 8) | 0xFF, (g << 8) | 0xFF, frac)
                } else { (poly.fill << 8) | 0xFF };
                let tok: u16 = (poly.add as u16) | (match poly.pat {
                    Pat::Flat => 0u16,
                    Pat::Noise => 2u16,
                } << 1);
                let (u, v) = if poly.pat == Pat::Noise {
                    ((x + 32768) as u16, (y + 32768) as u16)
                } else { (0, 0) };
                verts.push(V16 {
                    x: x.clamp(-32768, 32767) as i16,
                    y: y.clamp(-32768, 32767) as i16,
                    u, v, rgba, tok,
                });
            }
        }
    }
    if verts.is_empty() {
        return Err(format!("sprite #{}: no triangles (poly needs >= 3 points)", sd.atom));
    }
    Ok(SpriteInfo { atom: sd.atom, verts })
}

pub(crate) fn lerp_rgba(a: u32, b: u32, t: f32) -> u32 {
    let l = |x: u32, y: u32| -> u32 {
        let d = y as i32 - x as i32;
        (x as i32 + (d as f32 * t).round() as i32).clamp(0, 255) as u32
    };
    (l((a >> 24) & 0xFF, (b >> 24) & 0xFF) << 24)
        | (l((a >> 16) & 0xFF, (b >> 16) & 0xFF) << 16)
        | (l((a >> 8) & 0xFF, (b >> 8) & 0xFF) << 8)
        | l(a & 0xFF, b & 0xFF)
}

// ---------------- anim compression ----------------

pub(crate) fn compress_anim(ad: &AnimDecl) -> Result<AnimInfo, String> {
    let pts: Vec<(f32, f32)> = ad.keys.iter()
        .map(|(t, v)| (*t as f32, *v as f32 / 256.0)).collect();
    let simplified = rdp(&pts, 2.0 / 256.0 * 256.0); // epsilon in Q24.8 units: 2
    let duration = ad.keys.last().unwrap().0 as u16;
    let n = simplified.len();
    let mut segs = Vec::new();
    if n == 1 {
        // constant curve
        let p = (simplified[0].1 * 256.0).round() as i32;
        segs.push(AnimSeg { t0: 0, p0: p, m0: 0, p1: p, m1: 0 });
        return Ok(AnimInfo { atom: ad.atom, duration, segs });
    }
    for i in 0..n - 1 {
        let (t0, v0) = simplified[i];
        let (t1, v1) = simplified[i + 1];
        // Catmull-Rom tangents (per t-unit), clamped at ends
        let m0 = if i == 0 {
            (v1 - v0) / (t1 - t0)
        } else {
            let (tp, vp) = simplified[i - 1];
            (v1 - vp) / (t1 - tp)
        };
        let m1 = if i + 2 == n {
            (v1 - v0) / (t1 - t0)
        } else {
            let (tn, vn) = simplified[i + 2];
            (vn - v0) / (tn - t0)
        };
        segs.push(AnimSeg {
            t0: t0.round() as u16,
            p0: (v0 * 256.0).round() as i32,
            m0: (m0 * 256.0).round() as i32, // fixed per t-unit
            p1: (v1 * 256.0).round() as i32,
            m1: (m1 * 256.0).round() as i32,
        });
    }
    Ok(AnimInfo { atom: ad.atom, duration, segs })
}

pub(crate) fn rdp(pts: &[(f32, f32)], eps: f32) -> Vec<(f32, f32)> {
    if pts.len() < 3 { return pts.to_vec(); }
    let (x0, y0) = pts[0];
    let (xn, yn) = *pts.last().unwrap();
    let mut max_d = 0.0f32;
    let mut idx = 0usize;
    for (i, &(x, y)) in pts.iter().enumerate().skip(1).take(pts.len() - 2) {
        let dx = xn - x0;
        let dy = yn - y0;
        let len2 = dx * dx + dy * dy;
        let d = if len2 < 1e-12 {
            let ex = x - x0; let ey = y - y0;
            (ex * ex + ey * ey).sqrt()
        } else {
            ((dy * (x - x0) - dx * (y - y0)).abs()) / len2.sqrt()
        };
        if d > max_d { max_d = d; idx = i; }
    }
    if max_d <= eps {
        vec![pts[0], *pts.last().unwrap()]
    } else {
        let mut a = rdp(&pts[..idx + 1], eps);
        let b = rdp(&pts[idx..], eps);
        a.pop();
        a.extend(b);
        a
    }
}

