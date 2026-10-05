//! ============================================================================
//! LILA OPTIMIZER — SUBSYSTEM 4: AUDIO & WAVEFORMS (SIGNAL SPLINE SYNTHESIS)
//! ============================================================================
//!
//! THE DEV DROPS a recording next to the game and writes ONE line:
//!
//!     sfx #explosion { from: "assets/explosion.wav" }
//!
//! THE COMPILER (at build time — the player never sees a .wav, the game
//! ships ~10 bytes of fitted parameters, ZERO sample RAM):
//!
//!   1. PARSE PCM (RIFF/WAVE, 16-bit, mono/stereo->mono), trim silence,
//!      normalize.
//!   2. MEASURE — Hann-window radix-2 FFT at two analysis frames -> spectral
//!      peak picking (parabolic interpolation) -> partials; RMS envelope ->
//!      attack/decay; two-frame fundamental drift -> sweep.
//!   3. DECIDE THE MODEL —
//!        harmonic sideband structure  ->  FM:  f(t) = A(t)·sin(2π f0(t) t +
//!              I·sin(2π r f0 t))          (carrier/modulator parameters)
//!        inharmonic partials          ->  ADDITIVE TRIG SPLINE:
//!              f(t) = A(t)·Σ aᵢ·sin(2π rᵢ f0 t + φᵢ)   (≤ 4 partials)
//!      both with the engine's linear A(t) envelope (the same one the legacy
//!      sfx programs use), so the runtime voice code is shared.
//!   4. VERIFY (ground truth loop) — the fitted parameters are played back
//!      through a Rust mirror of the runtime's INTEGER synthesizer and the
//!      residual against the original PCM is printed. A bad fit is a loud
//!      build diagnostic, never a silent surprise.
//!   5. BAKE — params go into the .libyte v3 sfx section; the runtime
//!      synthesizes f(t) sample-by-sample from two phase accumulators
//!      (FM) or four (additive). No buffers, no decoding, O(1) per sample,
//!      instant pitch-shift (retune f0) and time-stretch (retune decay) —
//!      the "zero-buffer synthesis loop" mandate.
//!
//! Everything here is dev-time. The runtime's synth lives in
//! runtime/src/audio.zig and consumes ONLY the baked numbers.


pub const ANALYSIS_SR: u32 = 44100;

// ---------------- fitted models (the wire format mirrored in Zig) ----------------

/// FM voice:  out(t) = env(t)·[ (1-nz)·sin(φc + I·sin(φm)) + nz·noise ]
#[derive(Debug, Clone)]
pub struct FmParams {
    pub f0: u16,       // carrier base freq, Hz
    pub sweep: i16,    // Hz per second drift of the carrier
    pub ratio_q4: u8,  // modulator ratio r = ratio_q4 / 16  (1..=64 -> 0.0625..4.0)
    pub index_q4: u8,  // modulation index I = index_q4 / 16 (0..=64 -> 0..4 rad)
    pub decay: u8,     // envelope length in 1/60 s ticks (engine convention)
    pub vol: u8,       // 0..15
    pub noise: u8,     // 0..16 noise mix fraction
}

/// Additive trig-spline voice: out(t) = env(t)·Σ aᵢ·sin(2π rᵢ f0 t + φᵢ)
#[derive(Debug, Clone)]
pub struct AddParams {
    pub f0: u16,       // fundamental, Hz
    pub decay: u8,
    pub vol: u8,
    /// (ratio_q6 = fᵢ/f0 × 64, amp 0..=16, phase 0..=255 -> 0..2π)
    pub partials: Vec<(u8, u8, u8)>, // 1..=4 entries
}

#[derive(Debug, Clone)]
pub enum FittedModel {
    Fm(FmParams),
    Add(AddParams),
}

#[derive(Debug, Clone)]
pub struct FittedSfx {
    pub model: FittedModel,
    /// RMS residual of the runtime-mirror playback vs the original, as a
    /// fraction of the original RMS (0 = perfect).
    pub residual: f32,
    /// human-readable fitted equation for the build report
    pub equation: String,
}

// ---------------- WAV parsing ----------------

pub struct Pcm {
    pub samples: Vec<f32>, // mono, -1..1
    pub sr: u32,
}

pub fn parse_wav(bytes: &[u8]) -> Result<Pcm, String> {
    let rd_u16 = |p: usize| -> Result<u16, String> {
        bytes.get(p..p + 2).map(|s| u16::from_le_bytes([s[0], s[1]])).ok_or("wav truncated".into())
    };
    let rd_u32 = |p: usize| -> Result<u32, String> {
        bytes.get(p..p + 4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            .ok_or("wav truncated".into())
    };
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32)> = None; // (format, channels, sr)
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = rd_u32(pos + 4)? as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(sz).min(bytes.len());
        match id {
            b"fmt " => {
                let audio_format = rd_u16(body_start)?;
                let channels = rd_u16(body_start + 2)?;
                let sr = rd_u32(body_start + 4)?;
                fmt = Some((audio_format, channels, sr));
            }
            b"data" => data = Some(&bytes[body_start..body_end]),
            _ => {}
        }
        pos = body_start + sz + (sz & 1); // chunks are word-aligned
    }
    let (format, channels, sr) = fmt.ok_or("wav: missing fmt chunk")?;
    if format != 1 {
        return Err(format!("wav: unsupported format {} (PCM only)", format));
    }
    if channels == 0 || channels > 2 {
        return Err(format!("wav: {} channels unsupported", channels));
    }
    if !(4000..=192000).contains(&sr) {
        return Err(format!("wav: sample rate {} out of range", sr));
    }
    let raw = data.ok_or("wav: missing data chunk")?;
    let mut samples = Vec::with_capacity(raw.len() / 2 / channels as usize);
    let mut i = 0;
    while i + 2 * channels as usize <= raw.len() {
        let mut acc = 0i32;
        for ch in 0..channels as usize {
            let off = i + ch * 2;
            let s = i16::from_le_bytes([raw[off], raw[off + 1]]);
            acc += s as i32;
        }
        samples.push(acc as f32 / (channels as f32 * 32768.0));
        i += 2 * channels as usize;
    }
    if samples.is_empty() {
        return Err("wav: no samples".into());
    }
    Ok(Pcm { samples, sr })
}

fn resample(pcm: &Pcm, sr: u32) -> Vec<f32> {
    if pcm.sr == sr {
        return pcm.samples.clone();
    }
    let n = ((pcm.samples.len() as u64) * sr as u64 / pcm.sr as u64) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 * pcm.sr as f64 / sr as f64;
        let i0 = t as usize;
        let frac = (t - i0 as f64) as f32;
        let a = pcm.samples[i0.min(pcm.samples.len() - 1)];
        let b = pcm.samples[(i0 + 1).min(pcm.samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

// ---------------- FFT (radix-2, f64, iterative) ----------------

fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    // bit reversal
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j & m != 0 {
            j ^= m;
            m >>= 1;
        }
        j |= m;
    }
    // butterflies
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let a = i + k;
                let b = i + k + len / 2;
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Magnitude spectrum with a Hann window (zero-padded to N).
fn spectrum(x: &[f32], n_fft: usize) -> Vec<f32> {
    let n = n_fft;
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    for i in 0..x.len().min(n) {
        let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (x.len().max(2) - 1) as f64).cos();
        re[i] = x[i] as f64 * w;
    }
    fft(&mut re, &mut im);
    (0..n / 2).map(|i| ((re[i] * re[i] + im[i] * im[i]).sqrt()) as f32).collect()
}

/// Parabolic peak interpolation: returns (freq, magnitude) of local maxima.
fn spectral_peaks(mag: &[f32], sr: u32, n_fft: usize, max_peaks: usize) -> Vec<(f32, f32)> {
    let bin_hz = sr as f32 / n_fft as f32;
    let mmax = mag.iter().cloned().fold(0.0f32, f32::max);
    if mmax <= 0.0 {
        return vec![];
    }
    let mut peaks: Vec<(f32, f32)> = Vec::new();
    for i in 1..mag.len() - 1 {
        if mag[i] > mag[i - 1] && mag[i] >= mag[i + 1] && mag[i] > 0.08 * mmax {
            // parabolic interp on log-magnitudes
            let (a, b, c) = (
                (mag[i - 1] as f64 + 1e-12).ln(),
                (mag[i] as f64 + 1e-12).ln(),
                (mag[i + 1] as f64 + 1e-12).ln(),
            );
            let denom = a - 2.0 * b + c;
            let delta = if denom.abs() > 1e-12 { 0.5 * (a - c) / denom } else { 0.0 };
            let freq = (i as f32 + delta as f32) * bin_hz;
            let amp = mag[i];
            peaks.push((freq, amp));
        }
    }
    peaks.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(max_peaks);
    peaks.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    peaks
}

// ---------------- analysis + fitting ----------------

fn rms_windows(x: &[f32], win: usize) -> Vec<f32> {
    (0..x.len() / win).map(|i| {
        let s = &x[i * win..(i + 1) * win];
        (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt()
    }).collect()
}

/// Fit one .wav into an O(1) synthesis program. `name` is for error messages.
pub fn fit(name: &str, pcm: &Pcm) -> Result<FittedSfx, String> {
    let sr = ANALYSIS_SR;
    let x = resample(pcm, sr);

    // ---- trim + normalize ----
    let peak = x.iter().cloned().fold(0.0f32, f32::max).max(-x.iter().cloned().fold(0.0f32, f32::max));
    if peak < 1e-5 {
        return Err(format!("{}: wav is (near) silent", name));
    }
    let x: Vec<f32> = x.iter().map(|v| v / peak).collect();
    let win = (sr as usize / 200).max(1); // 5 ms
    let env = rms_windows(&x, win);
    let ethr = 0.02f32;
    let start = env.iter().position(|v| *v > ethr).unwrap_or(0) * win;
    let end = env.iter().rposition(|v| *v > ethr).map(|p| (p + 1) * win).unwrap_or(x.len());
    let x = &x[start.min(x.len())..end.min(x.len()).max(start + win)];
    if x.len() < 256 {
        return Err(format!("{}: usable audio shorter than 6 ms", name));
    }
    let x: Vec<f32> = if x.len() > sr as usize { x[..sr as usize].to_vec() } else { x.to_vec() };

    // ---- envelope: decay length = time until RMS falls under 6% of peak RMS ----
    let env = rms_windows(&x, win);
    let peak_rms = env.iter().cloned().fold(0.0f32, f32::max);
    let t_peak = env.iter().position(|v| *v >= peak_rms * 0.9).unwrap_or(0);
    let decay_end = env.iter().rposition(|v| *v > peak_rms * 0.06).unwrap_or(env.len() - 1);
    let dur_s = (x.len() as f32) / sr as f32;
    let decay_s = (((decay_end.max(t_peak) - t_peak) * win) as f32 / sr as f32).max(0.03).min(1.0);
    let decay_ticks = (decay_s * 60.0).round().clamp(1.0, 255.0) as u8;
    let _ = dur_s;

    // ---- two spectral frames (early + late) for the sweep ----
    let n_fft = 4096usize;
    let f1_at = (x.len() / 4).max(1);
    let f2_at = (x.len() * 3 / 4).max(f1_at + 1);
    let take = |from: usize| -> Vec<f32> {
        let l = n_fft.min(x.len().saturating_sub(from).max(64));
        x[from..from + l].to_vec()
    };
    let p1 = spectral_peaks(&spectrum(&take(f1_at), n_fft), sr, n_fft, 8);
    let p2 = spectral_peaks(&spectrum(&take(f2_at), n_fft), sr, n_fft, 8);
    if p1.is_empty() {
        return Err(format!("{}: no spectral content found", name));
    }

    // fundamental: lowest strong peak of the early frame
    let m1max = p1.iter().map(|p| p.1).fold(0.0f32, f32::max);
    let f0 = p1.iter().find(|p| p.1 > 0.25 * m1max).map(|p| p.0).unwrap_or(p1[0].0);
    if !(20.0..=8000.0).contains(&f0) {
        return Err(format!("{}: fundamental {:.0} Hz out of range", name, f0));
    }

    // ---- harmonic sideband structure? ----
    // FM produces sidebands at f0 ± k·r·f0; additive (non-FM) content sits at
    // near-integer multiples with slowly decaying amplitude. We score the
    // regularity of adjacent-peak spacing: consistent spacing != f0/2-ish
    // and >= 3 peaks -> FM.
    let spacing: Vec<f32> = p1.windows(2).map(|w| w[1].0 - w[0].0).filter(|d| *d > 15.0).collect();
    let med_spacing = median(&spacing);
    let spacing_consistent = spacing.len() >= 2
        && spacing.iter().all(|d| (*d - med_spacing).abs() < 0.15 * med_spacing.max(1.0));
    let use_fm = spacing_consistent && med_spacing > 0.4 * f0 && med_spacing < 3.5 * f0 && f0 > 60.0;

    let (mut model, mut equation) = if use_fm {
        let ratio = (med_spacing / f0).clamp(0.0625, 4.0);
        let ratio_q4 = (ratio * 16.0).round().clamp(1.0, 64.0) as u8;
        // Modulation index from BESSEL RATIOS: for sinusoidal FM the spectrum
        // is A_n = J_n(I) at f0 + n·r·f0, so rho = A_1/A_0 = J1(I)/J0(I) —
        // monotone in I, invertible by bisection. When r ≈ 2 the FIRST LOWER
        // sideband (f0 - r·f0 = -f0) folds onto the carrier: the measured
        // carrier becomes J0+J1 and the estimator uses that model instead.
        let bin_hz = sr as f32 / n_fft as f32;
        let carrier_bin = (f0 / bin_hz).round() as i64;
        let mag = spectrum(&take(f1_at), n_fft);
        let amp_at = |center: i64| -> f32 {
            (-2..=2).map(|k| {
                let b = (center + k).clamp(0, mag.len() as i64 - 1) as usize;
                mag[b]
            }).fold(0.0f32, f32::max)
        };
        let a0 = amp_at(carrier_bin);
        let sp = (med_spacing / bin_hz).round() as i64;
        let a1 = amp_at(carrier_bin + sp).max(amp_at(carrier_bin - sp));
        let rho = (a1 / a0.max(1e-9)) as f64;
        let folded = (ratio - 2.0).abs() < 0.25; // lower sideband lands on f0
        let idx0 = invert_bessel_ratio(rho, folded).clamp(0.0, 4.0);
        // ANALYSIS-BY-SYNTHESIS refinement of the modulation index: the
        // Bessel ratio is only a seed (folded spectra stack several Bessel
        // orders per bin, finite windows smear peaks). The compiler tries
        // every quantized I in a neighborhood through the INTEGER runtime
        // mirror and keeps the one that minimizes the actual residual —
        // the estimator directly optimizes the metric we ship. ~17 mirror
        // renders of 4096 samples: sub-millisecond at build time.
        let seg_len = n_fft.min(x.len());
        let seg: Vec<f32> = x[..seg_len].to_vec();
        let seed_q4 = (idx0 * 16.0).round() as i32;
        let mut best_q4 = seed_q4.clamp(0, 64);
        let mut best_r = f32::MAX;
        for dq in -8i32..=8 {
            let q4 = (seed_q4 + dq).clamp(0, 64);
            let trial = FittedModel::Fm(FmParams {
                f0: f0.round().clamp(20.0, 65535.0) as u16,
                sweep: 0, ratio_q4, index_q4: q4 as u8,
                decay: decay_ticks, vol: vol_from_peak(peak, peak_rms), noise: 0,
            });
            let sig = synth_reference(&trial, seg_len);
            let r = rms_residual(&sig, &seg);
            if r < best_r { best_r = r; best_q4 = q4; }
        }
        let index_q4 = best_q4 as u8;
        // sweep from the fundamental's drift between frames
        let f0_2 = p2.iter().find(|p| (p.0 - f0).abs() < 0.5 * f0.max(1.0))
            .map(|p| p.0).unwrap_or(f0);
        let dt = (f2_at - f1_at) as f32 / sr as f32;
        let sweep = (((f0_2 - f0) / dt.max(1e-3)).round() as i64).clamp(-32768, 32767) as i16;
        let vol = vol_from_peak(peak, peak_rms);
        (
            FittedModel::Fm(FmParams {
                f0: f0.round().clamp(20.0, 65535.0) as u16,
                sweep, ratio_q4, index_q4, decay: decay_ticks, vol, noise: 0,
            }),
            format!(
                "f(t) = A(t)·sin(2π·{}Hz·t + {:.2}·sin(2π·{:.3}·{}Hz·t))  sweep {:+}Hz/s",
                f0.round(), best_q4 as f32 / 16.0, ratio_q4 as f32 / 16.0, f0.round(), sweep
            ),
        )
    } else {
        // additive trig spline: top-4 partials by magnitude, ratios to f0
        let mut parts: Vec<(f32, f32)> = p1.clone();
        if parts.is_empty() {
            parts.push((f0, m1max));
        }
        parts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        parts.truncate(4);
        parts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let fmin = parts[0].0.max(20.0);
        let amax = parts.iter().map(|p| p.1).fold(0.0f32, f32::max).max(1e-9);
        let partials: Vec<(u8, u8, u8)> = parts.iter().map(|(f, a)| {
            let ratio_q6 = ((f / fmin) * 64.0).round().clamp(1.0, 255.0) as u8;
            let amp = (a / amax * 16.0).round().clamp(1.0, 16.0) as u8;
            (ratio_q6, amp, 0u8) // phase fit: modeled at 0 (perceptually negligible for decaying impacts)
        }).collect();
        let vol = vol_from_peak(peak, peak_rms);
        let desc = partials.iter()
            .map(|(r, a, _)| format!("{:.2}x{}@", *r as f32 / 64.0, a))
            .collect::<Vec<_>>().join("+");
        (
            FittedModel::Add(AddParams {
                f0: fmin.round().clamp(20.0, 65535.0) as u16,
                decay: decay_ticks, vol, partials,
            }),
            format!("f(t) = A(t)·Σ aᵢ·sin(2π·rᵢ·{}Hz·t)  [{}]", fmin.round(), desc),
        )
    };

    // ---- joint full-length refinement (see refine_joint docs): pick the
    // (decay, index, vol) that minimize the SHIPPED full-length residual.
    refine_joint(&mut model, &x, sr, decay_s);
    if let FittedModel::Fm(p) = &model {
        equation = format!(
            "f(t) = A(t)·sin(2π·{}Hz·t + {:.2}·sin(2π·{:.3}·{}Hz·t))  sweep {:+}Hz/s",
            p.f0, p.index_q4 as f32 / 16.0, p.ratio_q4 as f32 / 16.0, p.f0, p.sweep
        );
    }

    // ---- ground-truth loop: play the model through the runtime mirror ----
    let n = (decay_s * sr as f32).round().clamp(256.0, 96000.0) as usize;
    let model_signal = synth_reference(&model, n);
    let original: Vec<f32> = x.iter().copied().chain(std::iter::repeat(0.0)).take(n).collect();
    let residual = rms_residual(&model_signal, &original).min(9.9);

    Ok(FittedSfx { model, residual, equation })
}

fn vol_from_peak(peak: f32, peak_rms: f32) -> u8 {
    // crest-aware volume: RMS carries the loudness, peak caps the headroom
    let v = (peak_rms.max(peak * 0.3) * 24.0).round();
    v.clamp(4.0, 15.0) as u8
}

fn median(v: &[f32]) -> f32 {
    if v.is_empty() { return 0.0; }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[s.len() / 2]
}

// ---------------- joint full-length parameter refinement ----------------
//
// The seed estimators above run on short spectral frames and a 4096-sample
// segment with decay/vol frozen from envelope heuristics — but the metric we
// SHIP is the full-length runtime-mirror RMS residual. This stage chooses the
// parameters that minimize EXACTLY that metric. Runtime semantics are
// untouched: every candidate plays through the same integer mirror the
// player's mixer runs, so we are only picking better points in the space the
// voice format can already express.

fn residual_full(model: &FittedModel, orig: &[f32], e0: f32) -> f32 {
    let m = synth_reference(model, orig.len());
    let e: f32 = m.iter().zip(orig.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
    (e / e0.max(1e-9)).sqrt().min(9.9)
}

fn model_decay(model: &FittedModel) -> u8 {
    match model { FittedModel::Fm(p) => p.decay, FittedModel::Add(p) => p.decay }
}

fn set_decay(model: &mut FittedModel, d: u8) {
    match model { FittedModel::Fm(p) => p.decay = d, FittedModel::Add(p) => p.decay = d }
}

fn model_vol(model: &FittedModel) -> u8 {
    match model { FittedModel::Fm(p) => p.vol, FittedModel::Add(p) => p.vol }
}

fn set_vol(model: &mut FittedModel, v: u8) {
    match model { FittedModel::Fm(p) => p.vol = v, FittedModel::Add(p) => p.vol = v }
}

/// Amplitude projection: render at vol=15, solve the least-squares scale
/// g* = <m15, orig> / <m15, m15> (vol rides LINEARLY in the voice output),
/// then try the integer vols around 15·g*. This replaces the RMS-heuristic
/// vol with the quantized amplitude closest to the true optimum.
fn project_vol(model: &mut FittedModel, orig: &[f32], e0: f32, best_r: &mut f32, best: &mut FittedModel) {
    let v0 = model_vol(model);
    set_vol(model, 15);
    let m15 = synth_reference(model, orig.len());
    set_vol(model, v0);
    let mm: f32 = m15.iter().map(|v| v * v).sum();
    if mm < 1e-12 { return; }
    let mo: f32 = m15.iter().zip(orig.iter()).map(|(a, b)| a * b).sum();
    let v_ideal = (mo / mm * 15.0).round().clamp(4.0, 15.0) as u8;
    let cands = [v_ideal.saturating_sub(1).max(4), v_ideal, (v_ideal + 1).min(15)];
    for v in cands {
        set_vol(model, v);
        let r = residual_full(model, orig, e0);
        if r < *best_r { *best_r = r; *best = model.clone(); }
    }
}

/// Full-length joint refinement:
///   stage 1 — decay grid (±8 ticks), vol projection per step;
///   stage 2 — FM only: exhaustive modulation-index sweep (0..=64 quanta) at
///             the winning decay;
///   stage 3 — final vol re-projection at the winning shape.
/// ~110 full-length mirror renders, still well under a millisecond.
fn refine_joint(model: &mut FittedModel, x: &[f32], sr: u32, decay_s: f32) {
    let n = (decay_s * sr as f32).round().clamp(256.0, 96000.0) as usize;
    let orig: Vec<f32> = x.iter().copied().chain(std::iter::repeat(0.0)).take(n).collect();
    let e0: f32 = orig.iter().map(|b| b * b).sum::<f32>().max(1e-9);
    let mut best = model.clone();
    let mut best_r = residual_full(model, &orig, e0);

    let d0 = model_decay(model);
    for d in (d0.saturating_sub(8)).max(1)..=(d0.saturating_add(8)).min(255) {
        set_decay(model, d);
        project_vol(model, &orig, e0, &mut best_r, &mut best);
    }

    if matches!(model, FittedModel::Fm(_)) {
        let d_best = model_decay(&best);
        let i0 = match model { FittedModel::Fm(p) => p.index_q4, _ => 0 };
        let v_in = model_vol(&best);
        let mut trial = model.clone();
        if let FittedModel::Fm(tp) = &mut trial {
            tp.decay = d_best;
            tp.vol = v_in;
        }
        for dq in -64i32..=64 {
            let q = (i0 as i32 + dq).clamp(0, 64) as u8;
            if q == i0 { continue; }
            if let FittedModel::Fm(tp) = &mut trial { tp.index_q4 = q; }
            let r = residual_full(&trial, &orig, e0);
            if r < best_r { best_r = r; best = trial.clone(); }
        }
    }

    {
        let d_win = model_decay(&best);
        let i_win = match &best { FittedModel::Fm(p) => p.index_q4, _ => 0 };
        let mut trial = best.clone();
        set_decay(&mut trial, d_win);
        if let FittedModel::Fm(tp) = &mut trial { tp.index_q4 = i_win; }
        project_vol(&mut trial, &orig, e0, &mut best_r, &mut best);
    }

    *model = best;
}

/// RMS of (model - original) relative to the original's RMS — the fit metric
/// reported for every sfx (0 = the baked f(t) reproduces the PCM exactly).
fn rms_residual(model: &[f32], orig: &[f32]) -> f32 {
    let n = model.len().min(orig.len());
    let e: f32 = model[..n].iter().zip(&orig[..n])
        .map(|(a, b)| (a - b) * (a - b)).sum();
    let e0: f32 = orig[..n].iter().map(|b| b * b).sum::<f32>().max(1e-9);
    (e / e0).sqrt()
}

/// Bessel J0/J1 (ascending series — exact to ~1e-12 over 0..3).
fn bessel_j0(x: f64) -> f64 {
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    let mut k = 1usize;
    while k < 24 {
        term *= -(x * x / 4.0) / (k * k) as f64;
        sum += term;
        if term.abs() < 1e-12 { break; }
        k += 1;
    }
    sum
}

fn bessel_j1(x: f64) -> f64 {
    let mut term = x / 2.0;
    let mut sum = term;
    let mut k = 1usize;
    while k < 24 {
        term *= -(x * x / 4.0) / (k as f64 * (k + 1) as f64);
        sum += term;
        if term.abs() < 1e-12 { break; }
        k += 1;
    }
    sum
}

/// Invert rho = J1(I)/J0(I) (or J1(I)/(J0+J1) for the r=2 folded spectrum)
/// for I in 0..2.3 — both are monotone increasing there. Bisection, 48 iters.
fn invert_bessel_ratio(rho: f64, folded: bool) -> f64 {
    let model = |i: f64| -> f64 {
        let (j0, j1) = (bessel_j0(i), bessel_j1(i));
        if folded { j1 / (j0 + j1).max(1e-9) } else { j1 / j0.max(1e-9) }
    };
    let (mut lo, mut hi) = (0.0f64, 2.3f64);
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if model(mid) < rho { lo = mid; } else { hi = mid; }
    }
    0.5 * (lo + hi)
}

// ---------------- the runtime-mirror synthesizer (verification) ----------------
//
// This mirrors runtime/src/audio.zig's FM/ADD voices EXACTLY (same integer
// phases, same 4096-entry Q15 sine LUT, same linear envelope) so the residual
// measures what the player will actually hear, not an idealized float model.

pub const LUT_BITS: u32 = 12;
pub const LUT_SIZE: usize = 1 << LUT_BITS;

pub fn sin_lut() -> Vec<i16> {
    (0..LUT_SIZE).map(|i| {
        let a = i as f64 / LUT_SIZE as f64 * 2.0 * std::f64::consts::PI;
        (a.sin() * 32767.0) as i16
    }).collect()
}

fn phase_step(hz: f64) -> u32 {
    ((hz * 4294967296.0) / ANALYSIS_SR as f64) as u32
}

/// Integer-exact playback of a fitted model (the Zig runtime does this in
/// its mixer; this is the compiler-side mirror used for the residual check).
pub fn synth_reference(model: &FittedModel, n: usize) -> Vec<f32> {
    let lut = sin_lut();
    let s15 = |ph: u32| -> i32 { lut[((ph >> 20) & 0xFFF) as usize] as i32 };
    match model {
        FittedModel::Fm(p) => {
            let mut pc: u32 = 0;
            let mut pm: u32 = 0;
            let mut ns: u32 = 0x2545F491;
            let decay_samples = (p.decay as i64 * ANALYSIS_SR as i64 / 60) as i64;
            // Q15 sine units -> 32-bit phase units: dev·2^32/(2π·2^15).
            // 20861 = 4294967296 / (2π·32768). The modulation index rides in
            // dev as I·sin(φm)·2^15 — this constant converts it EXACTLY once.
            let fm_k: i64 = 20861;
            (0..n).map(|age| -> f32 {
                if age as i64 >= decay_samples {
                    return 0.0;
                }
                // INTEGER instantaneous freqs — bit-identical to the runtime
                // voice in runtime/src/audio.zig (divTrunc semantics).
                let f_now: i64 = p.f0 as i64 + p.sweep as i64 * age as i64 / ANALYSIS_SR as i64;
                let fmod: i64 = (f_now * p.ratio_q4 as i64 / 16).max(0);
                pc = pc.wrapping_add(phase_step(f_now as f64));
                pm = pm.wrapping_add(phase_step(fmod as f64));
                let dev: i64 = s15(pm) as i64 * p.index_q4 as i64 / 16;
                let off = ((dev * fm_k) as u64) as u32; // wrapping add below
                let carr = pc.wrapping_add(off);
                let mut out = (s15(carr) >> 7) as f64 / 256.0;
                if p.noise > 0 {
                    ns ^= ns << 13;
                    ns ^= ns >> 17;
                    ns ^= ns << 5;
                    let nz = if ns & 0x80000000 != 0 { 1.0 } else { -1.0 };
                    out = out * (16 - p.noise) as f64 / 16.0 + nz * p.noise as f64 / 16.0;
                }
                // 1/16-quantized linear envelope — the engine's voice convention
                let env16: i64 = (decay_samples - age as i64) * 16 / decay_samples;
                (out * env16 as f64 * (p.vol as f64 / 15.0) / 16.0) as f32
            }).collect()
        }
        FittedModel::Add(p) => {
            let mut phases = [0u32; 4];
            let decay_samples = (p.decay as i64 * ANALYSIS_SR as i64 / 60) as i64;
            let steps: Vec<u32> = p.partials.iter().map(|(r, _, _)| {
                phase_step(p.f0 as f64 * *r as f64 / 64.0)
            }).collect();
            (0..n).map(|age| -> f32 {
                if age as i64 >= decay_samples {
                    return 0.0;
                }
                let mut acc: i64 = 0;
                for (i, (_, amp, ph0)) in p.partials.iter().enumerate() {
                    phases[i] = phases[i].wrapping_add(steps[i])
                        .wrapping_add(if age == 0 {
                            (*ph0 as u32) << 24
                        } else { 0 });
                    acc += (s15(phases[i]) >> 7) as i64 * *amp as i64;
                }
                let out = acc as f64 / 256.0 / 16.0; // amps sum <= 16
                // 1/16-quantized linear envelope — the engine's voice convention
                let env16: i64 = (decay_samples - age as i64) * 16 / decay_samples;
                (out * env16 as f64 * (p.vol as f64 / 15.0) / 16.0) as f32
            }).collect()
        }
    }
}

// ---------------- tests (ground-truth fixtures) ----------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesize an FM tone with KNOWN parameters through an independent
    /// float model, write PCM bytes in-memory, fit them back, and require the
    /// recovered parameters to land near the originals (the compiler "hears"
    /// what it was played). The fixture uses the LINEAR envelope the engine's
    /// voices implement (the model class under test) — an exponential-decay
    /// fixture would fail on envelope shape, not on parameter recovery.
    #[test]
    fn wavfit_recovers_known_fm() {
        let sr = 44100.0f32;
        let n = (0.7 * sr) as usize;
        let f0 = 220.0f32;
        let ratio = 2.0f32;
        let index = 1.5f32;
        let dur = 0.35f32; // linear decay over 0.35 s (= 21 engine ticks)
        let pcm: Vec<f32> = (0..n).map(|i| {
            let t = i as f32 / sr;
            let env = (1.0 - t / dur).max(0.0);
            env * (2.0 * std::f32::consts::PI * f0 * t
                + index * (2.0 * std::f32::consts::PI * ratio * f0 * t).sin()).sin()
        }).collect();
        let fitted = fit("fixture-fm", &Pcm { samples: pcm, sr: sr as u32 }).expect("fit ok");
        // recovered frequency must be close (peak interp + frame effects)
        match &fitted.model {
            FittedModel::Fm(p) => {
                let got = p.f0 as f32;
                assert!((got - f0).abs() < 0.15 * f0, "f0 {:.0} vs {:.0}", got, f0);
                assert!((p.ratio_q4 as f32 / 16.0 - ratio).abs() < 0.5,
                    "ratio {:.2} vs {:.2}", p.ratio_q4 as f32 / 16.0, ratio);
                // residual must be small: the model CAN express this tone
                assert!(fitted.residual < 0.35, "residual {:.2}", fitted.residual);
            }
            other => panic!("expected FM model, got {:?}", other),
        }
    }

    #[test]
    fn wavfit_recovers_additive_pure_sine() {
        let sr = 44100u32;
        let n = sr as usize / 2;
        let f0 = 440.0f32;
        let pcm: Vec<f32> = (0..n).map(|i| {
            let t = i as f32 / sr as f32;
            let env = (1.0 - t / 0.5).max(0.0);
            env * (2.0 * std::f32::consts::PI * f0 * t).sin()
        }).collect();
        let fitted = fit("fixture-sine", &Pcm { samples: pcm, sr }).expect("fit ok");
        assert!(fitted.residual < 0.25, "residual {:.2}", fitted.residual);
    }

    #[test]
    fn wav_parser_rejects_garbage() {
        assert!(parse_wav(b"nope").is_err());
    }

    #[test]
    fn synth_reference_is_deterministic() {
        let m = FittedModel::Fm(FmParams {
            f0: 300, sweep: -120, ratio_q4: 32, index_q4: 24, decay: 20, vol: 12, noise: 0,
        });
        let a = synth_reference(&m, 1000);
        let b = synth_reference(&m, 1000);
        assert_eq!(a, b);
        assert!(a.iter().any(|v| v.abs() > 0.01));
    }
}
