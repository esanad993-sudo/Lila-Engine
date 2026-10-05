//! v15 shading test battery: gamma-correct blending (the opt-in knob) +
//! the 3D depth-cue curve. The tests live next to their implementations —
//! raster.zig: the comptime sRGB LUT pair (exact round-trip, monotonicity,
//! the 188-vs-128 anchor), putPx blend goldens in both color spaces, the
//! gradient midpoint (sRGB lerp vs linear-light lerp), flat-color
//! identity between modes, and the c.mem isolation proof (the rasterizer
//! is pixel-output-only — the state-hash chain cannot see it).
//! render3d.zig: depthShade curve properties (near passthrough,
//! monotonicity, the 96/256 floor, the demo-cube 79% calibration).
//!
//! Run: zig test src/test_raster.zig   (from runtime/)
//!   or scripts/build_engine.sh --test for the whole battery.

const std = @import("std");

test {
    _ = @import("raster.zig");
    _ = @import("render3d.zig");
}
