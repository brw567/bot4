//! # Utility functions for mathematical operations

/// Clip value between min and max
/// TODO: Add docs
pub fn clip(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// Linear interpolation
/// TODO: Add docs
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Check if value is approximately equal
/// TODO: Add docs
pub fn approx_equal(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}