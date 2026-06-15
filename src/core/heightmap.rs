//! Color mapping for the topographic height layer

/// Lowest block height mapped by the height ramp (Minecraft world bottom)
const MIN_HEIGHT: f32 = -64.0;
/// Highest block height mapped by the height ramp (Minecraft world top)
const MAX_HEIGHT: f32 = 320.0;

/// Color stops of the hypsometric tint, as `(t, [r, g, b])`
///
/// `t` is the normalized height in the range 0..1.
const STOPS: &[(f32, [f32; 3])] = &[
	(0.0, [38.0, 70.0, 160.0]),    // deep (dark blue)
	(0.28, [60.0, 130.0, 200.0]),  // low water (blue)
	(0.34, [180.0, 200.0, 120.0]), // shoreline (sandy)
	(0.5, [60.0, 150.0, 60.0]),    // lowland (green)
	(0.68, [150.0, 130.0, 70.0]),  // hills (brown)
	(0.85, [120.0, 90.0, 60.0]),   // mountains (dark brown)
	(1.0, [245.0, 245.0, 245.0]),  // peaks (snow)
];

/// Linearly interpolates between two RGB colors
fn lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [u8; 3] {
	let mut out = [0u8; 3];
	for i in 0..3 {
		out[i] = (a[i] + (b[i] - a[i]) * t).round().clamp(0.0, 255.0) as u8;
	}
	out
}

/// Maps a block height to a hypsometric tint color
pub fn height_color(height: i32) -> [u8; 3] {
	let t = ((height as f32 - MIN_HEIGHT) / (MAX_HEIGHT - MIN_HEIGHT)).clamp(0.0, 1.0);

	let mut prev = STOPS[0];
	for &stop in &STOPS[1..] {
		if t <= stop.0 {
			let span = stop.0 - prev.0;
			let local = if span > 0.0 { (t - prev.0) / span } else { 0.0 };
			return lerp(prev.1, stop.1, local);
		}
		prev = stop;
	}
	let last = STOPS[STOPS.len() - 1].1;
	[last[0] as u8, last[1] as u8, last[2] as u8]
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_height_color_range() {
		// Endpoints and clamping
		assert_eq!(height_color(-1000), height_color(-64));
		assert_eq!(height_color(1000), height_color(320));
		// Monotonic-ish: a high point is brighter than a low point
		let low = height_color(-32);
		let high = height_color(300);
		let sum = |c: [u8; 3]| c[0] as u32 + c[1] as u32 + c[2] as u32;
		assert!(sum(high) > sum(low));
	}
}
