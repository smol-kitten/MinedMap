//! A reimplementation of `java.util.Random` and the Minecraft slime-chunk check
//!
//! Slime chunks in Java Edition are determined by seeding a `java.util.Random`
//! with a value derived from the world seed and the chunk coordinates, so
//! reproducing them requires a bit-for-bit compatible Random implementation.

/// Multiplier of the `java.util.Random` linear congruential generator
const MULTIPLIER: i64 = 0x5DEECE66D;
/// Increment of the `java.util.Random` linear congruential generator
const INCREMENT: i64 = 0xB;
/// Bit mask for the 48-bit `java.util.Random` state
const MASK: i64 = (1 << 48) - 1;

/// A bit-compatible reimplementation of `java.util.Random`
pub struct JavaRandom {
	/// Current 48-bit generator state
	seed: i64,
}

impl JavaRandom {
	/// Creates a new generator from a seed (matching `new Random(seed)`)
	pub fn new(seed: i64) -> Self {
		JavaRandom {
			seed: (seed ^ MULTIPLIER) & MASK,
		}
	}

	/// Returns the next `bits` pseudorandom bits (matching `Random.next(bits)`)
	fn next(&mut self, bits: u32) -> i32 {
		self.seed = self.seed.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT) & MASK;
		(self.seed >> (48 - bits)) as i32
	}

	/// Returns a pseudorandom value in `0..bound` (matching `Random.nextInt(int)`)
	pub fn next_int(&mut self, bound: i32) -> i32 {
		debug_assert!(bound > 0);

		// Power-of-two bounds use a faster path in the JDK
		if (bound & bound.wrapping_sub(1)) == 0 {
			return ((i64::from(bound) * i64::from(self.next(31))) >> 31) as i32;
		}

		loop {
			let bits = self.next(31);
			let val = bits % bound;
			if bits.wrapping_sub(val).wrapping_add(bound - 1) >= 0 {
				return val;
			}
		}
	}
}

/// Determines whether a chunk is a slime chunk in Java Edition
///
/// Replicates Minecraft's slime-chunk seeding exactly, including the 32-bit
/// integer overflow of the coordinate terms before they are widened to 64 bits.
pub fn is_slime_chunk(world_seed: i64, chunk_x: i32, chunk_z: i32) -> bool {
	let t1 = i64::from(chunk_x.wrapping_mul(chunk_x).wrapping_mul(0x4c1906));
	let t2 = i64::from(chunk_x.wrapping_mul(0x5ac0db));
	let t3 = i64::from(chunk_z.wrapping_mul(chunk_z)).wrapping_mul(0x4307a7);
	let t4 = i64::from(chunk_z.wrapping_mul(0x5f24f));

	let seed = world_seed
		.wrapping_add(t1)
		.wrapping_add(t2)
		.wrapping_add(t3)
		.wrapping_add(t4)
		^ 0x3ad8025f;

	JavaRandom::new(seed).next_int(10) == 0
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_java_random_known_vectors() {
		// The first two outputs of new Random(0).nextInt() are well-known JDK
		// values; they validate the core generator.
		let mut rng = JavaRandom::new(0);
		assert_eq!(rng.next(32), -1155484576);
		assert_eq!(rng.next(32), -723955400);
	}

	#[test]
	fn test_slime_chunk_density() {
		// Slime chunks occur with probability 1/10, so over a large grid the
		// fraction should be close to 10%.
		let seed = 1234567890;
		let mut count = 0;
		let total = 200 * 200;
		for x in -100..100 {
			for z in -100..100 {
				if is_slime_chunk(seed, x, z) {
					count += 1;
				}
			}
		}
		let fraction = count as f64 / total as f64;
		assert!(
			(0.08..0.12).contains(&fraction),
			"slime fraction out of range: {fraction}"
		);
	}
}
