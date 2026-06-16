//! Collection of world-level statistics for the `--emit-world-stats` output
//!
//! Produces a single `world.json` summarizing a world: its seed, per-dimension
//! region and inhabited-chunk counts, the total number of blocks the players
//! have mined and placed, and the on-disk size of the world.

use std::{
	collections::BTreeMap,
	ffi::OsStr,
	path::{Path, PathBuf},
};

use serde::Serialize;

use super::{common::*, overlay, player};

/// World-level statistics written to `world.json`
#[derive(Debug, Default, Serialize)]
pub struct WorldStats {
	/// World seed, if it could be determined
	#[serde(skip_serializing_if = "Option::is_none")]
	pub seed: Option<i64>,
	/// Number of region files per dimension
	pub regions: BTreeMap<&'static str, usize>,
	/// Number of chunks with a non-zero `InhabitedTime` per dimension
	pub inhabited_chunks: BTreeMap<&'static str, usize>,
	/// Total number of blocks mined by all players
	pub blocks_mined: i64,
	/// Total number of blocks placed by all players (approximated by item use)
	pub blocks_placed: i64,
	/// Total on-disk size of the input world directory, in bytes
	pub size_bytes: u64,
}

/// Returns whether a file name matches the `r.X.Z.mca` region pattern
fn is_region_filename(file_name: &OsStr) -> bool {
	file_name
		.to_str()
		.map(|name| {
			let parts: Vec<_> = name.split('.').collect();
			matches!(parts.as_slice(), ["r", x, z, "mca"]
				if x.parse::<i32>().is_ok() && z.parse::<i32>().is_ok())
		})
		.unwrap_or(false)
}

/// Counts the Anvil region files in a directory
pub fn count_region_files(dir: &Path) -> usize {
	let Ok(entries) = dir.read_dir() else {
		return 0;
	};
	entries
		.filter_map(Result::ok)
		.filter(|entry| entry.file_type().map(|t| t.is_file()).unwrap_or(false))
		.filter(|entry| is_region_filename(&entry.file_name()))
		.count()
}

/// Recursively sums the size of all regular files below a directory
///
/// Symlinks are not followed, so directory cycles cannot cause infinite
/// recursion.
pub fn dir_size(path: &Path) -> u64 {
	let Ok(entries) = path.read_dir() else {
		return 0;
	};
	let mut total = 0;
	for entry in entries.filter_map(Result::ok) {
		let Ok(file_type) = entry.file_type() else {
			continue;
		};
		if file_type.is_dir() {
			total += dir_size(&entry.path());
		} else if file_type.is_file()
			&& let Ok(meta) = entry.metadata()
		{
			total += meta.len();
		}
	}
	total
}

/// Sums the mined and placed block counts over all players
fn player_block_totals(config: &Config) -> (i64, i64) {
	let players = player::collect(
		&config.input_dir,
		config.player_data_dir.as_deref(),
		config.player_stats_dir.as_deref(),
		&config.usercache_files,
	);
	players
		.iter()
		.filter_map(|p| p.stats.as_ref())
		.fold((0, 0), |(mined, placed), s| {
			(mined + s.blocks_mined, placed + s.blocks_placed)
		})
}

/// Collects the world statistics of a Java Edition world
///
/// `overlays` provides the inhabited-chunk data already gathered during the
/// render pass.
pub fn collect_java(config: &Config, overlays: &overlay::OverlayData) -> WorldStats {
	let mut regions = BTreeMap::new();
	for (dimension, region_dir) in config.dimensions_to_render() {
		regions.insert(dimension.key(), count_region_files(&region_dir));
	}

	let mut inhabited_chunks = BTreeMap::new();
	for dimension in overlay::Dimension::ALL {
		inhabited_chunks.insert(
			dimension.key(),
			overlays.dimension(dimension).inhabited.len(),
		);
	}

	let (blocks_mined, blocks_placed) = player_block_totals(config);

	WorldStats {
		seed: config.world_seed,
		regions,
		inhabited_chunks,
		blocks_mined,
		blocks_placed,
		size_bytes: dir_size(&config.input_dir),
	}
}

/// Returns the standard path of the `world.json` output file
pub fn output_path(dir: &Path) -> PathBuf {
	dir.join("world.json")
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_region_filename() {
		assert!(is_region_filename(OsStr::new("r.0.0.mca")));
		assert!(is_region_filename(OsStr::new("r.-1.2.mca")));
		assert!(!is_region_filename(OsStr::new("r.0.0.bin")));
		assert!(!is_region_filename(OsStr::new("r.a.0.mca")));
		assert!(!is_region_filename(OsStr::new("level.dat")));
	}

	#[test]
	fn test_count_regions_and_size() {
		let dir = std::env::temp_dir().join(format!(
			"minedmap-world-test-{}-{:?}",
			std::process::id(),
			std::thread::current().id()
		));
		let region = dir.join("region");
		std::fs::create_dir_all(&region).unwrap();
		std::fs::write(region.join("r.0.0.mca"), [0u8; 100]).unwrap();
		std::fs::write(region.join("r.0.1.mca"), [0u8; 50]).unwrap();
		std::fs::write(region.join("not-a-region.txt"), [0u8; 10]).unwrap();
		// A nested subdirectory must be counted towards the size as well
		std::fs::create_dir_all(dir.join("sub")).unwrap();
		std::fs::write(dir.join("sub/data.bin"), [0u8; 25]).unwrap();

		assert_eq!(count_region_files(&region), 2);
		assert_eq!(dir_size(&dir), 100 + 50 + 10 + 25);

		let _ = std::fs::remove_dir_all(&dir);
	}
}
