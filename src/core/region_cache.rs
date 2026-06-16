//! Per-region caching of collected emit data for incremental `--since` runs
//!
//! The marker/overlay outputs (`pois.json`, `mobs.json`, `structures.json`,
//! `inhabited_heatmap.json`, …) are single files aggregated across every region
//! of a dimension, so producing them normally requires reading every source
//! region file on each run. To make re-runs cheap, each region's contribution is
//! cached on disk (next to the other processed intermediate data) with the
//! timestamp of the source file it was derived from. On a later run a region is
//! only re-read from source when it actually changed; otherwise its cached
//! contribution is reused. The per-region contributions are then merged into the
//! aggregated output.
//!
//! `--since <ts>` is an additional optimization: regions whose source file was
//! not modified after `<ts>` are taken from the cache without even checking the
//! cache timestamp, so a caller that knows when the world last changed (for
//! example the time of the last `rsync`) can skip the bulk of the work. To avoid
//! ever dropping data, a region with no cached contribution is always recomputed
//! regardless of `--since`.

use std::{
	ffi::OsStr,
	path::{Path, PathBuf},
	time::SystemTime,
};

use rayon::prelude::*;
use serde::{Serialize, de::DeserializeOwned};
use tracing::warn;

use crate::io::{fs, storage};

/// A per-region contribution that can be merged into an aggregate
pub trait Mergeable: Default {
	/// Merges another contribution into this one
	fn merge(&mut self, other: Self);
}

/// Parses an `r.X.Z.mca` region filename into its coordinates
fn parse_region_filename(file_name: &OsStr) -> Option<(i32, i32)> {
	let name = file_name.to_str()?;
	let parts: Vec<_> = name.split('.').collect();
	let ["r", x, z, "mca"] = parts.as_slice() else {
		return None;
	};
	Some((x.parse().ok()?, z.parse().ok()?))
}

/// Lists the region files of a source directory as `(coords, path)` pairs
fn list_regions(source_dir: &Path) -> Vec<((i32, i32), PathBuf)> {
	let Ok(dir) = source_dir.read_dir() else {
		return Vec::new();
	};
	dir.filter_map(Result::ok)
		.filter(|entry| entry.file_type().map(|t| t.is_file()).unwrap_or(false))
		.filter_map(|entry| {
			parse_region_filename(&entry.file_name()).map(|coords| (coords, entry.path()))
		})
		.collect()
}

/// Determines whether a region must be recomputed from source
///
/// A region is recomputed when its source changed relative to the cache, or —
/// with `--since` — when it was modified after the given timestamp. A region
/// without a cached contribution is always recomputed so that data is never
/// dropped.
pub fn needs_recompute(
	source_mtime: Option<SystemTime>,
	cache_timestamp: Option<SystemTime>,
	since: Option<SystemTime>,
) -> bool {
	match since {
		Some(since) => cache_timestamp.is_none() || source_mtime.map(|m| m > since).unwrap_or(true),
		// `None > Some` is false and `Some > None` is true, so a missing cache
		// (None) is recomputed and an unreadable source (None) is not.
		None => source_mtime > cache_timestamp,
	}
}

/// Computes or loads the cached contribution of a single region
fn region_contribution<T, F>(
	coords: (i32, i32),
	source: &Path,
	cache_dir: &Path,
	version: fs::FileMetaVersion,
	since: Option<SystemTime>,
	compute: &F,
) -> T
where
	T: Mergeable + Serialize + DeserializeOwned,
	F: Fn(&Path) -> anyhow::Result<T>,
{
	let cache_path = cache_dir.join(format!("r.{}.{}.bin", coords.0, coords.1));
	let source_mtime = fs::modified_timestamp(source).ok();
	let cache_timestamp = fs::read_timestamp(&cache_path, version);

	if !needs_recompute(source_mtime, cache_timestamp, since) {
		if cache_timestamp.is_some()
			&& let Ok(data) = storage::read_file(&cache_path, storage::Format::Postcard)
		{
			return data;
		}
		return T::default();
	}

	match compute(source) {
		Ok(data) => {
			let timestamp = source_mtime.unwrap_or_else(SystemTime::now);
			if let Err(err) = storage::write_file(
				&cache_path,
				&data,
				storage::Format::Postcard,
				version,
				timestamp,
			) {
				warn!("Failed to write cache {}: {:?}", cache_path.display(), err);
			}
			data
		}
		Err(err) => {
			warn!("Failed to read region {}: {:?}", source.display(), err);
			// Fall back to any cached contribution rather than losing the region
			if cache_timestamp.is_some() {
				storage::read_file(&cache_path, storage::Format::Postcard).unwrap_or_default()
			} else {
				T::default()
			}
		}
	}
}

/// Collects per-region data with caching for incremental runs
///
/// For each region file in `source_dir`, the contribution is recomputed with
/// `compute` and cached in `cache_dir`, or reused from the cache when the source
/// is unchanged (see [needs_recompute]). All contributions are merged.
pub fn collect_cached<T, F>(
	source_dir: &Path,
	cache_dir: &Path,
	version: fs::FileMetaVersion,
	since: Option<SystemTime>,
	compute: F,
) -> T
where
	T: Mergeable + Serialize + DeserializeOwned + Send,
	F: Fn(&Path) -> anyhow::Result<T> + Sync,
{
	let regions = list_regions(source_dir);
	if regions.is_empty() {
		return T::default();
	}
	let _ = fs::create_dir_all(cache_dir);

	regions
		.par_iter()
		.map(|(coords, source)| {
			region_contribution(*coords, source, cache_dir, version, since, &compute)
		})
		.reduce(T::default, |mut a, b| {
			a.merge(b);
			a
		})
}

#[cfg(test)]
mod test {
	use super::*;
	use std::time::Duration;

	/// A simple mergeable accumulator for tests
	#[derive(Default, Serialize, serde::Deserialize, PartialEq, Debug)]
	struct Acc(Vec<i32>);

	impl Mergeable for Acc {
		fn merge(&mut self, mut other: Self) {
			self.0.append(&mut other.0);
		}
	}

	#[test]
	fn test_needs_recompute() {
		let t0 = SystemTime::UNIX_EPOCH;
		let t1 = t0 + Duration::from_secs(100);
		let t2 = t0 + Duration::from_secs(200);

		// No --since: recompute when source is newer than cache or cache missing
		assert!(needs_recompute(Some(t2), Some(t1), None));
		assert!(!needs_recompute(Some(t1), Some(t1), None));
		assert!(!needs_recompute(Some(t1), Some(t2), None));
		assert!(needs_recompute(Some(t1), None, None));

		// --since: recompute when source newer than `since`, OR cache missing
		assert!(needs_recompute(Some(t2), Some(t0), Some(t1)));
		assert!(!needs_recompute(Some(t0), Some(t0), Some(t1)));
		// cache missing always recomputes, even for old source
		assert!(needs_recompute(Some(t0), None, Some(t1)));
	}

	#[test]
	fn test_collect_cached() {
		use std::sync::atomic::{AtomicUsize, Ordering};

		let dir = std::env::temp_dir().join(format!(
			"minedmap-cache-test-{}-{:?}",
			std::process::id(),
			std::thread::current().id()
		));
		let source = dir.join("source");
		let cache = dir.join("cache");
		std::fs::create_dir_all(&source).unwrap();
		std::fs::write(source.join("r.0.0.mca"), b"a").unwrap();
		std::fs::write(source.join("r.0.1.mca"), b"b").unwrap();
		std::fs::write(source.join("ignored.txt"), b"x").unwrap();

		let version = fs::FileMetaVersion(1);
		let calls = AtomicUsize::new(0);
		let compute = |path: &Path| {
			calls.fetch_add(1, Ordering::SeqCst);
			// Derive a deterministic value from the region coordinates
			let name = path.file_name().unwrap().to_str().unwrap();
			let n = if name.contains("0.0") { 1 } else { 2 };
			Ok(Acc(vec![n]))
		};

		// First run computes every region and populates the cache
		let mut first: Acc = collect_cached(&source, &cache, version, None, compute);
		first.0.sort_unstable();
		assert_eq!(first, Acc(vec![1, 2]));
		assert_eq!(calls.load(Ordering::SeqCst), 2);

		// Second run reuses the cache: no recomputation, identical result
		calls.store(0, Ordering::SeqCst);
		let mut second: Acc = collect_cached(&source, &cache, version, None, compute);
		second.0.sort_unstable();
		assert_eq!(second, Acc(vec![1, 2]));
		assert_eq!(calls.load(Ordering::SeqCst), 0);

		let _ = std::fs::remove_dir_all(&dir);
	}
}
