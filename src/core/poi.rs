//! Collection of points of interest (POIs) for the viewer marker layers
//!
//! Java Edition stores POIs — village meeting points, villager beds and job
//! sites, nether portals, lodestones, … — in `poi/*.mca` files using the same
//! Anvil region format as block data. This module reads those files for one
//! dimension and returns the categorized POI positions as [PoiData]. The caller
//! ([crate::core]) merges the per-dimension results into the dimension-keyed
//! `pois.json` consumed by the viewer. The output schema is documented in the
//! README ("Output data files").

use std::{ffi::OsStr, path::Path};

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::warn;

use super::common::*;

/// A single POI record in a POI chunk section
#[derive(Debug, Deserialize)]
struct PoiRecord {
	/// Block position of the POI (`[x, y, z]`)
	pos: fastnbt::IntArray,
	/// POI type, for example `minecraft:meeting`
	#[serde(rename = "type")]
	poi_type: String,
}

/// A section of a POI chunk
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct PoiSection {
	/// POI records in the section
	#[serde(default)]
	records: Vec<PoiRecord>,
}

/// A POI chunk (one entry of a POI region file)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct PoiChunk {
	/// Sections keyed by section Y coordinate
	#[serde(default)]
	sections: std::collections::HashMap<String, PoiSection>,
}

/// Categorizes a POI type into a viewer marker category
fn category(poi_type: &str) -> &'static str {
	let name = poi_type.strip_prefix("minecraft:").unwrap_or(poi_type);
	match name {
		"meeting" => "village",
		"home" => "home",
		"nether_portal" => "portal",
		"lodestone" => "lodestone",
		"armorer" | "butcher" | "cartographer" | "cleric" | "farmer" | "fisherman" | "fletcher"
		| "leatherworker" | "librarian" | "mason" | "shepherd" | "toolsmith" | "weaponsmith" => "jobsite",
		_ => "other",
	}
}

/// Collected POI marker positions by category (block coordinates)
#[derive(Debug, Default, Serialize)]
pub struct PoiData {
	/// Village meeting points
	village: Vec<(i32, i32)>,
	/// Villager beds
	home: Vec<(i32, i32)>,
	/// Villager job sites
	jobsite: Vec<(i32, i32)>,
	/// Nether portals
	portal: Vec<(i32, i32)>,
	/// Lodestones
	lodestone: Vec<(i32, i32)>,
	/// Other POIs
	other: Vec<(i32, i32)>,
}

impl PoiData {
	/// Adds a POI of the given category at a block position
	fn push(&mut self, category: &str, x: i32, z: i32) {
		let list = match category {
			"village" => &mut self.village,
			"home" => &mut self.home,
			"jobsite" => &mut self.jobsite,
			"portal" => &mut self.portal,
			"lodestone" => &mut self.lodestone,
			_ => &mut self.other,
		};
		list.push((x, z));
	}

	/// Merges another [PoiData] into this one
	fn merge(&mut self, mut other: PoiData) {
		self.village.append(&mut other.village);
		self.home.append(&mut other.home);
		self.jobsite.append(&mut other.jobsite);
		self.portal.append(&mut other.portal);
		self.lodestone.append(&mut other.lodestone);
		self.other.append(&mut other.other);
	}

	/// Sorts and deduplicates all categories for deterministic output
	fn finish(&mut self) {
		for list in [
			&mut self.village,
			&mut self.home,
			&mut self.jobsite,
			&mut self.portal,
			&mut self.lodestone,
			&mut self.other,
		] {
			list.sort_unstable();
			list.dedup();
		}
	}
}

/// Parses a `r.X.Z.mca` POI file name (the X/Z values are not needed)
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

/// Reads the POIs of a single POI region file
fn collect_file(path: &Path) -> Result<PoiData> {
	let mut data = PoiData::default();
	crate::nbt::region::from_file(path)?.foreach_chunk(|_coords, chunk: PoiChunk| {
		for section in chunk.sections.values() {
			for record in &section.records {
				if record.pos.len() < 3 {
					continue;
				}
				let category = category(&record.poi_type);
				data.push(category, record.pos[0], record.pos[2]);
			}
		}
		Ok(())
	})?;
	Ok(data)
}

/// Collects the points of interest from the POI files of one dimension
pub fn collect(config: &Config) -> PoiData {
	let mut files = Vec::new();
	if let Ok(dir) = config.poi_dir.read_dir() {
		for entry in dir.filter_map(Result::ok) {
			if entry.file_type().map(|t| t.is_file()).unwrap_or(false)
				&& is_region_filename(&entry.file_name())
			{
				files.push(entry.path());
			}
		}
	}

	let mut data = files
		.par_iter()
		.map(|path| {
			collect_file(path).unwrap_or_else(|err| {
				warn!("Failed to read POI file {}: {:?}", path.display(), err);
				PoiData::default()
			})
		})
		.reduce(PoiData::default, |mut a, b| {
			a.merge(b);
			a
		});
	data.finish();
	data
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_category() {
		assert_eq!(category("minecraft:meeting"), "village");
		assert_eq!(category("minecraft:home"), "home");
		assert_eq!(category("minecraft:nether_portal"), "portal");
		assert_eq!(category("minecraft:armorer"), "jobsite");
		assert_eq!(category("minecraft:lodestone"), "lodestone");
		assert_eq!(category("minecraft:something_else"), "other");
	}

	#[test]
	fn test_poi_chunk_parsing() {
		// A POI chunk with a meeting point and a bed
		let value = fastnbt::nbt!({
			"Sections": {
				"4": {
					"Valid": 1i8,
					"Records": [
						{ "pos": [I; 10, 64, -20], "type": "minecraft:meeting", "free_tickets": 1 },
						{ "pos": [I; 5, 63, 5], "type": "minecraft:home", "free_tickets": 1 },
					],
				},
			},
		});
		let bytes = fastnbt::to_bytes(&value).unwrap();
		let chunk: PoiChunk = fastnbt::from_bytes(&bytes).unwrap();

		let mut data = PoiData::default();
		for section in chunk.sections.values() {
			for record in &section.records {
				data.push(category(&record.poi_type), record.pos[0], record.pos[2]);
			}
		}
		data.finish();

		assert_eq!(data.village, vec![(10, -20)]);
		assert_eq!(data.home, vec![(5, 5)]);
		assert!(data.jobsite.is_empty());
	}
}
