//! Collection of mob (entity) markers for the viewer
//!
//! Java Edition 1.17+ stores entities in `entities/*.mca` files using the same
//! Anvil region format as block data. This module reads those files for one
//! dimension and returns the hostile/passive mob positions as [MobData]. The
//! caller ([crate::core]) merges the per-dimension results into the
//! dimension-keyed `mobs.json` consumed by the viewer. The output schema is
//! documented in the README ("Output data files").

use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::common::*;

/// A single entity record in an entity chunk
#[derive(Debug, Deserialize)]
struct EntityRecord {
	/// Entity type ID (for example `minecraft:zombie`)
	id: String,
	/// Entity position (`[x, y, z]` doubles)
	#[serde(rename = "Pos")]
	pos: Vec<f64>,
}

/// An entity chunk (one entry of an entity region file)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct EntityChunk {
	/// Entities stored in the chunk
	#[serde(default)]
	entities: Vec<EntityRecord>,
}

/// Hostile mob types
const HOSTILE: &[&str] = &[
	"zombie",
	"husk",
	"drowned",
	"zombie_villager",
	"skeleton",
	"stray",
	"bogged",
	"wither_skeleton",
	"creeper",
	"spider",
	"cave_spider",
	"enderman",
	"endermite",
	"silverfish",
	"witch",
	"slime",
	"magma_cube",
	"blaze",
	"ghast",
	"zombified_piglin",
	"piglin",
	"piglin_brute",
	"hoglin",
	"zoglin",
	"phantom",
	"guardian",
	"elder_guardian",
	"shulker",
	"vex",
	"evoker",
	"vindicator",
	"pillager",
	"ravager",
	"illusioner",
	"warden",
	"breeze",
	"creaking",
];

/// Passive and neutral mob types
const PASSIVE: &[&str] = &[
	"cow",
	"mooshroom",
	"pig",
	"sheep",
	"chicken",
	"rabbit",
	"horse",
	"donkey",
	"mule",
	"skeleton_horse",
	"zombie_horse",
	"llama",
	"trader_llama",
	"wolf",
	"cat",
	"ocelot",
	"parrot",
	"fox",
	"panda",
	"bee",
	"turtle",
	"dolphin",
	"cod",
	"salmon",
	"pufferfish",
	"tropical_fish",
	"squid",
	"glow_squid",
	"bat",
	"villager",
	"wandering_trader",
	"iron_golem",
	"snow_golem",
	"axolotl",
	"goat",
	"frog",
	"tadpole",
	"allay",
	"camel",
	"sniffer",
	"armadillo",
	"strider",
	"polar_bear",
];

/// Categorizes an entity ID into a viewer marker category, or `None` to skip it
fn category(id: &str) -> Option<&'static str> {
	let name = id.strip_prefix("minecraft:").unwrap_or(id);
	if HOSTILE.contains(&name) {
		Some("hostile")
	} else if PASSIVE.contains(&name) {
		Some("passive")
	} else {
		None
	}
}

/// Collected mob marker positions by category (block coordinates)
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MobData {
	/// Hostile mobs
	hostile: Vec<(i32, i32)>,
	/// Passive and neutral mobs
	passive: Vec<(i32, i32)>,
}

impl MobData {
	/// Adds a mob of the given category at a block position
	fn push(&mut self, category: &str, x: i32, z: i32) {
		match category {
			"hostile" => self.hostile.push((x, z)),
			_ => self.passive.push((x, z)),
		}
	}

	/// Sorts all categories for deterministic output
	fn finish(&mut self) {
		self.hostile.sort_unstable();
		self.passive.sort_unstable();
	}
}

impl super::region_cache::Mergeable for MobData {
	fn merge(&mut self, mut other: MobData) {
		self.hostile.append(&mut other.hostile);
		self.passive.append(&mut other.passive);
	}
}

/// Reads the mobs of a single entity region file
fn collect_file(path: &Path) -> Result<MobData> {
	let mut data = MobData::default();
	crate::nbt::region::from_file(path)?.foreach_chunk(|_coords, chunk: EntityChunk| {
		for entity in &chunk.entities {
			if entity.pos.len() < 3 {
				continue;
			}
			if let Some(category) = category(&entity.id) {
				data.push(
					category,
					entity.pos[0].floor() as i32,
					entity.pos[2].floor() as i32,
				);
			}
		}
		Ok(())
	})?;
	Ok(data)
}

/// Collects mob markers from the entity region files of one dimension
///
/// Per-region contributions are cached so unchanged regions are not re-read on
/// later (incremental) runs; see [super::region_cache].
pub fn collect(config: &Config) -> MobData {
	let mut data: MobData = super::region_cache::collect_cached(
		&config.entity_region_dir,
		&config.mob_cache_dir(),
		EMIT_CACHE_META_VERSION,
		config.since,
		collect_file,
	);
	data.finish();
	data
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_category() {
		assert_eq!(category("minecraft:zombie"), Some("hostile"));
		assert_eq!(category("minecraft:creeper"), Some("hostile"));
		assert_eq!(category("minecraft:cow"), Some("passive"));
		assert_eq!(category("villager"), Some("passive"));
		assert_eq!(category("minecraft:item"), None);
		assert_eq!(category("minecraft:arrow"), None);
	}

	#[test]
	fn test_entity_chunk_parsing() {
		let value = fastnbt::nbt!({
			"Entities": [
				{ "id": "minecraft:zombie", "Pos": [10.5, 64.0, -20.5] },
				{ "id": "minecraft:cow", "Pos": [5.0, 63.0, 5.0] },
				{ "id": "minecraft:item", "Pos": [0.0, 0.0, 0.0] },
			],
		});
		let bytes = fastnbt::to_bytes(&value).unwrap();
		let chunk: EntityChunk = fastnbt::from_bytes(&bytes).unwrap();

		let mut data = MobData::default();
		for entity in &chunk.entities {
			if let Some(category) = category(&entity.id) {
				data.push(
					category,
					entity.pos[0].floor() as i32,
					entity.pos[2].floor() as i32,
				);
			}
		}
		data.finish();

		assert_eq!(data.hostile, vec![(10, -21)]);
		assert_eq!(data.passive, vec![(5, 5)]);
	}
}
