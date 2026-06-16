//! Emission of per-chunk overlay data
//!
//! When the `--emit-overlays` option is passed, MinedMap accumulates two kinds
//! of per-chunk information while it is already visiting every chunk for tile
//! generation, and writes them out as JSON files:
//!
//! * `inhabited_heatmap.json` — the `InhabitedTime` of each chunk
//! * `block_features.json` — presence of notable blocks (rails, farmland,
//!   portals) and a "built" score derived from player-placed block entities
//!
//! This allows downstream tooling to avoid a separate, slow pass over the save
//! data. The data is collected per dimension and aggregated per region to keep
//! memory usage bounded.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::{io::fs, world::de};

/// A Minecraft dimension
///
/// Used as the top-level key of the overlay output files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dimension {
	/// The overworld (default dimension)
	Overworld,
	/// The nether (dimension -1 in Java, 1 in Bedrock)
	Nether,
	/// The end (dimension 1 in Java, 2 in Bedrock)
	End,
}

impl Dimension {
	/// All dimensions in output order
	pub const ALL: [Dimension; 3] = [Dimension::Overworld, Dimension::Nether, Dimension::End];
}

/// Block entity IDs counted towards the "built" score
///
/// These are player-placed functional blocks; the number of such block
/// entities in a chunk is used as a rough measure of how built-up it is.
const BUILT_BLOCK_ENTITIES: &[&str] = &[
	"chest",
	"trapped_chest",
	"barrel",
	"furnace",
	"blast_furnace",
	"smoker",
	"crafting_table",
	"enchanting_table",
	"hopper",
	"dropper",
	"dispenser",
	"anvil",
	"beacon",
	"brewing_stand",
	"jukebox",
	"lectern",
	"loom",
	"cartography_table",
	"stonecutter",
	"campfire",
	"soul_campfire",
];

/// Strips a `minecraft:` namespace prefix from a block or block entity ID
fn strip_namespace(id: &str) -> &str {
	id.strip_prefix("minecraft:").unwrap_or(id)
}

/// Returns whether a block entity ID counts towards the "built" score
pub fn is_built_block_entity(id: &str) -> bool {
	let name = strip_namespace(id);
	BUILT_BLOCK_ENTITIES.contains(&name)
}

/// A generated structure's bounding box, for the structures viewer layer
#[derive(Debug, Clone, Serialize)]
pub struct Structure {
	/// Structure type ID (for example `minecraft:village`)
	#[serde(rename = "type")]
	pub structure_type: String,
	/// Bounding box in block coordinates `[minX, minZ, maxX, maxZ]`
	pub bb: [i32; 4],
}

/// Extends a bounding box accumulator with an NBT `[minX, minY, minZ, maxX, maxY, maxZ]`
fn union_bb(acc: &mut Option<[i32; 4]>, arr: &fastnbt::IntArray) {
	if arr.len() < 6 {
		return;
	}
	let (min_x, min_z, max_x, max_z) = (arr[0], arr[2], arr[3], arr[5]);
	match acc {
		None => *acc = Some([min_x, min_z, max_x, max_z]),
		Some(bb) => {
			bb[0] = bb[0].min(min_x);
			bb[1] = bb[1].min(min_z);
			bb[2] = bb[2].max(max_x);
			bb[3] = bb[3].max(max_z);
		}
	}
}

/// Computes a structure's full bounding box from its start and child pieces
fn structure_bb(start: &de::StructureStart) -> Option<[i32; 4]> {
	let mut acc = None;
	if let Some(bb) = &start.bb {
		union_bb(&mut acc, bb);
	}
	for child in &start.children {
		if let Some(bb) = &child.bb {
			union_bb(&mut acc, bb);
		}
	}
	acc
}

/// Extracts the generated structures starting in a Java [chunk](de::Chunk)
pub fn java_chunk_structures(chunk: &de::Chunk) -> Vec<Structure> {
	let structures = match &chunk.chunk {
		de::ChunkVariant::V1_18 { .. } => chunk.structures.as_ref(),
		de::ChunkVariant::V0 { level } => level.structures.as_ref(),
	};
	let Some(structures) = structures else {
		return Vec::new();
	};

	let mut result = Vec::new();
	for (name, start) in &structures.starts {
		if start.id.as_deref() == Some("INVALID") {
			continue;
		}
		if let Some(bb) = structure_bb(start) {
			result.push(Structure {
				structure_type: name.clone(),
				bb,
			});
		}
	}
	result
}

/// Per-chunk overlay information
///
/// Collected while a chunk is visited; merged into a [DimensionOverlay]
/// afterwards using the chunk's absolute coordinates.
#[derive(Debug, Clone, Default)]
pub struct ChunkOverlayInfo {
	/// Cumulative number of ticks players have spent in the chunk
	pub inhabited_time: i64,
	/// The chunk's block palette contains a rail
	pub rail: bool,
	/// The chunk's block palette contains farmland
	pub farmland: bool,
	/// The chunk's block palette contains a nether portal block
	pub nether_portal: bool,
	/// The chunk's block palette contains an end portal block
	pub end_portal: bool,
	/// The chunk is a slime chunk (Java Edition, derived from the world seed)
	pub slime: bool,
	/// Number of player-placed block entities in the chunk
	pub built: u32,
}

impl ChunkOverlayInfo {
	/// Updates the feature flags for a single block ID found in the palette
	pub fn note_block(&mut self, id: &str) {
		let name = strip_namespace(id);
		if name == "rail" || name.ends_with("_rail") {
			self.rail = true;
		} else if name == "farmland" {
			self.farmland = true;
		} else if name == "nether_portal" {
			self.nether_portal = true;
		} else if name == "end_portal" {
			self.end_portal = true;
		}
	}
}

/// Accumulated overlay data for a single dimension
///
/// Each entry stores absolute chunk coordinates (block coordinate `>> 4`).
#[derive(Debug, Default)]
pub struct DimensionOverlay {
	/// `[chunkX, chunkZ, inhabitedTimeTicks]` for chunks with ticks > 0
	pub inhabited: Vec<(i32, i32, i64)>,
	/// `[chunkX, chunkZ]` of chunks containing a rail
	pub rail: Vec<(i32, i32)>,
	/// `[chunkX, chunkZ]` of chunks containing farmland
	pub farmland: Vec<(i32, i32)>,
	/// `[chunkX, chunkZ]` of chunks containing a nether portal block
	pub nether_portal: Vec<(i32, i32)>,
	/// `[chunkX, chunkZ]` of chunks containing an end portal block
	pub end_portal: Vec<(i32, i32)>,
	/// `[chunkX, chunkZ]` of slime chunks
	pub slime: Vec<(i32, i32)>,
	/// `[chunkX, chunkZ, score]` for chunks with a "built" score > 0
	pub built: Vec<(i32, i32, u32)>,
	/// Generated structures (written to a separate `structures.json`)
	pub structures: Vec<Structure>,
}

impl DimensionOverlay {
	/// Adds a chunk's overlay info at the given absolute chunk coordinates
	pub fn add(&mut self, chunk_x: i32, chunk_z: i32, info: &ChunkOverlayInfo) {
		if info.inhabited_time > 0 {
			self.inhabited.push((chunk_x, chunk_z, info.inhabited_time));
		}
		if info.rail {
			self.rail.push((chunk_x, chunk_z));
		}
		if info.farmland {
			self.farmland.push((chunk_x, chunk_z));
		}
		if info.nether_portal {
			self.nether_portal.push((chunk_x, chunk_z));
		}
		if info.end_portal {
			self.end_portal.push((chunk_x, chunk_z));
		}
		if info.slime {
			self.slime.push((chunk_x, chunk_z));
		}
		if info.built > 0 {
			self.built.push((chunk_x, chunk_z, info.built));
		}
	}

	/// Merges another [DimensionOverlay] into this one
	pub fn merge(&mut self, mut other: DimensionOverlay) {
		self.inhabited.append(&mut other.inhabited);
		self.rail.append(&mut other.rail);
		self.farmland.append(&mut other.farmland);
		self.nether_portal.append(&mut other.nether_portal);
		self.end_portal.append(&mut other.end_portal);
		self.slime.append(&mut other.slime);
		self.built.append(&mut other.built);
		self.structures.append(&mut other.structures);
	}

	/// Sorts all entries by chunk coordinates for deterministic output
	fn sort(&mut self) {
		self.inhabited.sort_unstable_by_key(|&(x, z, _)| (x, z));
		self.rail.sort_unstable();
		self.farmland.sort_unstable();
		self.nether_portal.sort_unstable();
		self.end_portal.sort_unstable();
		self.slime.sort_unstable();
		self.built.sort_unstable_by_key(|&(x, z, _)| (x, z));
		self.structures
			.sort_by(|a, b| (a.bb, &a.structure_type).cmp(&(b.bb, &b.structure_type)));
	}
}

/// Accumulated overlay data for all dimensions
#[derive(Debug, Default)]
pub struct OverlayData {
	/// Overworld overlay data
	pub overworld: DimensionOverlay,
	/// Nether overlay data
	pub nether: DimensionOverlay,
	/// End overlay data
	pub end: DimensionOverlay,
}

impl OverlayData {
	/// Returns a mutable reference to the data for a dimension
	pub fn dimension_mut(&mut self, dim: Dimension) -> &mut DimensionOverlay {
		match dim {
			Dimension::Overworld => &mut self.overworld,
			Dimension::Nether => &mut self.nether,
			Dimension::End => &mut self.end,
		}
	}

	/// Sorts all dimensions for deterministic output
	fn sort(&mut self) {
		for dim in Dimension::ALL {
			self.dimension_mut(dim).sort();
		}
	}

	/// Writes the `inhabited_heatmap.json` and `block_features.json` files into
	/// each of the given directories
	pub fn write(mut self, dirs: &[&Path]) -> Result<()> {
		self.sort();

		let heatmap = HeatmapOutput {
			overworld: &self.overworld.inhabited,
			nether: &self.nether.inhabited,
			end: &self.end.inhabited,
		};
		let features = FeaturesOutput {
			overworld: (&self.overworld).into(),
			nether: (&self.nether).into(),
			end: (&self.end).into(),
		};

		for dir in dirs {
			fs::create_dir_all(dir)?;
			write_json(&dir.join("inhabited_heatmap.json"), &heatmap)
				.context("Failed to write inhabited_heatmap.json")?;
			write_json(&dir.join("block_features.json"), &features)
				.context("Failed to write block_features.json")?;
		}

		Ok(())
	}
}

/// Serializes a value as JSON to a file, replacing it atomically
fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
	fs::create_with_tmpfile(path, |file| {
		serde_json::to_writer(file, value).context("Failed to serialize overlay data")
	})
}

/// Serialization shape of `structures.json`
#[derive(Serialize)]
struct StructuresOutput<'a> {
	/// Generated structures
	structures: &'a [Structure],
}

/// Writes the `structures.json` file
pub fn write_structures(path: &Path, structures: &[Structure]) -> Result<()> {
	write_json(path, &StructuresOutput { structures }).context("Failed to write structures.json")
}

/// Serialization shape of `inhabited_heatmap.json`
#[derive(Serialize)]
struct HeatmapOutput<'a> {
	/// Overworld entries
	overworld: &'a [(i32, i32, i64)],
	/// Nether entries
	nether: &'a [(i32, i32, i64)],
	/// End entries
	end: &'a [(i32, i32, i64)],
}

/// Per-dimension serialization shape of `block_features.json`
#[derive(Serialize)]
struct FeaturesDimensionOutput<'a> {
	/// Chunks containing rails
	rail: &'a [(i32, i32)],
	/// Chunks containing farmland
	farmland: &'a [(i32, i32)],
	/// Chunks containing nether portal blocks
	nether_portal: &'a [(i32, i32)],
	/// Chunks containing end portal blocks
	end_portal: &'a [(i32, i32)],
	/// Slime chunks
	slime: &'a [(i32, i32)],
	/// Chunks with a "built" score
	built: &'a [(i32, i32, u32)],
}

impl<'a> From<&'a DimensionOverlay> for FeaturesDimensionOutput<'a> {
	fn from(value: &'a DimensionOverlay) -> Self {
		FeaturesDimensionOutput {
			rail: &value.rail,
			farmland: &value.farmland,
			nether_portal: &value.nether_portal,
			end_portal: &value.end_portal,
			slime: &value.slime,
			built: &value.built,
		}
	}
}

/// Serialization shape of `block_features.json`
#[derive(Serialize)]
struct FeaturesOutput<'a> {
	/// Overworld features
	overworld: FeaturesDimensionOutput<'a>,
	/// Nether features
	nether: FeaturesDimensionOutput<'a>,
	/// End features
	end: FeaturesDimensionOutput<'a>,
}

/// Extracts [ChunkOverlayInfo] from a deserialized Java [chunk](de::Chunk)
pub fn java_chunk_overlay_info(chunk: &de::Chunk) -> ChunkOverlayInfo {
	let mut info = ChunkOverlayInfo::default();

	match &chunk.chunk {
		de::ChunkVariant::V1_18 {
			sections,
			block_entities,
		} => {
			info.inhabited_time = chunk.inhabited_time.unwrap_or(0);
			for section in sections {
				if let de::SectionV1_18Variant::V1_18 { block_states, .. } = &section.section {
					for entry in &block_states.palette {
						info.note_block(&entry.name);
					}
				}
			}
			info.built = count_built(block_entities);
		}
		de::ChunkVariant::V0 { level } => {
			info.inhabited_time = chunk.inhabited_time.or(level.inhabited_time).unwrap_or(0);
			for section in &level.sections {
				if let de::SectionV0Variant::V1_13 { palette, .. } = &section.section {
					for entry in palette {
						info.note_block(&entry.name);
					}
				}
				// Pre-1.13 numeric sections do not carry a named block palette,
				// so feature detection is not available for them.
			}
			info.built = count_built(&level.tile_entities);
		}
	}

	info
}

/// Counts the player-placed block entities in a list
fn count_built(entities: &[de::BlockEntity]) -> u32 {
	entities
		.iter()
		.filter(|entity| is_built_block_entity(&entity.id))
		.count() as u32
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_java_chunk_structures() {
		use serde::Serialize;

		#[derive(Serialize)]
		struct Piece {
			#[serde(rename = "BB")]
			bb: fastnbt::IntArray,
		}
		#[derive(Serialize)]
		struct Start {
			#[serde(rename = "BB")]
			bb: fastnbt::IntArray,
			id: String,
			#[serde(rename = "Children")]
			children: Vec<Piece>,
		}
		#[derive(Serialize)]
		struct Invalid {
			id: String,
		}
		#[derive(Serialize)]
		struct Starts {
			#[serde(rename = "minecraft:village")]
			village: Start,
			#[serde(rename = "minecraft:mineshaft")]
			mineshaft: Invalid,
		}
		#[derive(Serialize)]
		struct StructuresNbt {
			starts: Starts,
		}
		#[derive(Serialize)]
		struct Chunk {
			#[serde(rename = "DataVersion")]
			data_version: i32,
			structures: StructuresNbt,
			sections: Vec<i32>,
			block_entities: Vec<i32>,
		}

		let chunk = Chunk {
			data_version: 3000,
			structures: StructuresNbt {
				starts: Starts {
					village: Start {
						bb: fastnbt::IntArray::new(vec![0, 60, 0, 16, 70, 16]),
						id: "minecraft:village".to_string(),
						// A child piece extends the bounding box to the east
						children: vec![Piece {
							bb: fastnbt::IntArray::new(vec![16, 60, 0, 40, 70, 24]),
						}],
					},
					mineshaft: Invalid {
						id: "INVALID".to_string(),
					},
				},
			},
			sections: Vec::new(),
			block_entities: Vec::new(),
		};

		let bytes = fastnbt::to_bytes(&chunk).unwrap();
		let decoded: de::Chunk = fastnbt::from_bytes(&bytes).unwrap();
		let structures = java_chunk_structures(&decoded);

		// The INVALID mineshaft is skipped; the village BB is the union of the
		// start and its child piece ([minX, minZ, maxX, maxZ]).
		assert_eq!(structures.len(), 1);
		assert_eq!(structures[0].structure_type, "minecraft:village");
		assert_eq!(structures[0].bb, [0, 0, 40, 24]);
	}

	#[test]
	fn test_note_block() {
		let mut info = ChunkOverlayInfo::default();
		info.note_block("minecraft:rail");
		info.note_block("minecraft:powered_rail");
		info.note_block("minecraft:farmland");
		info.note_block("minecraft:nether_portal");
		info.note_block("minecraft:end_portal");
		info.note_block("minecraft:stone");
		assert!(info.rail);
		assert!(info.farmland);
		assert!(info.nether_portal);
		assert!(info.end_portal);
	}

	#[test]
	fn test_built_ids() {
		assert!(is_built_block_entity("minecraft:chest"));
		assert!(is_built_block_entity("chest"));
		assert!(is_built_block_entity("minecraft:blast_furnace"));
		assert!(!is_built_block_entity("minecraft:stone"));
		assert!(!is_built_block_entity("minecraft:sign"));
	}

	#[test]
	fn test_heatmap_shape() {
		let data = HeatmapOutput {
			overworld: &[(1, 2, 100), (-3, 4, 5)],
			nether: &[],
			end: &[],
		};
		let json = serde_json::to_string(&data).unwrap();
		assert_eq!(
			json,
			r#"{"overworld":[[1,2,100],[-3,4,5]],"nether":[],"end":[]}"#
		);
	}

	#[test]
	fn test_java_chunk_overlay_info() {
		use serde::Serialize;

		#[derive(Serialize)]
		struct PaletteEntry {
			#[serde(rename = "Name")]
			name: String,
		}
		#[derive(Serialize)]
		struct BlockStates {
			palette: Vec<PaletteEntry>,
		}
		#[derive(Serialize)]
		struct Biomes {
			palette: Vec<String>,
		}
		#[derive(Serialize)]
		struct Section {
			#[serde(rename = "Y")]
			y: i32,
			block_states: BlockStates,
			biomes: Biomes,
		}
		#[derive(Serialize)]
		struct BlockEntity {
			id: String,
			x: i32,
			y: i32,
			z: i32,
		}
		#[derive(Serialize)]
		struct Chunk {
			#[serde(rename = "DataVersion")]
			data_version: i32,
			#[serde(rename = "InhabitedTime")]
			inhabited_time: i64,
			sections: Vec<Section>,
			block_entities: Vec<BlockEntity>,
		}

		let palette = |name: &str| PaletteEntry {
			name: name.to_string(),
		};
		let entity = |id: &str| BlockEntity {
			id: id.to_string(),
			x: 0,
			y: 0,
			z: 0,
		};

		let chunk = Chunk {
			data_version: 3000,
			inhabited_time: 4321,
			sections: vec![Section {
				y: 0,
				block_states: BlockStates {
					palette: vec![
						palette("minecraft:stone"),
						palette("minecraft:powered_rail"),
						palette("minecraft:farmland"),
					],
				},
				biomes: Biomes {
					palette: vec!["minecraft:plains".to_string()],
				},
			}],
			block_entities: vec![
				entity("minecraft:chest"),
				entity("minecraft:furnace"),
				entity("minecraft:bed"),
			],
		};

		let bytes = fastnbt::to_bytes(&chunk).unwrap();
		let decoded: de::Chunk = fastnbt::from_bytes(&bytes).unwrap();
		let info = java_chunk_overlay_info(&decoded);

		assert_eq!(info.inhabited_time, 4321);
		assert!(info.rail);
		assert!(info.farmland);
		assert!(!info.nether_portal);
		assert!(!info.end_portal);
		// chest + furnace are player-placed; bed is not in the built set
		assert_eq!(info.built, 2);
	}

	#[test]
	fn test_features_shape() {
		let dim = DimensionOverlay {
			rail: vec![(1, 2)],
			slime: vec![(5, 6)],
			built: vec![(3, 4, 7)],
			..Default::default()
		};
		let empty = DimensionOverlay::default();
		let data = FeaturesOutput {
			overworld: (&dim).into(),
			nether: (&empty).into(),
			end: (&empty).into(),
		};
		let json = serde_json::to_string(&data).unwrap();
		assert_eq!(
			json,
			concat!(
				r#"{"overworld":{"rail":[[1,2]],"farmland":[],"nether_portal":[],"#,
				r#""end_portal":[],"slime":[[5,6]],"built":[[3,4,7]]},"#,
				r#""nether":{"rail":[],"farmland":[],"nether_portal":[],"end_portal":[],"slime":[],"built":[]},"#,
				r#""end":{"rail":[],"farmland":[],"nether_portal":[],"end_portal":[],"slime":[],"built":[]}}"#,
			)
		);
	}
}
