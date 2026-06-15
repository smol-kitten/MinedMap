//! Bedrock Edition (LevelDB) support
//!
//! This module reads Bedrock Edition worlds and feeds them into the same
//! rendering pipeline used for Java Edition. Bedrock stores chunk data in a
//! LevelDB database ([db]) rather than Anvil region files; the chunk columns are
//! split into 16-block-high subchunks using a "paletted storage" format
//! ([subchunk]) with a little-endian NBT block palette ([nbt]). Block names are
//! translated to Java Edition identifiers ([blocks]) so the existing color
//! tables can be reused.
//!
//! For each 32×32 chunk region of the overworld a [ProcessedRegion] is produced
//! and written in the same format as the Java path, after which [TileRenderer],
//! [TileMipmapper] and [MetadataWriter] generate the map tiles and metadata.
//! Overlay data is collected for all dimensions.

mod blocks;
mod db;
mod nbt;
mod subchunk;

use std::{
	collections::BTreeMap,
	num::NonZeroU16,
	path::{Path, PathBuf},
	time::SystemTime,
};

use anyhow::{Context, Result};
use rayon::prelude::*;
use tokio::runtime::Runtime;
use tracing::{debug, info, warn};

use super::{
	common::*,
	metadata_writer::MetadataWriter,
	overlay::{self, ChunkOverlayInfo, Dimension, OverlayData},
	tile_mipmapper::TileMipmapper,
	tile_renderer::TileRenderer,
};
use crate::{
	io::{fs, storage},
	resource::{BlockFlag, BlockTypes},
	types::*,
	world::layer::{self, BlockHeight},
};

use db::BedrockDb;
use subchunk::{SubChunkLayer, block_offset};

/// LevelDB key tag for subchunk (paletted storage) data
const TAG_SUBCHUNK: u8 = 0x2f;
/// LevelDB key tag for block entity data
const TAG_BLOCK_ENTITIES: u8 = 0x31;

/// Returns whether a byte is a recognized chunk-data key tag
///
/// Used to avoid misinterpreting non-chunk string keys as chunk keys.
fn is_chunk_tag(tag: u8) -> bool {
	matches!(tag, 0x2b..=0x3b | 0x76..=0x78)
}

/// The relevant part of a parsed chunk key
enum KeyTag {
	/// Subchunk data with a Y-index
	SubChunk(i8),
	/// Block entity data
	BlockEntities,
	/// Some other (ignored) chunk key
	Other,
}

/// A parsed Bedrock chunk LevelDB key
struct ChunkKey {
	/// Dimension of the chunk
	dim: Dimension,
	/// Chunk X coordinate
	cx: i32,
	/// Chunk Z coordinate
	cz: i32,
	/// The kind of data the key refers to
	tag: KeyTag,
}

/// Reads a little-endian `i32` from a slice
fn read_i32(data: &[u8]) -> i32 {
	i32::from_le_bytes(data.try_into().unwrap())
}

/// Maps a Bedrock dimension index to a [Dimension]
fn dimension_from_index(index: i32) -> Option<Dimension> {
	match index {
		1 => Some(Dimension::Nether),
		2 => Some(Dimension::End),
		_ => None,
	}
}

/// Maps a [Dimension] to its Bedrock dimension index (overworld has none)
fn dimension_index(dim: Dimension) -> Option<i32> {
	match dim {
		Dimension::Overworld => None,
		Dimension::Nether => Some(1),
		Dimension::End => Some(2),
	}
}

/// Parses a Bedrock chunk LevelDB key
///
/// Returns `None` for keys that are not chunk-data keys.
fn parse_key(key: &[u8]) -> Option<ChunkKey> {
	let (dim, tag_off) = match key.len() {
		9 | 10 => (Dimension::Overworld, 8),
		13 | 14 => (dimension_from_index(read_i32(&key[8..12]))?, 12),
		_ => return None,
	};

	let tag = key[tag_off];
	if !is_chunk_tag(tag) {
		return None;
	}

	let cx = read_i32(&key[0..4]);
	let cz = read_i32(&key[4..8]);

	let tag = if tag == TAG_SUBCHUNK {
		// Subchunk keys carry a trailing Y-index byte
		let y = *key.get(tag_off + 1)? as i8;
		KeyTag::SubChunk(y)
	} else if key.len() == tag_off + 1 {
		if tag == TAG_BLOCK_ENTITIES {
			KeyTag::BlockEntities
		} else {
			KeyTag::Other
		}
	} else {
		return None;
	};

	Some(ChunkKey { dim, cx, cz, tag })
}

/// Builds the LevelDB key for a chunk's subchunk at a Y-index
fn subchunk_key(dim: Dimension, cx: i32, cz: i32, y: i8) -> Vec<u8> {
	let mut key = chunk_key_prefix(dim, cx, cz);
	key.push(TAG_SUBCHUNK);
	key.push(y as u8);
	key
}

/// Builds the LevelDB key for a single-tag chunk record
fn tag_key(dim: Dimension, cx: i32, cz: i32, tag: u8) -> Vec<u8> {
	let mut key = chunk_key_prefix(dim, cx, cz);
	key.push(tag);
	key
}

/// Builds the common coordinate (and dimension) prefix of a chunk key
fn chunk_key_prefix(dim: Dimension, cx: i32, cz: i32) -> Vec<u8> {
	let mut key = Vec::with_capacity(13);
	key.extend_from_slice(&cx.to_le_bytes());
	key.extend_from_slice(&cz.to_le_bytes());
	if let Some(index) = dimension_index(dim) {
		key.extend_from_slice(&index.to_le_bytes());
	}
	key
}

/// Number of chunks along each axis of a region
const REGION_SIZE: i32 = CHUNKS_PER_REGION as i32;

/// Which data keys exist for a single chunk
#[derive(Default)]
struct ChunkParts {
	/// Y-indices of present subchunks
	subchunks: Vec<i8>,
	/// Whether the chunk has block entity data
	has_block_entities: bool,
}

/// Index of chunk keys, grouped by dimension and region
type ChunkIndex = BTreeMap<(Dimension, i32, i32), BTreeMap<(i32, i32), ChunkParts>>;

/// Raw (compressed/encoded) data fetched for a single chunk
struct RawChunk {
	/// Chunk coordinates
	cx: i32,
	/// Chunk coordinates
	cz: i32,
	/// Subchunk data keyed by Y-index
	subchunks: Vec<(i8, Vec<u8>)>,
	/// Block entity data, if present
	block_entities: Option<Vec<u8>>,
}

/// Result of processing a single chunk
struct ProcessedChunkResult {
	/// Chunk coordinates
	cx: i32,
	/// Chunk coordinates
	cz: i32,
	/// Processed top-layer chunk data, if the chunk contained any blocks
	chunk: Option<Box<ProcessedChunk>>,
	/// Collected overlay info
	overlay: ChunkOverlayInfo,
}

/// Runs all MinedMap generation steps for a Bedrock Edition world
pub fn generate(config: &Config, rt: &Runtime) -> Result<()> {
	let mut db = BedrockDb::open(&config.input_dir)?;
	let block_types = BlockTypes::default();

	let input_timestamp = {
		let current: PathBuf = [&config.input_dir, Path::new("db/CURRENT")]
			.iter()
			.collect();
		fs::modified_timestamp(&current).unwrap_or_else(|_| SystemTime::now())
	};

	info!("Indexing Bedrock chunks...");
	let index = build_index(&mut db)?;

	fs::create_dir_all(&config.processed_dir)?;

	info!("Processing Bedrock chunks...");
	let mut overlays = OverlayData::default();
	let mut overworld_regions = Vec::new();

	for ((dim, rx, rz), chunks) in &index {
		let raw = fetch_region(&mut db, *dim, chunks);

		if *dim == Dimension::Overworld {
			let coords = TileCoords { x: *rx, z: *rz };
			let region_overlay =
				process_overworld_region(config, &block_types, coords, &raw, input_timestamp)?;
			overlays.overworld.merge(region_overlay);
			overworld_regions.push(coords);
		} else {
			let region_overlay = collect_region_overlay(&block_types, &raw);
			overlays.dimension_mut(*dim).merge(region_overlay);
		}
	}

	// Sort regions in a zig-zag pattern to optimize cache usage (matching the
	// Java region processor).
	overworld_regions
		.sort_unstable_by_key(|&TileCoords { x, z }| (x, if x % 2 == 0 { z } else { -z }));

	info!(
		"Processed Bedrock chunks ({} overworld regions)",
		overworld_regions.len()
	);

	TileRenderer::new(config, rt, &overworld_regions).run()?;
	let tiles = TileMipmapper::new(config, &overworld_regions).run()?;
	MetadataWriter::new(config, &tiles).run()?;

	if let Some(dir) = &config.emit_overlays {
		overlays.write(dir)?;
	}

	Ok(())
}

/// Iterates the database once to record which chunk keys exist
fn build_index(db: &mut BedrockDb) -> Result<ChunkIndex> {
	let mut index = ChunkIndex::new();

	db.for_each_key(|key| {
		let Some(ChunkKey { dim, cx, cz, tag }) = parse_key(key) else {
			return;
		};

		let region = (dim, cx.div_euclid(REGION_SIZE), cz.div_euclid(REGION_SIZE));
		let parts = index
			.entry(region)
			.or_default()
			.entry((cx, cz))
			.or_default();

		match tag {
			KeyTag::SubChunk(y) => parts.subchunks.push(y),
			KeyTag::BlockEntities => parts.has_block_entities = true,
			KeyTag::Other => {}
		}
	})?;

	Ok(index)
}

/// Fetches the raw chunk data for a whole region from the database
///
/// This is done single-threaded (the database handle is not thread-safe);
/// the returned owned data is then processed in parallel.
fn fetch_region(
	db: &mut BedrockDb,
	dim: Dimension,
	chunks: &BTreeMap<(i32, i32), ChunkParts>,
) -> Vec<RawChunk> {
	chunks
		.iter()
		.map(|(&(cx, cz), parts)| {
			let subchunks = parts
				.subchunks
				.iter()
				.filter_map(|&y| db.get(&subchunk_key(dim, cx, cz, y)).map(|data| (y, data)))
				.collect();
			let block_entities = if parts.has_block_entities {
				db.get(&tag_key(dim, cx, cz, TAG_BLOCK_ENTITIES))
			} else {
				None
			};
			RawChunk {
				cx,
				cz,
				subchunks,
				block_entities,
			}
		})
		.collect()
}

/// Decodes the subchunks of a chunk into a map of section Y to storage layer
fn decode_sections(raw: &RawChunk) -> BTreeMap<i32, SubChunkLayer> {
	let mut sections = BTreeMap::new();
	for (y, data) in &raw.subchunks {
		match subchunk::parse_block_layer(data) {
			Ok(Some(layer)) => {
				sections.insert(i32::from(*y), layer);
			}
			Ok(None) => {}
			Err(err) => {
				debug!(
					"Failed to decode subchunk ({}, {}, {}): {:?}",
					raw.cx, y, raw.cz, err
				);
			}
		}
	}
	sections
}

/// Builds the [ProcessedRegion] and overlay data for an overworld region
fn process_overworld_region(
	config: &Config,
	block_types: &BlockTypes,
	coords: TileCoords,
	raw: &[RawChunk],
	timestamp: SystemTime,
) -> Result<overlay::DimensionOverlay> {
	let want_overlays = config.emit_overlays.is_some();

	let results: Vec<ProcessedChunkResult> = raw
		.par_iter()
		.map(|raw_chunk| {
			let sections = decode_sections(raw_chunk);
			let chunk = build_processed_chunk(block_types, &sections);
			let overlay = if want_overlays {
				chunk_overlay_info(block_types, &sections, raw_chunk.block_entities.as_deref())
			} else {
				ChunkOverlayInfo::default()
			};
			ProcessedChunkResult {
				cx: raw_chunk.cx,
				cz: raw_chunk.cz,
				chunk,
				overlay,
			}
		})
		.collect();

	let mut region = ProcessedRegion {
		// A single fallback biome is used for Bedrock worlds; biome-tinted
		// blocks (grass, foliage, water) are colored using plains values.
		biome_list: vec![*block_types_fallback_biome()],
		chunks: Default::default(),
	};
	let mut region_overlay = overlay::DimensionOverlay::default();

	for result in results {
		let chunk_coords = ChunkCoords {
			x: ChunkX::new(result.cx.rem_euclid(REGION_SIZE) as u32),
			z: ChunkZ::new(result.cz.rem_euclid(REGION_SIZE) as u32),
		};
		region.chunks[chunk_coords] = result.chunk;
		if want_overlays {
			region_overlay.add(result.cx, result.cz, &result.overlay);
		}
	}

	storage::write_file(
		&config.processed_path(coords),
		&region,
		storage::Format::Postcard,
		REGION_FILE_META_VERSION,
		timestamp,
	)
	.with_context(|| format!("Failed to write processed region {coords:?}"))?;

	Ok(region_overlay)
}

/// Collects only overlay data for a non-overworld region
fn collect_region_overlay(block_types: &BlockTypes, raw: &[RawChunk]) -> overlay::DimensionOverlay {
	let mut region_overlay = overlay::DimensionOverlay::default();
	for raw_chunk in raw {
		let sections = decode_sections(raw_chunk);
		let info = chunk_overlay_info(block_types, &sections, raw_chunk.block_entities.as_deref());
		region_overlay.add(raw_chunk.cx, raw_chunk.cz, &info);
	}
	region_overlay
}

/// Returns the plains biome used as the fallback for Bedrock worlds
fn block_types_fallback_biome() -> &'static minedmap_resource::Biome {
	use std::sync::OnceLock;
	/// Cached fallback biome value
	static BIOME: OnceLock<minedmap_resource::Biome> = OnceLock::new();
	BIOME.get_or_init(|| {
		let biomes = minedmap_resource::BiomeTypes::default();
		*biomes
			.get("plains")
			.unwrap_or_else(|| biomes.get_fallback())
	})
}

/// Builds top-layer [ProcessedChunk] data from decoded subchunks
fn build_processed_chunk(
	block_types: &BlockTypes,
	sections: &BTreeMap<i32, SubChunkLayer>,
) -> Option<Box<ProcessedChunk>> {
	if sections.is_empty() {
		return None;
	}

	let mut blocks = Box::new(layer::BlockArray::default());
	let mut biomes = Box::new(layer::BiomeArray::default());
	let mut depths = Box::new(layer::DepthArray::default());

	for x in 0..subchunk::SUBCHUNK_SIZE {
		for z in 0..subchunk::SUBCHUNK_SIZE {
			let xz = LayerBlockCoords {
				x: BlockX::new(x),
				z: BlockZ::new(z),
			};
			fill_column(
				block_types,
				sections,
				x,
				z,
				xz,
				&mut blocks,
				&mut biomes,
				&mut depths,
			);
		}
	}

	Some(Box::new(ProcessedChunk {
		blocks,
		biomes,
		depths,
	}))
}

/// Fills the top-layer data for a single block column
#[allow(clippy::too_many_arguments)]
fn fill_column(
	block_types: &BlockTypes,
	sections: &BTreeMap<i32, SubChunkLayer>,
	x: usize,
	z: usize,
	xz: LayerBlockCoords,
	blocks: &mut layer::BlockArray,
	biomes: &mut layer::BiomeArray,
	depths: &mut layer::DepthArray,
) {
	let mut block_set = false;

	for (&section_y, sec) in sections.iter().rev() {
		for y in (0..subchunk::SUBCHUNK_SIZE).rev() {
			let Some(name) = sec.name_at(block_offset(x, y, z)) else {
				continue;
			};
			let (color, _unknown) = blocks::block_color(name, block_types);
			if !color.is(BlockFlag::Opaque) {
				continue;
			}

			if !block_set {
				blocks[xz] = Some(color);
				biomes[xz] = NonZeroU16::new(1);
				block_set = true;
			}

			// Water blocks contribute their color but the depth is taken from
			// the first non-water block below (as in the Java path).
			if color.is(BlockFlag::Water) {
				continue;
			}

			let height = section_y * subchunk::SUBCHUNK_SIZE as i32 + y as i32;
			depths[xz] = Some(BlockHeight(height));
			return;
		}
	}
}

/// Collects [ChunkOverlayInfo] for a Bedrock chunk
fn chunk_overlay_info(
	_block_types: &BlockTypes,
	sections: &BTreeMap<i32, SubChunkLayer>,
	block_entities: Option<&[u8]>,
) -> ChunkOverlayInfo {
	let mut info = ChunkOverlayInfo::default();

	// Bedrock has no direct InhabitedTime equivalent; report 0.
	for sec in sections.values() {
		for name in &sec.palette {
			// Translate so that Bedrock-specific names (golden_rail, portal, ...)
			// are recognized as their Java equivalents.
			info.note_block(blocks::translate_block_name(name));
		}
	}

	if let Some(data) = block_entities {
		match nbt::read_all(data) {
			Ok(values) => {
				for value in values {
					if let Some(id) = value.get("id").and_then(nbt::Value::as_str)
						&& overlay::is_built_block_entity(id)
					{
						info.built += 1;
					}
				}
			}
			Err(err) => warn!("Failed to decode Bedrock block entities: {err:?}"),
		}
	}

	info
}

/// Reads the spawn point from a Bedrock `level.dat` file
///
/// The Bedrock `level.dat` consists of an 8-byte header followed by a single
/// little-endian NBT compound containing `SpawnX`/`SpawnZ` integers.
pub fn read_spawn(input_dir: &Path) -> Option<(i32, i32)> {
	let path: PathBuf = [input_dir, Path::new("level.dat")].iter().collect();
	let data = std::fs::read(path).ok()?;
	let body = data.get(8..)?;
	let mut reader = nbt::Reader::new(body);
	let value = reader.read_value().ok()??;

	let get_int = |key: &str| match value.get(key) {
		Some(nbt::Value::Int(v)) => Some(*v),
		_ => None,
	};
	Some((get_int("SpawnX")?, get_int("SpawnZ")?))
}

#[cfg(test)]
mod test {
	use super::*;
	use regex::RegexSet;

	/// Builds an NBT palette entry compound (little-endian) for a block name
	fn palette_entry(name: &str) -> Vec<u8> {
		let mut data = vec![10u8, 0, 0]; // compound, empty name
		data.push(8); // string tag
		data.extend_from_slice(&4u16.to_le_bytes());
		data.extend_from_slice(b"name");
		data.extend_from_slice(&(name.len() as u16).to_le_bytes());
		data.extend_from_slice(name.as_bytes());
		data.push(0); // end
		data
	}

	/// Builds a subchunk value (version 8) fully filled with palette index 1
	fn subchunk_value(palette: &[&str]) -> Vec<u8> {
		let mut data = vec![8u8, 1]; // version 8, 1 storage layer
		let bits = 2u8; // enough for up to 4 palette entries
		data.push(bits << 1);
		// 16 blocks per word, all set to index 1 (pattern 01 repeated)
		for _ in 0..256 {
			data.extend_from_slice(&0x5555_5555u32.to_le_bytes());
		}
		data.extend_from_slice(&(palette.len() as u32).to_le_bytes());
		for name in palette {
			data.extend_from_slice(&palette_entry(name));
		}
		data
	}

	/// Builds a block entity value (little-endian NBT) for a single block entity
	fn block_entity_value(id: &str) -> Vec<u8> {
		let mut data = vec![10u8, 0, 0]; // compound, empty name
		data.push(8); // string tag "id"
		data.extend_from_slice(&2u16.to_le_bytes());
		data.extend_from_slice(b"id");
		data.extend_from_slice(&(id.len() as u16).to_le_bytes());
		data.extend_from_slice(id.as_bytes());
		data.push(0); // end
		data
	}

	/// Builds a Bedrock-style `level.dat` with the given spawn coordinates
	fn level_dat(spawn_x: i32, spawn_z: i32) -> Vec<u8> {
		let mut body = vec![10u8, 0, 0]; // compound, empty name
		for (name, value) in [("SpawnX", spawn_x), ("SpawnZ", spawn_z)] {
			body.push(3); // int tag
			body.extend_from_slice(&(name.len() as u16).to_le_bytes());
			body.extend_from_slice(name.as_bytes());
			body.extend_from_slice(&value.to_le_bytes());
		}
		body.push(0); // end

		let mut data = Vec::new();
		data.extend_from_slice(&8u32.to_le_bytes()); // version
		data.extend_from_slice(&(body.len() as u32).to_le_bytes());
		data.extend_from_slice(&body);
		data
	}

	fn test_config(input_dir: &Path, output_dir: &Path, overlay_dir: &Path) -> Config {
		let processed_dir = output_dir.join("processed");
		Config {
			edition: Edition::Bedrock,
			input_dir: input_dir.to_path_buf(),
			emit_overlays: Some(overlay_dir.to_path_buf()),
			region_dir: input_dir.join("region"),
			level_dat_path: input_dir.join("level.dat"),
			level_dat_old_path: input_dir.join("level.dat_old"),
			output_dir: output_dir.to_path_buf(),
			entities_dir: processed_dir.join("entities"),
			entities_path_final: processed_dir.join("entities/entities.bin"),
			processed_dir,
			viewer_info_path: output_dir.join("info.json"),
			viewer_entities_path: output_dir.join("entities.json"),
			image_format: ImageFormat::Png,
			sign_patterns: RegexSet::empty(),
			sign_transforms: Vec::new(),
		}
	}

	#[test]
	fn test_bedrock_generate_end_to_end() {
		let base = std::env::temp_dir().join(format!(
			"minedmap-bedrock-test-{}-{}",
			std::process::id(),
			std::time::SystemTime::now()
				.duration_since(std::time::UNIX_EPOCH)
				.unwrap()
				.as_nanos(),
		));
		let input_dir = base.join("world");
		let output_dir = base.join("out");
		let overlay_dir = base.join("overlays");
		std::fs::create_dir_all(input_dir.join("db")).unwrap();

		// Write a minimal Bedrock database
		{
			let mut db = db::open_writable(&input_dir.join("db")).unwrap();
			// Overworld chunk (0,0): one subchunk at Y=0 full of stone, with a
			// rail and farmland in the palette, plus a chest block entity.
			let sub_key = subchunk_key(Dimension::Overworld, 0, 0, 0);
			let palette = [
				"minecraft:air",
				"minecraft:stone",
				"minecraft:golden_rail",
				"minecraft:farmland",
			];
			db.put(&sub_key, &subchunk_value(&palette)).unwrap();
			let be_key = tag_key(Dimension::Overworld, 0, 0, TAG_BLOCK_ENTITIES);
			db.put(&be_key, &block_entity_value("minecraft:chest"))
				.unwrap();
			db.flush().unwrap();
			db.close().unwrap();
		}

		std::fs::write(input_dir.join("level.dat"), level_dat(64, -32)).unwrap();

		let config = test_config(&input_dir, &output_dir, &overlay_dir);
		let rt = tokio::runtime::Builder::new_current_thread()
			.build()
			.unwrap();

		generate(&config, &rt).unwrap();

		// A map tile must have been rendered for region (0, 0)
		assert!(output_dir.join("map/0/r.0.0.png").is_file());
		assert!(output_dir.join("info.json").is_file());

		// Overlay data must reflect the chunk's blocks and block entity
		let features: serde_json::Value = serde_json::from_slice(
			&std::fs::read(overlay_dir.join("block_features.json")).unwrap(),
		)
		.unwrap();
		assert_eq!(features["overworld"]["rail"], serde_json::json!([[0, 0]]));
		assert_eq!(
			features["overworld"]["farmland"],
			serde_json::json!([[0, 0]])
		);
		assert_eq!(
			features["overworld"]["built"],
			serde_json::json!([[0, 0, 1]])
		);

		let heatmap: serde_json::Value = serde_json::from_slice(
			&std::fs::read(overlay_dir.join("inhabited_heatmap.json")).unwrap(),
		)
		.unwrap();
		// Bedrock has no InhabitedTime, so the heatmap is empty
		assert_eq!(heatmap["overworld"], serde_json::json!([]));

		// Spawn point must be read from the Bedrock level.dat
		let info: serde_json::Value =
			serde_json::from_slice(&std::fs::read(output_dir.join("info.json")).unwrap()).unwrap();
		assert_eq!(info["spawn"]["x"], 64);
		assert_eq!(info["spawn"]["z"], -32);

		let _ = std::fs::remove_dir_all(&base);
	}
}
