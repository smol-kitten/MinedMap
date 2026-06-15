//! Common data types and functions used by multiple generation steps

use std::{
	collections::{BTreeMap, BTreeSet},
	fmt::Debug,
	path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use clap::ValueEnum;
use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};

use crate::{
	core::region_processor,
	io::fs::FileMetaVersion,
	resource::Biome,
	types::*,
	world::{block_entity::BlockEntity, layer},
};

// Increase to force regeneration of all output files

/// MinedMap processed region data version number
///
/// Increase when the generation of processed regions from region data changes
/// (usually because of updated resource data)
pub const REGION_FILE_META_VERSION: FileMetaVersion = FileMetaVersion(13);

/// MinedMap map tile data version number
///
/// Increase when the generation of map tiles from processed regions changes
/// (because of code changes in tile generation)
pub const MAP_FILE_META_VERSION: FileMetaVersion = FileMetaVersion(0);

/// MinedMap heightmap tile data version number
///
/// Increase when the generation of heightmap tiles from processed regions
/// changes (because of code changes in heightmap tile generation)
pub const HEIGHTMAP_FILE_META_VERSION: FileMetaVersion = FileMetaVersion(0);

/// MinedMap biome map tile data version number
///
/// Increase when the generation of biome map tiles changes (because of code
/// changes or updated biome color data)
pub const BIOMEMAP_FILE_META_VERSION: FileMetaVersion = FileMetaVersion(0);

/// MinedMap cave map tile data version number
///
/// Increase when the generation of cave map tiles changes (because of code
/// changes or updated block color data)
pub const CAVEMAP_FILE_META_VERSION: FileMetaVersion = FileMetaVersion(0);

/// MinedMap textured tile data version number
///
/// Increase when the generation of textured tiles changes (because of code
/// changes in textured tile generation)
pub const TEXTURED_FILE_META_VERSION: FileMetaVersion = FileMetaVersion(0);

/// MinedMap lightmap data version number
///
/// Increase when the generation of lightmap tiles from region data changes
/// (usually because of updated resource data)
pub const LIGHTMAP_FILE_META_VERSION: FileMetaVersion = FileMetaVersion(10);

/// MinedMap mipmap data version number
///
/// Increase when the mipmap generation changes (this should not happen)
pub const MIPMAP_FILE_META_VERSION: FileMetaVersion = FileMetaVersion(0);

/// MinedMap processed entity data version number
///
/// Increase when entity collection changes bacause of code changes.
pub const ENTITIES_FILE_META_VERSION: FileMetaVersion = FileMetaVersion(4);

/// Coordinate pair of a generated tile
///
/// Each tile corresponds to one Minecraft region file
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileCoords {
	/// The X coordinate
	pub x: i32,
	/// The Z coordinate
	pub z: i32,
}

impl Debug for TileCoords {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "({}, {})", self.x, self.z)
	}
}

/// Set of tile coordinates
///
/// Used to store list of populated tiles for each mipmap level in the
/// viewer metadata file.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(transparent)]
pub struct TileCoordMap(pub BTreeMap<i32, BTreeSet<i32>>);

impl TileCoordMap {
	/// Checks whether the map contains a given coordinate pair
	pub fn contains(&self, coords: TileCoords) -> bool {
		let Some(xs) = self.0.get(&coords.z) else {
			return false;
		};

		xs.contains(&coords.x)
	}
}

/// Data structure for storing chunk data between processing and rendering steps
#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessedChunk {
	/// Block type data
	pub blocks: Box<layer::BlockArray>,
	/// Biome data
	pub biomes: Box<layer::BiomeArray>,
	/// Block height/depth data
	pub depths: Box<layer::DepthArray>,
}

/// Data structure for storing region data between processing and rendering steps
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ProcessedRegion {
	/// List of biomes used in the region
	///
	/// Indexed by [ProcessedChunk] biome data
	pub biome_list: Vec<Biome>,
	/// Processed chunk data
	pub chunks: ChunkArray<Option<Box<ProcessedChunk>>>,
}

/// Data structure for storing entity data between processing and collection steps
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ProcessedEntities {
	/// List of block entities
	pub block_entities: Vec<BlockEntity>,
}

/// Derives a filename from region coordinates and a file extension
///
/// Can be used for input regions, processed data or rendered tiles
fn coord_filename(coords: TileCoords, ext: &str) -> String {
	format!("r.{}.{}.{}", coords.x, coords.z, ext)
}

/// Tile kind corresponding to a map layer
#[derive(Debug, Clone, Copy)]
pub enum TileKind {
	/// Regular map tile contains block colors
	Map,
	/// Lightmap tile for illumination layer
	Lightmap,
	/// Heightmap tile for the topographic layer
	Heightmap,
	/// Biome map tile for the biome/climate layer
	Biomemap,
	/// Cave map tile for the underground layer
	Cavemap,
	/// High-resolution textured map tile
	Textured,
}

/// How unrecognized (for example modded) blocks are rendered
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum UnknownBlocks {
	/// Treat unknown blocks as transparent (do not render them)
	#[default]
	Hide,
	/// Render unknown blocks in a neutral gray
	Gray,
	/// Render unknown blocks in a stable color derived from their name
	Color,
}

impl From<UnknownBlocks> for crate::resource::UnknownBlockMode {
	fn from(value: UnknownBlocks) -> Self {
		match value {
			UnknownBlocks::Hide => crate::resource::UnknownBlockMode::Hide,
			UnknownBlocks::Gray => crate::resource::UnknownBlockMode::Gray,
			UnknownBlocks::Color => crate::resource::UnknownBlockMode::Color,
		}
	}
}

/// Edition of the input Minecraft save data
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum Edition {
	/// Auto-detect the edition from the input directory layout
	#[default]
	Auto,
	/// Java Edition (Anvil region files)
	Java,
	/// Bedrock Edition (LevelDB database)
	Bedrock,
}

impl Edition {
	/// Resolves [Edition::Auto] to a concrete edition based on the input directory
	///
	/// Bedrock Edition is detected by the presence of a `db/CURRENT` file;
	/// anything else is treated as Java Edition.
	pub fn resolve(self, input_dir: &Path) -> Edition {
		match self {
			Edition::Auto => {
				let bedrock_marker: PathBuf = [input_dir, Path::new("db/CURRENT")].iter().collect();
				if bedrock_marker.exists() {
					Edition::Bedrock
				} else {
					Edition::Java
				}
			}
			other => other,
		}
	}
}

/// `WorldGenSettings` element of level.dat, used to read the world seed (1.16+)
#[derive(Debug, Deserialize)]
struct SeedWorldGenSettings {
	/// World seed
	seed: Option<i64>,
}

/// `Data` element of level.dat, reduced to the world seed fields
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct SeedDataInner {
	/// World seed (pre-1.16)
	random_seed: Option<i64>,
	/// World generation settings containing the seed (1.16+)
	world_gen_settings: Option<SeedWorldGenSettings>,
}

/// Minimal level.dat structure for reading the world seed
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct SeedData {
	/// The `Data` field
	data: SeedDataInner,
}

/// Common configuration based on command line arguments
#[derive(Debug)]
pub struct Config {
	/// Resolved edition of the input save data
	pub edition: Edition,
	/// Path of the input Minecraft save directory
	pub input_dir: PathBuf,
	/// Directory to emit overlay data to, if requested
	pub emit_overlays: Option<PathBuf>,
	/// Whether to generate viewer overlay layers from the overlay data
	pub overlay_layers: bool,
	/// Whether to generate the topographic height layer
	pub height_layer: bool,
	/// Whether to generate the biome/climate layer
	pub biome_layer: bool,
	/// Whether to generate the cave/underground layer
	pub cave_layer: bool,
	/// Resource pack directory for the textured layer, if requested
	pub block_textures: Option<PathBuf>,
	/// Per-block texture size (in pixels) for the textured layer
	pub texture_scale: u32,
	/// How to render unrecognized (for example modded) blocks
	pub unknown_blocks: crate::resource::UnknownBlockMode,
	/// World seed, used for the slime-chunk overlay (Java Edition only)
	pub world_seed: Option<i64>,
	/// Whether to collect points of interest for viewer marker layers
	pub poi_markers: bool,
	/// Path of input POI directory
	pub poi_dir: PathBuf,
	/// Path of input region directory
	pub region_dir: PathBuf,
	/// Path of input `level.dat` file
	pub level_dat_path: PathBuf,
	/// Path of input `level.dat_old` file
	pub level_dat_old_path: PathBuf,
	/// Base path for storage of rendered tile data
	pub output_dir: PathBuf,
	/// Path for storage of intermediate processed data files
	pub processed_dir: PathBuf,
	/// Path for storage of processed entity data files
	pub entities_dir: PathBuf,
	/// Path for storage of the final merged processed entity data file
	pub entities_path_final: PathBuf,
	/// Path of viewer metadata file
	pub viewer_info_path: PathBuf,
	/// Path of viewer entities file
	pub viewer_entities_path: PathBuf,
	/// Path of viewer POI file
	pub viewer_pois_path: PathBuf,
	/// Format of generated map tiles
	pub image_format: ImageFormat,
	/// Sign text filter patterns
	pub sign_patterns: RegexSet,
	/// Sign text transformation pattern
	pub sign_transforms: Vec<(Regex, String)>,
}

impl Config {
	/// Crates a new [Config] from [command line arguments](super::Args)
	pub fn new(args: &super::Args) -> Result<Self> {
		let mut region_dir: PathBuf = [
			&args.input_dir,
			Path::new("dimensions/minecraft/overworld/region"),
		]
		.iter()
		.collect();

		if !region_processor::has_regions(&region_dir) {
			region_dir = [&args.input_dir, Path::new("region")].iter().collect();
		}

		let mut poi_dir: PathBuf = [
			&args.input_dir,
			Path::new("dimensions/minecraft/overworld/poi"),
		]
		.iter()
		.collect();
		if !poi_dir.is_dir() {
			poi_dir = [&args.input_dir, Path::new("poi")].iter().collect();
		}

		let level_dat_path: PathBuf = [&args.input_dir, Path::new("level.dat")].iter().collect();
		let level_dat_old_path: PathBuf = [&args.input_dir, Path::new("level.dat_old")]
			.iter()
			.collect();
		let processed_dir: PathBuf = [&args.output_dir, Path::new("processed")].iter().collect();
		let entities_dir: PathBuf = [&processed_dir, Path::new("entities")].iter().collect();
		let entities_path_final = [&entities_dir, Path::new("entities.bin")].iter().collect();
		let viewer_info_path = [&args.output_dir, Path::new("info.json")].iter().collect();
		let viewer_entities_path = [&args.output_dir, Path::new("entities.json")]
			.iter()
			.collect();
		let viewer_pois_path = [&args.output_dir, Path::new("pois.json")].iter().collect();

		let sign_patterns = Self::sign_patterns(args).context("Failed to parse sign patterns")?;
		let sign_transforms =
			Self::sign_transforms(args).context("Failed to parse sign transforms")?;

		let edition = args.edition.resolve(&args.input_dir);

		// The slime-chunk algorithm is Java-specific, so only read the seed for
		// Java worlds.
		let world_seed = (edition != Edition::Bedrock)
			.then(|| Self::read_world_seed(&level_dat_path, &level_dat_old_path))
			.flatten();

		Ok(Config {
			edition,
			input_dir: args.input_dir.clone(),
			emit_overlays: args.emit_overlays.clone(),
			overlay_layers: args.overlay_layers,
			height_layer: args.height_layer,
			biome_layer: args.biome_layer,
			cave_layer: args.cave_layer,
			block_textures: args.block_textures.clone(),
			texture_scale: args.texture_scale,
			unknown_blocks: args.unknown_blocks.into(),
			world_seed,
			poi_markers: args.poi_markers,
			poi_dir,
			region_dir,
			level_dat_path,
			level_dat_old_path,
			output_dir: args.output_dir.clone(),
			processed_dir,
			entities_dir,
			entities_path_final,
			viewer_info_path,
			viewer_entities_path,
			viewer_pois_path,
			image_format: args.image_format,
			sign_patterns,
			sign_transforms,
		})
	}

	/// Reads the world seed from a Java Edition level.dat (1.16+ or older)
	fn read_world_seed(level_dat_path: &Path, level_dat_old_path: &Path) -> Option<i64> {
		let data: SeedData = crate::nbt::data::from_file(level_dat_path)
			.or_else(|_| crate::nbt::data::from_file(level_dat_old_path))
			.ok()?;
		data.data
			.world_gen_settings
			.and_then(|settings| settings.seed)
			.or(data.data.random_seed)
	}

	/// Parses the sign prefixes and sign filters into a [RegexSet]
	fn sign_patterns(args: &super::Args) -> Result<RegexSet> {
		let prefix_patterns: Vec<_> = args
			.sign_prefix
			.iter()
			.map(|prefix| format!("^{}", regex::escape(prefix)))
			.collect();
		Ok(RegexSet::new(
			prefix_patterns.iter().chain(args.sign_filter.iter()),
		)?)
	}

	/// Parses the sign transform argument into a vector of [Regex] and
	/// corresponding replacement strings
	fn sign_transforms(args: &super::Args) -> Result<Vec<(Regex, String)>> {
		let splitter = Regex::new(r"^s/((?:[^\\/]|\\.)*)/((?:[^\\/]|\\.)*)/$").unwrap();

		args.sign_transform
			.iter()
			.map(|t| Self::sign_transform(&splitter, t))
			.collect()
	}

	/// Parses the sign transform argument into a [Regex] and its corresponding
	/// replacement string
	fn sign_transform(splitter: &Regex, transform: &str) -> Result<(Regex, String)> {
		let captures = splitter
			.captures(transform)
			.with_context(|| format!("Invalid transform pattern '{transform}'"))?;
		let regexp = Regex::new(&captures[1])?;
		let replacement = captures[2].to_string();
		Ok((regexp, replacement))
	}

	/// Constructs the path to an input region file
	pub fn region_path(&self, coords: TileCoords) -> PathBuf {
		let filename = coord_filename(coords, "mca");
		[&self.region_dir, Path::new(&filename)].iter().collect()
	}

	/// Constructs the path of an intermediate processed region file
	pub fn processed_path(&self, coords: TileCoords) -> PathBuf {
		let filename = coord_filename(coords, "bin");
		[&self.processed_dir, Path::new(&filename)].iter().collect()
	}

	/// Constructs the base output path for processed entity data
	pub fn entities_dir(&self, level: usize) -> PathBuf {
		[&self.entities_dir, Path::new(&level.to_string())]
			.iter()
			.collect()
	}

	/// Constructs the path of a processed entity data file
	pub fn entities_path(&self, level: usize, coords: TileCoords) -> PathBuf {
		let filename = coord_filename(coords, "bin");
		let dir = self.entities_dir(level);
		[Path::new(&dir), Path::new(&filename)].iter().collect()
	}

	/// Constructs the base output path for a [TileKind] and mipmap level
	pub fn tile_dir(&self, kind: TileKind, level: usize) -> PathBuf {
		let prefix = match kind {
			TileKind::Map => "map",
			TileKind::Lightmap => "light",
			TileKind::Heightmap => "height",
			TileKind::Biomemap => "biome",
			TileKind::Cavemap => "cave",
			TileKind::Textured => "textured",
		};
		let dir = format!("{prefix}/{level}");
		[&self.output_dir, Path::new(&dir)].iter().collect()
	}

	/// Returns whether per-chunk overlay data should be collected
	pub fn wants_overlays(&self) -> bool {
		self.emit_overlays.is_some() || self.overlay_layers
	}

	/// Returns the directory viewer overlay layers are written to
	pub fn overlay_layers_dir(&self) -> PathBuf {
		[&self.output_dir, Path::new("overlays")].iter().collect()
	}

	/// Returns the directories the overlay data files should be written to
	pub fn overlay_output_dirs(&self) -> Vec<PathBuf> {
		let mut dirs = Vec::new();
		if let Some(dir) = &self.emit_overlays {
			dirs.push(dir.clone());
		}
		if self.overlay_layers {
			dirs.push(self.overlay_layers_dir());
		}
		dirs
	}

	/// Returns the file extension for the configured image format
	pub fn tile_extension(&self) -> &'static str {
		match self.image_format {
			ImageFormat::Png => "png",
			ImageFormat::Webp => "webp",
		}
	}
	/// Returns the configurured image format for the image library
	pub fn tile_image_format(&self) -> image::ImageFormat {
		match self.image_format {
			ImageFormat::Png => image::ImageFormat::Png,
			ImageFormat::Webp => image::ImageFormat::WebP,
		}
	}

	/// Constructs the path of an output tile image
	pub fn tile_path(&self, kind: TileKind, level: usize, coords: TileCoords) -> PathBuf {
		let filename = coord_filename(coords, self.tile_extension());
		let dir = self.tile_dir(kind, level);
		[Path::new(&dir), Path::new(&filename)].iter().collect()
	}
}

/// Format of generated map tiles
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum ImageFormat {
	/// Generate PNG images
	#[default]
	Png,
	/// Generate WebP images
	Webp,
}

/// Copies a chunk image into a region tile
pub fn overlay_chunk<I, J>(image: &mut I, chunk: &J, coords: ChunkCoords)
where
	I: image::GenericImage,
	J: image::GenericImageView<Pixel = I::Pixel>,
{
	image::imageops::overlay(
		image,
		chunk,
		coords.x.0 as i64 * BLOCKS_PER_CHUNK as i64,
		coords.z.0 as i64 * BLOCKS_PER_CHUNK as i64,
	);
}

#[cfg(test)]
mod test {
	use super::*;

	fn seed_from_nbt(value: &fastnbt::Value) -> Option<i64> {
		let bytes = fastnbt::to_bytes(value).unwrap();
		let data: SeedData = fastnbt::from_bytes(&bytes).unwrap();
		data.data
			.world_gen_settings
			.and_then(|settings| settings.seed)
			.or(data.data.random_seed)
	}

	#[test]
	fn test_world_seed_parsing() {
		// 1.16+ layout: Data.WorldGenSettings.seed
		let v1_16 = fastnbt::nbt!({
			"Data": { "WorldGenSettings": { "seed": 123456789i64 } },
		});
		assert_eq!(seed_from_nbt(&v1_16), Some(123456789));

		// Pre-1.16 layout: Data.RandomSeed
		let v0 = fastnbt::nbt!({
			"Data": { "RandomSeed": -42i64 },
		});
		assert_eq!(seed_from_nbt(&v0), Some(-42));
	}
}
