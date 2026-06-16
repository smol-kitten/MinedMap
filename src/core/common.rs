//! Common data types and functions used by multiple generation steps

use std::{
	collections::{BTreeMap, BTreeSet},
	fmt::Debug,
	path::{Path, PathBuf},
	time::{Duration, SystemTime},
};

use anyhow::{Context, Result};
use clap::ValueEnum;
use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};

use crate::{
	core::{overlay::Dimension, region_processor},
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

/// MinedMap mob-spawn tile data version number
///
/// Increase when the generation of mob-spawn tiles changes
pub const MOBSPAWN_FILE_META_VERSION: FileMetaVersion = FileMetaVersion(0);

/// MinedMap contour tile data version number
///
/// Increase when the generation of contour tiles changes
pub const CONTOUR_FILE_META_VERSION: FileMetaVersion = FileMetaVersion(0);

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

/// MinedMap per-region overlay/marker cache version number
///
/// Increase when the cached per-region overlay, POI or mob data layout changes,
/// to invalidate stale `--since` caches.
pub const EMIT_CACHE_META_VERSION: FileMetaVersion = FileMetaVersion(0);

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

/// Converts a (possibly negative) Unix timestamp in seconds to a [SystemTime]
fn unix_to_systemtime(ts: i64) -> SystemTime {
	if ts >= 0 {
		SystemTime::UNIX_EPOCH + Duration::from_secs(ts as u64)
	} else {
		SystemTime::UNIX_EPOCH - Duration::from_secs(ts.unsigned_abs())
	}
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
	/// Mob-spawn (spawn-proofing) tile
	Mobspawn,
	/// Contour (elevation lines) tile
	Contourmap,
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
#[derive(Debug, Clone)]
pub struct Config {
	/// Resolved edition of the input save data
	pub edition: Edition,
	/// Output subdirectory for the current dimension (empty for the overworld)
	pub dim_subdir: PathBuf,
	/// Whether to also render the nether dimension
	pub render_nether: bool,
	/// Whether to also render the end dimension
	pub render_end: bool,
	/// Path of input nether region directory
	pub nether_region_dir: PathBuf,
	/// Path of input end region directory
	pub end_region_dir: PathBuf,
	/// Path of the input Minecraft save directory
	pub input_dir: PathBuf,
	/// Directory to emit overlay data to, if requested
	pub emit_overlays: Option<PathBuf>,
	/// Directory to emit per-player data to, if requested
	pub emit_player_data: Option<PathBuf>,
	/// Override for the player data directory
	pub player_data_dir: Option<PathBuf>,
	/// Override for the player statistics directory
	pub player_stats_dir: Option<PathBuf>,
	/// Explicit player name cache files (overriding auto-detection)
	pub usercache_files: Vec<PathBuf>,
	/// Directory to emit world-level statistics to, if requested
	pub emit_world_stats: Option<PathBuf>,
	/// Only recompute per-region emit data for regions modified after this time
	pub since: Option<SystemTime>,
	/// Whether to generate viewer overlay layers from the overlay data
	pub overlay_layers: bool,
	/// Whether to generate the topographic height layer
	pub height_layer: bool,
	/// Whether to generate the biome/climate layer
	pub biome_layer: bool,
	/// Whether to generate the cave/underground layer
	pub cave_layer: bool,
	/// Whether to generate the mob-spawn (spawn-proofing) layer
	pub mob_spawn: bool,
	/// Whether to generate the contour (elevation lines) layer
	pub contour_layer: bool,
	/// Resource pack directory for the textured layer, if requested
	pub block_textures: Option<PathBuf>,
	/// Per-block texture size (in pixels) for the textured layer
	pub texture_scale: u32,
	/// How to render unrecognized (for example modded) blocks
	pub unknown_blocks: crate::resource::UnknownBlockMode,
	/// World seed, used for the slime-chunk overlay (Java Edition only)
	pub world_seed: Option<i64>,
	/// Whether to collect generated structure bounding boxes
	pub structures: bool,
	/// Whether to collect points of interest for viewer marker layers
	pub poi_markers: bool,
	/// Path of input POI directory
	pub poi_dir: PathBuf,
	/// Whether to collect mob markers for the viewer
	pub mob_markers: bool,
	/// Path of input entity region directory
	pub entity_region_dir: PathBuf,
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
	/// Path of viewer structures file
	pub viewer_structures_path: PathBuf,
	/// Path of viewer mobs file
	pub viewer_mobs_path: PathBuf,
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

		let mut entity_region_dir: PathBuf = [
			&args.input_dir,
			Path::new("dimensions/minecraft/overworld/entities"),
		]
		.iter()
		.collect();
		if !entity_region_dir.is_dir() {
			entity_region_dir = [&args.input_dir, Path::new("entities")].iter().collect();
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
		let viewer_structures_path = [&args.output_dir, Path::new("structures.json")]
			.iter()
			.collect();
		let viewer_mobs_path = [&args.output_dir, Path::new("mobs.json")].iter().collect();

		let sign_patterns = Self::sign_patterns(args).context("Failed to parse sign patterns")?;
		let sign_transforms =
			Self::sign_transforms(args).context("Failed to parse sign transforms")?;

		let edition = args.edition.resolve(&args.input_dir);

		let nether_region_dir = Self::dimension_region_dir(
			&args.input_dir,
			"dimensions/minecraft/the_nether/region",
			"DIM-1/region",
		);
		let end_region_dir = Self::dimension_region_dir(
			&args.input_dir,
			"dimensions/minecraft/the_end/region",
			"DIM1/region",
		);

		// The slime-chunk algorithm is Java-specific, so only read the seed for
		// Java worlds.
		let world_seed = (edition != Edition::Bedrock)
			.then(|| Self::read_world_seed(&level_dat_path, &level_dat_old_path))
			.flatten();

		Ok(Config {
			edition,
			dim_subdir: PathBuf::new(),
			render_nether: args.nether,
			render_end: args.end,
			nether_region_dir,
			end_region_dir,
			input_dir: args.input_dir.clone(),
			emit_overlays: args.emit_overlays.clone(),
			emit_player_data: args.emit_player_data.clone(),
			player_data_dir: args.player_data_dir.clone(),
			player_stats_dir: args.stats_dir.clone(),
			usercache_files: args.usercache.clone(),
			emit_world_stats: args.emit_world_stats.clone(),
			since: args.since.map(unix_to_systemtime),
			overlay_layers: args.overlay_layers,
			height_layer: args.height_layer,
			biome_layer: args.biome_layer,
			cave_layer: args.cave_layer,
			mob_spawn: args.mob_spawn,
			contour_layer: args.contour_layer,
			block_textures: args.block_textures.clone(),
			texture_scale: args.texture_scale,
			unknown_blocks: args.unknown_blocks.into(),
			world_seed,
			structures: args.structures,
			poi_markers: args.poi_markers,
			poi_dir,
			mob_markers: args.mob_markers,
			entity_region_dir,
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
			viewer_structures_path,
			viewer_mobs_path,
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

	/// Constructs the path of a cached per-region overlay contribution
	pub fn overlay_cache_path(&self, coords: TileCoords) -> PathBuf {
		let filename = coord_filename(coords, "overlay");
		[&self.processed_dir, Path::new(&filename)].iter().collect()
	}

	/// Directory holding cached per-region POI contributions
	pub fn poi_cache_dir(&self) -> PathBuf {
		[&self.processed_dir, Path::new("poi")].iter().collect()
	}

	/// Directory holding cached per-region mob contributions
	pub fn mob_cache_dir(&self) -> PathBuf {
		[&self.processed_dir, Path::new("mob")].iter().collect()
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
			TileKind::Mobspawn => "mobspawn",
			TileKind::Contourmap => "contour",
			TileKind::Textured => "textured",
		};
		let dir = format!("{prefix}/{level}");
		Self::join_dim(&self.output_dir, &self.dim_subdir, Path::new(&dir))
	}

	/// Joins an output-relative path, inserting the dimension subdirectory
	fn join_dim(output_dir: &Path, dim_subdir: &Path, rest: &Path) -> PathBuf {
		if dim_subdir.as_os_str().is_empty() {
			[output_dir, rest].iter().collect()
		} else {
			[output_dir, dim_subdir, rest].iter().collect()
		}
	}

	/// Detects a dimension's region directory (modern or legacy layout)
	fn dimension_region_dir(input_dir: &Path, modern: &str, legacy: &str) -> PathBuf {
		let modern_dir: PathBuf = [input_dir, Path::new(modern)].iter().collect();
		if region_processor::has_regions(&modern_dir) {
			modern_dir
		} else {
			[input_dir, Path::new(legacy)].iter().collect()
		}
	}

	/// Returns the dimensions to render and their input region directories
	pub fn dimensions_to_render(&self) -> Vec<(Dimension, PathBuf)> {
		let mut dims = vec![(Dimension::Overworld, self.region_dir.clone())];
		if self.render_nether && region_processor::has_regions(&self.nether_region_dir) {
			dims.push((Dimension::Nether, self.nether_region_dir.clone()));
		}
		if self.render_end && region_processor::has_regions(&self.end_region_dir) {
			dims.push((Dimension::End, self.end_region_dir.clone()));
		}
		dims
	}

	/// Derives a [Config] for a specific dimension and input region directory
	pub fn for_dimension(&self, dimension: Dimension, region_dir: PathBuf) -> Config {
		let dim_subdir = match dimension {
			Dimension::Overworld => PathBuf::new(),
			Dimension::Nether => PathBuf::from("nether"),
			Dimension::End => PathBuf::from("end"),
		};
		let processed_dir = Self::join_dim(&self.output_dir, &dim_subdir, Path::new("processed"));
		let entities_dir: PathBuf = [&processed_dir, Path::new("entities")].iter().collect();
		let entities_path_final: PathBuf =
			[&entities_dir, Path::new("entities.bin")].iter().collect();
		let (poi_dir, entity_region_dir) = match dimension {
			Dimension::Overworld => (self.poi_dir.clone(), self.entity_region_dir.clone()),
			Dimension::Nether => (
				Self::dim_sub_input(&self.input_dir, "the_nether", "DIM-1", "poi"),
				Self::dim_sub_input(&self.input_dir, "the_nether", "DIM-1", "entities"),
			),
			Dimension::End => (
				Self::dim_sub_input(&self.input_dir, "the_end", "DIM1", "poi"),
				Self::dim_sub_input(&self.input_dir, "the_end", "DIM1", "entities"),
			),
		};
		Config {
			dim_subdir,
			region_dir,
			processed_dir,
			entities_dir,
			entities_path_final,
			poi_dir,
			entity_region_dir,
			..self.clone()
		}
	}

	/// Detects a per-dimension input subdirectory (modern or legacy layout)
	fn dim_sub_input(input_dir: &Path, modern_dim: &str, legacy_dim: &str, sub: &str) -> PathBuf {
		let modern: PathBuf = [
			input_dir,
			Path::new(&format!("dimensions/minecraft/{modern_dim}/{sub}")),
		]
		.iter()
		.collect();
		if modern.is_dir() {
			modern
		} else {
			[input_dir, Path::new(&format!("{legacy_dim}/{sub}"))]
				.iter()
				.collect()
		}
	}

	/// Returns whether per-chunk overlay data should be collected
	///
	/// World statistics need the inhabited-chunk data from the overlay pass.
	pub fn wants_overlays(&self) -> bool {
		self.emit_overlays.is_some() || self.overlay_layers || self.emit_world_stats.is_some()
	}

	/// Returns whether generated structure bounding boxes should be collected
	///
	/// Collection is enabled by the `--structures` viewer layer as well as by
	/// `--emit-overlays`, which consolidates all derived data into one directory.
	pub fn collect_structures(&self) -> bool {
		self.structures || self.emit_overlays.is_some()
	}

	/// Returns whether points of interest should be collected
	pub fn collect_pois(&self) -> bool {
		self.poi_markers || self.emit_overlays.is_some()
	}

	/// Returns whether mob markers should be collected
	pub fn collect_mobs(&self) -> bool {
		self.mob_markers || self.emit_overlays.is_some()
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
	fn test_join_dim() {
		// Overworld (empty subdir) keeps the path unchanged for compatibility
		assert_eq!(
			Config::join_dim(Path::new("/out"), Path::new(""), Path::new("map/0")),
			PathBuf::from("/out/map/0"),
		);
		// Other dimensions are placed under a subdirectory
		assert_eq!(
			Config::join_dim(Path::new("/out"), Path::new("nether"), Path::new("map/0")),
			PathBuf::from("/out/nether/map/0"),
		);
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
