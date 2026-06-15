//! The [RegionProcessor] and related functions

use std::{
	ffi::OsStr,
	path::{Path, PathBuf},
	sync::mpsc,
	time::SystemTime,
};

use anyhow::{Context, Result};
use enum_map::{Enum, EnumMap};
use indexmap::IndexSet;
use minedmap_resource::Biome;
use rayon::prelude::*;
use tracing::{debug, info, warn};

use super::common::*;
use super::java_random;
use super::overlay::{self, OverlayData};
use super::texture;
use crate::{
	io::{fs, storage},
	resource,
	types::*,
	world::{self, layer},
};

/// Parses a filename in the format r.X.Z.mca into the contained X and Z values
fn parse_region_filename(file_name: &OsStr) -> Option<TileCoords> {
	let parts: Vec<_> = file_name.to_str()?.split('.').collect();
	let &["r", x, z, "mca"] = parts.as_slice() else {
		return None;
	};

	Some(TileCoords {
		x: x.parse().ok()?,
		z: z.parse().ok()?,
	})
}

/// Builds an iterator over the regions of input Minecraft save data
fn region_iterator(region_dir: &Path) -> Result<impl Iterator<Item = TileCoords>> {
	Ok(region_dir
		.read_dir()
		.with_context(|| format!("Failed to read directory {}", region_dir.display()))?
		.filter_map(|entry| entry.ok())
		.filter(|entry| {
			(|| {
				// We are only interested in regular files
				let file_type = entry.file_type().ok()?;
				if !file_type.is_file() {
					return None;
				}

				let metadata = entry.metadata().ok()?;
				if metadata.len() == 0 {
					return None;
				}
				Some(())
			})()
			.is_some()
		})
		.filter_map(|entry| parse_region_filename(&entry.file_name())))
}

/// Determines whether the given directory contains any Minecraft region files
pub fn has_regions(region_dir: &Path) -> bool {
	let Ok(mut iter) = region_iterator(region_dir) else {
		return false;
	};
	iter.next().is_some()
}

/// [RegionProcessor::process_region] return values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
enum RegionProcessorStatus {
	/// Region was processed
	Ok,
	/// Region was processed, unknown blocks or biomes were encountered
	OkWithUnknown,
	/// Region was unchanged and skipped
	Skipped,
	/// Reading the region failed, previous processed data is reused
	ErrorOk,
	/// Reading the region failed, no previous data available
	ErrorMissing,
}

/// Data of a region being processed by a [SingleRegionProcessor]
#[derive(Debug)]
struct SingleRegionData {
	/// [IndexSet] of biomes used by the processed region
	biome_list: IndexSet<Biome>,
	/// [IndexSet] of block names used by the processed region (for textures)
	name_list: IndexSet<String>,
	/// Processed region chunk intermediate data
	chunks: ChunkArray<Option<Box<ProcessedChunk>>>,
	/// Lightmap intermediate data
	lightmap: image::GrayAlphaImage,
	/// Textured layer intermediate data, if the textured layer is enabled
	textured: Option<image::RgbaImage>,
	/// Processed entity intermediate data
	entities: ProcessedEntities,
	/// Accumulated overlay data for the region (overworld dimension)
	overlay: overlay::DimensionOverlay,
	/// True if any unknown block or biome types were encountered during processing
	has_unknown: bool,
}

impl Default for SingleRegionData {
	fn default() -> Self {
		/// Width/height of the region data
		const N: u32 = (BLOCKS_PER_CHUNK * CHUNKS_PER_REGION) as u32;

		let lightmap = image::GrayAlphaImage::new(N, N);
		Self {
			biome_list: Default::default(),
			name_list: Default::default(),
			chunks: Default::default(),
			lightmap,
			textured: None,
			entities: Default::default(),
			overlay: Default::default(),
			has_unknown: false,
		}
	}
}

/// Handles processing for a single region
struct SingleRegionProcessor<'a> {
	/// Registry of known block types
	block_types: &'a resource::BlockTypes,
	/// Registry of known biome types
	biome_types: &'a resource::BiomeTypes,
	/// Coordinates of the region this instance is processing
	coords: TileCoords,
	/// Input region filename
	input_path: PathBuf,
	/// Processed region data output filename
	output_path: PathBuf,
	/// Lightmap output filename
	lightmap_path: PathBuf,
	/// Processed entity output filename
	entities_path: PathBuf,
	/// Textured tile output filename
	textured_path: PathBuf,
	/// Timestamp of last modification of input file
	input_timestamp: SystemTime,
	/// Timestamp of last modification of processed region output file (if valid)
	output_timestamp: Option<SystemTime>,
	/// Timestamp of last modification of lightmap output file (if valid)
	lightmap_timestamp: Option<SystemTime>,
	/// Timestamp of last modification of entity list output file (if valid)
	entities_timestamp: Option<SystemTime>,
	/// True if processed region output file needs to be updated
	output_needed: bool,
	/// True if lightmap output file needs to be updated
	lightmap_needed: bool,
	/// True if entity output file needs to be updated
	entities_needed: bool,
	/// True if per-chunk overlay data should be collected
	overlays_needed: bool,
	/// True if the textured tile needs to be updated
	textured_needed: bool,
	/// Texture atlas for the textured layer, if enabled
	texture_atlas: Option<&'a texture::TextureAtlas>,
	/// How to render unrecognized blocks
	unknown_blocks: resource::UnknownBlockMode,
	/// World seed for the slime-chunk overlay, if available
	world_seed: Option<i64>,
	/// Format of generated map tiles
	image_format: image::ImageFormat,
}

impl<'a> SingleRegionProcessor<'a> {
	/// Initializes a [SingleRegionProcessor]
	fn new(processor: &'a RegionProcessor<'a>, coords: TileCoords) -> Result<Self> {
		let input_path = processor.config.region_path(coords);
		let input_timestamp = fs::modified_timestamp(&input_path)?;

		let output_path = processor.config.processed_path(coords);
		let output_timestamp = fs::read_timestamp(&output_path, REGION_FILE_META_VERSION);

		let lightmap_path = processor.config.tile_path(TileKind::Lightmap, 0, coords);
		let lightmap_timestamp = fs::read_timestamp(&lightmap_path, LIGHTMAP_FILE_META_VERSION);

		let entities_path = processor.config.entities_path(0, coords);
		let entities_timestamp = fs::read_timestamp(&entities_path, ENTITIES_FILE_META_VERSION);

		let textured_path = processor.config.tile_path(TileKind::Textured, 0, coords);
		let textured_timestamp = fs::read_timestamp(&textured_path, TEXTURED_FILE_META_VERSION);

		let output_needed = Some(input_timestamp) > output_timestamp;
		let lightmap_needed = Some(input_timestamp) > lightmap_timestamp;
		let entities_needed = Some(input_timestamp) > entities_timestamp;
		let textured_needed =
			processor.texture_atlas.is_some() && Some(input_timestamp) > textured_timestamp;

		Ok(SingleRegionProcessor {
			block_types: &processor.block_types,
			biome_types: &processor.biome_types,
			coords,
			input_path,
			output_path,
			lightmap_path,
			entities_path,
			textured_path,
			input_timestamp,
			output_timestamp,
			lightmap_timestamp,
			entities_timestamp,
			output_needed,
			lightmap_needed,
			entities_needed,
			overlays_needed: processor.config.wants_overlays(),
			textured_needed,
			texture_atlas: processor.texture_atlas.as_ref(),
			unknown_blocks: processor.config.unknown_blocks,
			world_seed: processor.config.world_seed,
			image_format: processor.config.tile_image_format(),
		})
	}

	/// Renders a lightmap subtile from chunk block light data
	fn render_chunk_lightmap(
		block_light: Box<world::layer::BlockLightArray>,
	) -> image::GrayAlphaImage {
		/// Width/height of generated chunk lightmap
		const N: u32 = BLOCKS_PER_CHUNK as u32;

		image::GrayAlphaImage::from_fn(N, N, |x, z| {
			let v: f32 = block_light[LayerBlockCoords {
				x: BlockX::new(x),
				z: BlockZ::new(z),
			}]
			.into();
			image::LumaA([0, (192.0 * (1.0 - v / 15.0)) as u8])
		})
	}

	/// Saves processed region data
	///
	/// The timestamp is the time of the last modification of the input region data.
	fn save_region(&self, processed_region: &ProcessedRegion) -> Result<()> {
		if !self.output_needed {
			return Ok(());
		}

		storage::write_file(
			&self.output_path,
			processed_region,
			storage::Format::Postcard,
			REGION_FILE_META_VERSION,
			self.input_timestamp,
		)
	}

	/// Saves a lightmap tile
	///
	/// The timestamp is the time of the last modification of the input region data.
	fn save_lightmap(&self, lightmap: &image::GrayAlphaImage) -> Result<()> {
		if !self.lightmap_needed {
			return Ok(());
		}

		fs::create_with_timestamp(
			&self.lightmap_path,
			LIGHTMAP_FILE_META_VERSION,
			self.input_timestamp,
			|file| {
				lightmap
					.write_to(file, self.image_format)
					.context("Failed to save image")
			},
		)
	}

	/// Saves a textured tile
	///
	/// The timestamp is the time of the last modification of the input region data.
	fn save_textured(&self, textured: &image::RgbaImage) -> Result<()> {
		if !self.textured_needed {
			return Ok(());
		}

		fs::create_with_timestamp(
			&self.textured_path,
			TEXTURED_FILE_META_VERSION,
			self.input_timestamp,
			|file| {
				textured
					.write_to(file, self.image_format)
					.context("Failed to save image")
			},
		)
	}

	/// Saves processed entity data
	///
	/// The timestamp is the time of the last modification of the input region data.
	fn save_entities(&self, entities: &mut ProcessedEntities) -> Result<()> {
		if !self.entities_needed {
			return Ok(());
		}

		entities.block_entities.sort_unstable();

		storage::write_file(
			&self.entities_path,
			entities,
			storage::Format::Json,
			ENTITIES_FILE_META_VERSION,
			self.input_timestamp,
		)
	}

	/// Processes a single chunk
	fn process_chunk(
		&self,
		data: &mut SingleRegionData,
		chunk_coords: ChunkCoords,
		chunk_data: world::de::Chunk,
	) -> Result<()> {
		if self.overlays_needed {
			let abs_x = self.coords.x * CHUNKS_PER_REGION as i32 + i32::from(chunk_coords.x.0);
			let abs_z = self.coords.z * CHUNKS_PER_REGION as i32 + i32::from(chunk_coords.z.0);
			let mut info = overlay::java_chunk_overlay_info(&chunk_data);
			info.slime = self
				.world_seed
				.is_some_and(|seed| java_random::is_slime_chunk(seed, abs_x, abs_z));
			data.overlay.add(abs_x, abs_z, &info);
		}

		if !self.output_needed
			&& !self.lightmap_needed
			&& !self.entities_needed
			&& !self.textured_needed
		{
			return Ok(());
		}

		let (chunk, has_unknown) =
			world::chunk::Chunk::new(&chunk_data, self.block_types, self.biome_types)
				.with_context(|| format!("Failed to decode chunk {chunk_coords:?}"))?;
		data.has_unknown |= has_unknown;

		if (self.output_needed || self.lightmap_needed || self.textured_needed)
			&& let Some(layer::LayerData {
				blocks,
				biomes,
				names,
				block_light,
				depths,
			}) = world::layer::top_layer(
				&mut data.biome_list,
				&mut data.name_list,
				self.textured_needed,
				self.unknown_blocks,
				&chunk,
			)
			.with_context(|| format!("Failed to process chunk {chunk_coords:?}"))?
		{
			if let (true, Some(atlas), Some(textured)) = (
				self.textured_needed,
				self.texture_atlas,
				data.textured.as_mut(),
			) {
				let chunk_image = texture::render_chunk(
					atlas,
					&blocks,
					&biomes,
					&names,
					&depths,
					&data.biome_list,
					&data.name_list,
				);
				let scale = atlas.scale() as i64;
				image::imageops::overlay(
					textured,
					&chunk_image,
					i64::from(chunk_coords.x.0) * BLOCKS_PER_CHUNK as i64 * scale,
					i64::from(chunk_coords.z.0) * BLOCKS_PER_CHUNK as i64 * scale,
				);
			}

			if self.output_needed {
				data.chunks[chunk_coords] = Some(Box::new(ProcessedChunk {
					blocks,
					biomes,
					depths,
				}));
			}

			if self.lightmap_needed {
				let chunk_lightmap = Self::render_chunk_lightmap(block_light);
				overlay_chunk(&mut data.lightmap, &chunk_lightmap, chunk_coords);
			}
		}

		if self.entities_needed {
			let mut block_entities = chunk.block_entities().with_context(|| {
				format!("Failed to process block entities for chunk {chunk_coords:?}")
			})?;
			data.entities.block_entities.append(&mut block_entities);
		}

		Ok(())
	}

	/// Processes the chunks of the region
	fn process_chunks(&self, data: &mut SingleRegionData) -> Result<()> {
		crate::nbt::region::from_file(&self.input_path)?.foreach_chunk(
			|chunk_coords, chunk_data| self.process_chunk(data, chunk_coords, chunk_data),
		)
	}

	/// Processes the region
	fn run(&self) -> Result<(RegionProcessorStatus, overlay::DimensionOverlay)> {
		if !self.output_needed
			&& !self.lightmap_needed
			&& !self.entities_needed
			&& !self.overlays_needed
			&& !self.textured_needed
		{
			debug!(
				"Skipping unchanged region r.{}.{}.mca",
				self.coords.x, self.coords.z
			);
			return Ok((RegionProcessorStatus::Skipped, Default::default()));
		}

		debug!(
			"Processing region r.{}.{}.mca",
			self.coords.x, self.coords.z
		);

		let mut data = SingleRegionData::default();
		if let Some(atlas) = self.texture_atlas.filter(|_| self.textured_needed) {
			let n = (BLOCKS_PER_CHUNK * CHUNKS_PER_REGION) as u32 * atlas.scale();
			data.textured = Some(image::RgbaImage::new(n, n));
		}

		if let Err(err) = self.process_chunks(&mut data) {
			if self.output_timestamp.is_some()
				&& self.lightmap_timestamp.is_some()
				&& self.entities_timestamp.is_some()
			{
				warn!(
					"Failed to process region {:?}, using old data: {:?}",
					self.coords, err
				);
				return Ok((RegionProcessorStatus::ErrorOk, Default::default()));
			} else {
				warn!(
					"Failed to process region {:?}, no old data available: {:?}",
					self.coords, err
				);
				return Ok((RegionProcessorStatus::ErrorMissing, Default::default()));
			}
		}

		let overlay = std::mem::take(&mut data.overlay);
		let textured = data.textured.take();

		let processed_region = ProcessedRegion {
			biome_list: data.biome_list.into_iter().collect(),
			chunks: data.chunks,
		};

		self.save_region(&processed_region)?;
		self.save_lightmap(&data.lightmap)?;
		self.save_entities(&mut data.entities)?;
		if let Some(textured) = textured {
			self.save_textured(&textured)?;
		}

		let status = if data.has_unknown {
			RegionProcessorStatus::OkWithUnknown
		} else {
			RegionProcessorStatus::Ok
		};
		Ok((status, overlay))
	}
}

/// Type with methods for processing the regions of a Minecraft save directory
///
/// The RegionProcessor builds lightmap tiles as well as processed region data
/// consumed by subsequent generation steps.
pub struct RegionProcessor<'a> {
	/// Registry of known block types
	block_types: resource::BlockTypes,
	/// Registry of known biome types
	biome_types: resource::BiomeTypes,
	/// Texture atlas for the textured layer, if enabled
	texture_atlas: Option<texture::TextureAtlas>,
	/// Common MinedMap configuration from command line
	config: &'a Config,
}

impl<'a> RegionProcessor<'a> {
	/// Constructs a new RegionProcessor
	pub fn new(config: &'a Config) -> Self {
		let texture_atlas = config
			.block_textures
			.as_ref()
			.map(|dir| texture::TextureAtlas::new(dir, config.texture_scale));
		RegionProcessor {
			block_types: resource::BlockTypes::default(),
			biome_types: resource::BiomeTypes::default(),
			texture_atlas,
			config,
		}
	}

	/// Generates a list of all regions of the input Minecraft save data
	fn collect_regions(&self) -> Result<Vec<TileCoords>> {
		Ok(region_iterator(&self.config.region_dir)?.collect())
	}

	/// Processes a single region file
	fn process_region(
		&self,
		coords: TileCoords,
	) -> Result<(RegionProcessorStatus, overlay::DimensionOverlay)> {
		SingleRegionProcessor::new(self, coords)?.run()
	}

	/// Iterates over all region files of a Minecraft save directory
	///
	/// Returns a list of the coordinates of all processed regions, together
	/// with the accumulated overlay data (empty unless `--emit-overlays` was
	/// passed).
	pub fn run(self) -> Result<(Vec<TileCoords>, OverlayData)> {
		use RegionProcessorStatus as Status;

		fs::create_dir_all(&self.config.processed_dir)?;
		fs::create_dir_all(&self.config.tile_dir(TileKind::Lightmap, 0))?;
		fs::create_dir_all(&self.config.entities_dir(0))?;
		if self.texture_atlas.is_some() {
			fs::create_dir_all(&self.config.tile_dir(TileKind::Textured, 0))?;
		}

		info!("Processing region files...");

		let (region_send, region_recv) = mpsc::channel();
		let (status_send, status_recv) = mpsc::channel();
		let (overlay_send, overlay_recv) = mpsc::channel();

		self.collect_regions()?.par_iter().try_for_each(|&coords| {
			let (ret, region_overlay) = self
				.process_region(coords)
				.with_context(|| format!("Failed to process region {coords:?}"))?;

			if ret != Status::ErrorMissing {
				region_send.send(coords).unwrap();
			}

			status_send.send(ret).unwrap();
			overlay_send.send(region_overlay).unwrap();

			anyhow::Ok(())
		})?;

		drop(region_send);
		let mut regions: Vec<_> = region_recv.into_iter().collect();

		drop(status_send);

		// Java rendering only covers the overworld dimension, so all overlay
		// data is merged into the overworld.
		drop(overlay_send);
		let mut overlays = OverlayData::default();
		for region_overlay in overlay_recv {
			overlays.overworld.merge(region_overlay);
		}

		let mut status = EnumMap::<_, usize>::default();
		for ret in status_recv {
			status[ret] += 1;
		}

		info!(
			"Processed region files ({} processed, {} unchanged, {} errors)",
			status[Status::Ok] + status[Status::OkWithUnknown],
			status[Status::Skipped],
			status[Status::ErrorOk] + status[Status::ErrorMissing],
		);

		if status[Status::OkWithUnknown] > 0 {
			warn!("Unknown block or biome types found during processing");
			eprint!(concat!(
				"\n",
				"  If you're encountering this issue with an unmodified Minecraft version supported by MinedMap,\n",
				"  please file a bug report including the output with the --verbose flag.\n",
				"\n",
			));
		}

		// Sort regions in a zig-zag pattern to optimize cache usage
		regions.sort_unstable_by_key(|&TileCoords { x, z }| (x, if x % 2 == 0 { z } else { -z }));

		Ok((regions, overlays))
	}
}
