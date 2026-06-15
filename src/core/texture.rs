//! Loading of block textures for the high-resolution textured map layer
//!
//! When `--block-textures <dir>` is passed, the textured layer samples the
//! top-face texture of each surface block from a user-provided resource pack
//! instead of using a single flat color per block. Textures are looked up
//! lazily by block identifier and cached.

use std::{
	collections::HashMap,
	path::{Path, PathBuf},
	sync::{Arc, RwLock},
};

use image::{RgbaImage, imageops::FilterType};
use indexmap::IndexSet;

use crate::{
	resource::{Biome, block_color},
	types::*,
	world::layer::{BiomeArray, BlockArray, DepthArray, NameArray},
};

/// A loaded and resized block texture
pub struct LoadedTexture {
	/// The texture image, resized to the configured per-block size
	pub image: RgbaImage,
	/// Mean color of the (opaque) texels, used to normalize against the flat color
	pub mean: [f32; 3],
}

/// Lazily loads and caches block top-face textures from a resource pack
pub struct TextureAtlas {
	/// Root directory of the resource pack (or texture directory)
	root: PathBuf,
	/// Output size (in pixels) of each block texture
	scale: u32,
	/// Cache of loaded textures (None marks a known-missing texture)
	cache: RwLock<HashMap<String, Option<Arc<LoadedTexture>>>>,
}

impl TextureAtlas {
	/// Creates a new texture atlas for a resource pack directory
	pub fn new(root: &Path, scale: u32) -> Self {
		TextureAtlas {
			root: root.to_path_buf(),
			scale,
			cache: RwLock::new(HashMap::new()),
		}
	}

	/// Returns the per-block texture size
	pub fn scale(&self) -> u32 {
		self.scale
	}

	/// Builds the candidate file paths for a block identifier
	fn candidates(&self, name: &str) -> Vec<PathBuf> {
		let id = name.strip_prefix("minecraft:").unwrap_or(name);
		let names = [format!("{id}_top"), id.to_string()];
		let dirs = [
			self.root.join("assets/minecraft/textures/block"),
			self.root.join("assets/minecraft/textures/blocks"),
			self.root.join("block"),
			self.root.join("blocks"),
			self.root.clone(),
		];
		let mut paths = Vec::new();
		for dir in &dirs {
			for n in &names {
				paths.push(dir.join(format!("{n}.png")));
			}
		}
		paths
	}

	/// Loads and resizes a texture, computing its mean color
	fn load(&self, name: &str) -> Option<Arc<LoadedTexture>> {
		let path = self.candidates(name).into_iter().find(|p| p.is_file())?;
		let image = image::open(&path).ok()?.to_rgba8();
		let image = image::imageops::resize(&image, self.scale, self.scale, FilterType::Triangle);

		let mut sum = [0f32; 3];
		let mut count = 0u32;
		for pixel in image.pixels() {
			if pixel[3] == 0 {
				continue;
			}
			for i in 0..3 {
				sum[i] += pixel[i] as f32;
			}
			count += 1;
		}
		let mean = if count > 0 {
			[
				sum[0] / count as f32,
				sum[1] / count as f32,
				sum[2] / count as f32,
			]
		} else {
			[1.0; 3]
		};

		Some(Arc::new(LoadedTexture { image, mean }))
	}

	/// Returns the texture for a block identifier, loading it if necessary
	pub fn get(&self, name: &str) -> Option<Arc<LoadedTexture>> {
		// Fast path: shared read lock for the common (already cached) case.
		if let Some(entry) = self.cache.read().unwrap().get(name) {
			return entry.clone();
		}

		// Load outside the lock, then insert under a write lock.
		let loaded = self.load(name);
		self.cache
			.write()
			.unwrap()
			.entry(name.to_string())
			.or_insert(loaded)
			.clone()
	}
}

/// Renders a textured chunk subtile
///
/// Each surface block is drawn using its top-face texture (looked up by name in
/// the atlas), normalized to the block's flat map color so that biome tinting
/// and depth shading are preserved. Blocks without a matching texture fall back
/// to the flat color.
pub fn render_chunk(
	atlas: &TextureAtlas,
	blocks: &BlockArray,
	biomes: &BiomeArray,
	names: &NameArray,
	depths: &DepthArray,
	biome_list: &IndexSet<Biome>,
	name_list: &IndexSet<String>,
) -> RgbaImage {
	let scale = atlas.scale();
	let size = BLOCKS_PER_CHUNK as u32 * scale;
	let mut image = RgbaImage::new(size, size);

	for z in BlockZ::iter() {
		for x in BlockX::iter() {
			let xz = LayerBlockCoords { x, z };
			let Some(block) = blocks[xz] else {
				continue;
			};

			let depth = depths[xz].map(|d| d.0 as f32).unwrap_or(0.0);
			let biome = biomes[xz].and_then(|i| biome_list.get_index(usize::from(i.get()) - 1));
			let flat = block_color(block, biome, depth);

			let texture = names[xz]
				.and_then(|i| name_list.get_index(usize::from(i.get()) - 1))
				.and_then(|name| atlas.get(name));

			let base_x = u32::from(x.0) * scale;
			let base_z = u32::from(z.0) * scale;
			for ty in 0..scale {
				for tx in 0..scale {
					let rgb = match &texture {
						Some(tex) => {
							let texel = tex.image.get_pixel(tx, ty);
							let mut out = [0u8; 3];
							for i in 0..3 {
								let v = if tex.mean[i] > 0.0 {
									texel[i] as f32 * flat[i] / tex.mean[i]
								} else {
									flat[i]
								};
								out[i] = v.clamp(0.0, 255.0) as u8;
							}
							out
						}
						None => [
							flat[0].clamp(0.0, 255.0) as u8,
							flat[1].clamp(0.0, 255.0) as u8,
							flat[2].clamp(0.0, 255.0) as u8,
						],
					};
					image.put_pixel(
						base_x + tx,
						base_z + ty,
						image::Rgba([rgb[0], rgb[1], rgb[2], 255]),
					);
				}
			}
		}
	}

	image
}
