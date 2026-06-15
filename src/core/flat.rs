//! Simple flat per-block chunk rendering
//!
//! Renders a chunk subtile by coloring each block with its flat map color
//! (biome tint and depth shading applied, but without the cross-region biome
//! smoothing used for the main map). Used by the cave/underground layer.

use image::RgbaImage;
use indexmap::IndexSet;

use crate::{
	resource::{Biome, block_color},
	types::*,
	world::layer::{BiomeArray, BlockArray, DepthArray},
};

/// Renders a chunk subtile from block, biome and depth data
pub fn render_chunk(
	blocks: &BlockArray,
	biomes: &BiomeArray,
	depths: &DepthArray,
	biome_list: &IndexSet<Biome>,
) -> RgbaImage {
	/// Width/height of a chunk subtile
	const N: u32 = BLOCKS_PER_CHUNK as u32;

	RgbaImage::from_fn(N, N, |x, z| {
		let xz = LayerBlockCoords {
			x: BlockX::new(x),
			z: BlockZ::new(z),
		};
		let Some(block) = blocks[xz] else {
			return image::Rgba([0, 0, 0, 0]);
		};

		let depth = depths[xz].map(|d| d.0 as f32).unwrap_or(0.0);
		let biome = biomes[xz].and_then(|i| biome_list.get_index(usize::from(i.get()) - 1));
		let color = block_color(block, biome, depth);
		image::Rgba([
			color[0].clamp(0.0, 255.0) as u8,
			color[1].clamp(0.0, 255.0) as u8,
			color[2].clamp(0.0, 255.0) as u8,
			255,
		])
	})
}
