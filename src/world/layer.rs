//! Functions to search the "top" layer of a chunk

use std::num::NonZeroU16;

use anyhow::{Context, Result};
use indexmap::IndexSet;
use serde::{Deserialize, Serialize};

use super::chunk::{Chunk, SectionIterItem};
use crate::{
	resource::{Biome, BlockColor, BlockFlag, UnknownBlockMode},
	types::*,
};

/// Height (Y coordinate) of a block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockHeight(pub i32);

impl BlockHeight {
	/// Constructs a new [BlockHeight] from section and block Y indices
	///
	/// Returns an error if the resulting coordindate does not fit into
	/// an [i32].
	pub fn new(section: SectionY, block: BlockY) -> Result<Self> {
		let height = section
			.0
			.checked_mul(BLOCKS_PER_CHUNK as i32)
			.and_then(|y| y.checked_add_unsigned(block.0.into()))
			.context("Block height out of bounds")?;
		Ok(BlockHeight(height))
	}
}

/// Array optionally storing a [BlockColor] for each coordinate of a chunk
pub type BlockArray = LayerBlockArray<Option<BlockColor>>;

/// Array optionally storing a biome index for each coordinate of a chunk
///
/// The entries refer to a biome list generated with the top layer data.
/// Indices are stored incremented by 1 to allow using a [NonZeroU16].
pub type BiomeArray = LayerBlockArray<Option<NonZeroU16>>;

/// Array optionally storing a block-name index for each coordinate of a chunk
///
/// The entries refer to a block-name list generated with the top layer data.
/// Indices are stored incremented by 1 to allow using a [NonZeroU16]. This data
/// is only used in-memory to render the textured layer; it is never serialized.
pub type NameArray = LayerBlockArray<Option<NonZeroU16>>;

/// Array storing a block light value for each coordinate for a chunk
pub type BlockLightArray = LayerBlockArray<u8>;

/// Array optionally storing a depth value for each coordinate for a chunk
pub type DepthArray = LayerBlockArray<Option<BlockHeight>>;

/// References to LayerData entries for a single coordinate pair
struct LayerEntry<'a> {
	/// The block type of the referenced entry
	block: &'a mut Option<BlockColor>,
	/// The biome type of the referenced entry
	biome: &'a mut Option<NonZeroU16>,
	/// The block-name index of the referenced entry
	name: &'a mut Option<NonZeroU16>,
	/// The block light of the referenced entry
	block_light: &'a mut u8,
	/// The depth value of the referenced entry
	depth: &'a mut Option<BlockHeight>,
}

impl LayerEntry<'_> {
	/// Returns true if the entry has not been filled yet (no opaque block has been encountered)
	///
	/// The depth value is filled separately when a non-water block is encountered after the block type
	/// has already been filled.
	fn is_empty(&self) -> bool {
		self.block.is_none()
	}

	/// Returns true if the entry has been filled including its depth (an opaque non-water block has been
	/// encountered)
	fn done(&self) -> bool {
		self.depth.is_some()
	}

	/// Fills in the LayerEntry
	///
	/// Checks whether the passed coordinates point at an opaque or non-water block and
	/// fills in the entry accordingly. Returns true when the block has been filled including its depth.
	fn fill(
		&mut self,
		biome_list: &mut IndexSet<Biome>,
		name_list: &mut IndexSet<String>,
		capture_names: bool,
		unknown: UnknownBlockMode,
		section: SectionIterItem,
		coords: SectionBlockCoords,
	) -> Result<bool> {
		let block_color = match section.section.block_at(coords)? {
			Some(block_type) => Some(block_type.block_color),
			// Unrecognized (for example modded) block: render it according to
			// the configured mode instead of always treating it as transparent.
			None => BlockColor::unknown(
				section.section.block_name_at(coords).unwrap_or_default(),
				unknown,
			),
		};

		let Some(block_color) = block_color.filter(|color| color.is(BlockFlag::Opaque)) else {
			if self.is_empty() {
				*self.block_light = section.block_light.block_light_at(coords);
			}

			return Ok(false);
		};

		if self.is_empty() {
			*self.block = Some(block_color);

			let biome = section.biomes.biome_at(section.y, coords)?;
			let (biome_index, _) = biome_list.insert_full(*biome);
			*self.biome = NonZeroU16::new(
				(biome_index + 1)
					.try_into()
					.expect("biome index not in range"),
			);

			if capture_names && let Some(name) = section.section.block_name_at(coords) {
				let (name_index, _) = name_list.insert_full(name.to_string());
				// Skip the name (texture falls back to the flat color) rather
				// than panicking if a region somehow exceeds u16 distinct names.
				*self.name = u16::try_from(name_index + 1).ok().and_then(NonZeroU16::new);
			}
		}

		if block_color.is(BlockFlag::Water) {
			return Ok(false);
		}

		let height = BlockHeight::new(section.y, coords.y)?;
		*self.depth = Some(height);

		Ok(true)
	}
}

/// Top layer data
///
/// A LayerData stores block type, biome, block light and depth data for
/// each coordinate of a chunk.
#[derive(Debug, Default)]
pub struct LayerData {
	/// Block type data
	pub blocks: Box<BlockArray>,
	/// Biome data
	pub biomes: Box<BiomeArray>,
	/// Block-name index data (for the textured layer; not serialized)
	pub names: Box<NameArray>,
	/// Block light data
	pub block_light: Box<BlockLightArray>,
	/// Depth data
	pub depths: Box<DepthArray>,
}

impl LayerData {
	/// Builds a [LayerEntry] referencing the LayerData at a given coordinate pair
	fn entry(&mut self, coords: LayerBlockCoords) -> LayerEntry<'_> {
		LayerEntry {
			block: &mut self.blocks[coords],
			biome: &mut self.biomes[coords],
			name: &mut self.names[coords],
			block_light: &mut self.block_light[coords],
			depth: &mut self.depths[coords],
		}
	}
}

/// Fills in a [LayerData] with the information of the chunk's top
/// block layer
///
/// For each (X, Z) coordinate pair, the topmost opaque block is
/// determined as the block that should be visible on the rendered
/// map. For water blocks, the height of the first non-water block
/// is additionally filled in as the water depth (the block height is
/// used as depth otherwise).
pub fn top_layer(
	biome_list: &mut IndexSet<Biome>,
	name_list: &mut IndexSet<String>,
	capture_names: bool,
	unknown: UnknownBlockMode,
	chunk: &Chunk,
) -> Result<Option<LayerData>> {
	use BLOCKS_PER_CHUNK as N;

	if chunk.is_empty() {
		return Ok(None);
	}

	let mut done = 0;
	let mut ret = LayerData::default();

	for section in chunk.sections().rev() {
		for y in BlockY::iter().rev() {
			for z in BlockZ::iter() {
				for x in BlockX::iter() {
					let xz = LayerBlockCoords { x, z };

					let mut entry = ret.entry(xz);
					if entry.done() {
						continue;
					}

					let coords = SectionBlockCoords { xz, y };
					if !entry.fill(
						biome_list,
						name_list,
						capture_names,
						unknown,
						section,
						coords,
					)? {
						continue;
					}

					assert!(entry.done());
					done += 1;
					if done == N * N {
						break;
					}
				}
			}
		}
	}

	Ok(Some(ret))
}

/// Fills in a [LayerData] with the floor of the topmost cave in each column
///
/// For each (X, Z) coordinate, the column is scanned downwards: after passing
/// through the surface/roof (the first run of opaque blocks) the first air gap
/// is treated as a cave, and the next opaque block below it is recorded as the
/// cave floor. Columns without a cave under solid ground stay empty.
pub fn cave_layer(biome_list: &mut IndexSet<Biome>, chunk: &Chunk) -> Result<Option<LayerData>> {
	if chunk.is_empty() {
		return Ok(None);
	}

	/// Column has not encountered any opaque block yet (still above the surface)
	const PHASE_ABOVE: u8 = 0;
	/// Column is inside the solid surface/roof
	const PHASE_ROOF: u8 = 1;
	/// Column has entered a cave (air below the roof), looking for the floor
	const PHASE_CAVE: u8 = 2;

	let mut ret = LayerData::default();
	let mut phase = LayerBlockArray::<u8>::default();

	for section in chunk.sections().rev() {
		for y in BlockY::iter().rev() {
			for z in BlockZ::iter() {
				for x in BlockX::iter() {
					let xz = LayerBlockCoords { x, z };
					if ret.depths[xz].is_some() {
						continue;
					}

					let coords = SectionBlockCoords { xz, y };
					let block = section.section.block_at(coords)?;
					let opaque = block
						.is_some_and(|block_type| block_type.block_color.is(BlockFlag::Opaque));

					match (phase[xz], opaque) {
						(PHASE_ABOVE, true) => phase[xz] = PHASE_ROOF,
						(PHASE_ROOF, false) => phase[xz] = PHASE_CAVE,
						(PHASE_CAVE, true) => {
							ret.blocks[xz] = Some(block.unwrap().block_color);

							let biome = section.biomes.biome_at(section.y, coords)?;
							let (biome_index, _) = biome_list.insert_full(*biome);
							ret.biomes[xz] = NonZeroU16::new(
								(biome_index + 1)
									.try_into()
									.expect("biome index not in range"),
							);

							ret.depths[xz] = Some(BlockHeight::new(section.y, y)?);
						}
						_ => {}
					}
				}
			}
		}
	}

	Ok(Some(ret))
}
