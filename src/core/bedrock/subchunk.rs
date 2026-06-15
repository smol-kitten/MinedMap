//! Decoding of Bedrock Edition subchunk "paletted storage" data
//!
//! A subchunk (LevelDB key tag `0x2f`) stores a 16×16×16 block volume as one or
//! more storage layers. Each layer consists of a header byte (bits-per-block and
//! a runtime flag), a packed array of palette indices, and a little-endian NBT
//! block palette. Supported container versions are 1, 8 and 9 (version 9 carries
//! an extra subchunk Y-index byte).

use anyhow::{Context, Result, bail};

use super::nbt;

/// Number of blocks along each axis of a subchunk
pub const SUBCHUNK_SIZE: usize = 16;
/// Total number of blocks in a subchunk
const BLOCKS_PER_SUBCHUNK: usize = SUBCHUNK_SIZE * SUBCHUNK_SIZE * SUBCHUNK_SIZE;

/// Computes the storage offset of a block within a subchunk
///
/// Bedrock orders subchunk blocks with X as the most significant and Y as the
/// least significant index.
#[inline]
pub fn block_offset(x: usize, y: usize, z: usize) -> usize {
	(x * SUBCHUNK_SIZE + z) * SUBCHUNK_SIZE + y
}

/// A single decoded storage layer of a subchunk
#[derive(Debug)]
pub struct SubChunkLayer {
	/// Block palette (names including the `minecraft:` prefix)
	pub palette: Vec<String>,
	/// Number of bits used per palette index
	bits: u8,
	/// Packed palette indices, as little-endian 32-bit words
	words: Vec<u32>,
}

impl SubChunkLayer {
	/// Returns the palette index of the block at a storage offset
	#[inline]
	pub fn palette_index(&self, offset: usize) -> usize {
		if self.bits == 0 {
			return 0;
		}
		let bits = self.bits as usize;
		let per_word = 32 / bits;
		let word = offset / per_word;
		let shift = (offset % per_word) * bits;
		let mask = (1u32 << bits) - 1;
		match self.words.get(word) {
			Some(&w) => ((w >> shift) & mask) as usize,
			None => 0,
		}
	}

	/// Returns the block name at a storage offset
	#[inline]
	pub fn name_at(&self, offset: usize) -> Option<&str> {
		self.palette
			.get(self.palette_index(offset))
			.map(String::as_str)
	}
}

/// Reads a little-endian `u32` at `pos`, advancing it
fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32> {
	let end = pos.checked_add(4).filter(|&end| end <= data.len());
	let Some(end) = end else {
		bail!("Unexpected end of subchunk data");
	};
	let value = u32::from_le_bytes(data[*pos..end].try_into().unwrap());
	*pos = end;
	Ok(value)
}

/// Reads a single byte at `pos`, advancing it
fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8> {
	let value = *data.get(*pos).context("Unexpected end of subchunk data")?;
	*pos += 1;
	Ok(value)
}

/// Parses a single storage layer starting at `pos`
fn parse_layer(data: &[u8], pos: &mut usize) -> Result<SubChunkLayer> {
	let header = read_u8(data, pos)?;
	let bits = header >> 1;
	let runtime = (header & 1) == 1;
	if runtime {
		bail!("Runtime block palettes are not supported in saved data");
	}
	// Valid Bedrock palettes use at most 16 bits per block; reject anything
	// larger (which would also cause a divide-by-zero below) as corrupt data.
	if bits > 16 {
		bail!("Invalid subchunk palette bit width {bits}");
	}

	let word_count = if bits == 0 {
		0
	} else {
		let per_word = 32 / bits as usize;
		BLOCKS_PER_SUBCHUNK.div_ceil(per_word)
	};

	let mut words = Vec::with_capacity(word_count);
	for _ in 0..word_count {
		words.push(read_u32(data, pos)?);
	}

	let palette_size = read_u32(data, pos)? as usize;

	let mut reader = nbt::Reader::new(&data[*pos..]);
	let mut palette = Vec::with_capacity(palette_size);
	for _ in 0..palette_size {
		let value = reader
			.read_value()?
			.context("Truncated subchunk block palette")?;
		let name = value
			.get("name")
			.and_then(nbt::Value::as_str)
			.context("Subchunk palette entry missing name")?;
		palette.push(name.to_string());
	}
	*pos += reader.position();

	Ok(SubChunkLayer {
		palette,
		bits,
		words,
	})
}

/// Parses the block storage (first layer) of a subchunk value
///
/// Returns `None` for subchunks without any storage layers.
pub fn parse_block_layer(data: &[u8]) -> Result<Option<SubChunkLayer>> {
	if data.is_empty() {
		return Ok(None);
	}

	let mut pos = 0;
	let version = read_u8(data, &mut pos)?;
	let storage_count = match version {
		1 => 1,
		8 => read_u8(data, &mut pos)?,
		9 => {
			let count = read_u8(data, &mut pos)?;
			// Skip the subchunk Y-index byte (authoritative value comes from the key)
			let _y_index = read_u8(data, &mut pos)?;
			count
		}
		other => bail!("Unsupported subchunk version {other}"),
	};

	if storage_count == 0 {
		return Ok(None);
	}

	let layer = parse_layer(data, &mut pos)?;
	Ok(Some(layer))
}

#[cfg(test)]
mod test {
	use super::*;

	/// Builds a minimal NBT palette entry compound for a block name
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

	#[test]
	fn test_parse_single_palette() {
		// version 8, 1 storage layer, bits=1, palette [air, stone]
		let mut data = vec![8u8, 1];
		data.push(1 << 1); // header: bits=1, not runtime
		// 4096 blocks, 1 bit each, 32 blocks per word -> 128 words.
		// Set block 0 to index 1 (stone), rest 0 (air).
		let mut words = vec![0u32; 128];
		words[0] = 1;
		for w in &words {
			data.extend_from_slice(&w.to_le_bytes());
		}
		data.extend_from_slice(&2u32.to_le_bytes()); // palette size
		data.extend_from_slice(&palette_entry("minecraft:air"));
		data.extend_from_slice(&palette_entry("minecraft:stone"));

		let layer = parse_block_layer(&data).unwrap().unwrap();
		assert_eq!(layer.palette, vec!["minecraft:air", "minecraft:stone"]);
		assert_eq!(
			layer.name_at(block_offset(0, 0, 0)),
			Some("minecraft:stone")
		);
		assert_eq!(layer.name_at(block_offset(0, 1, 0)), Some("minecraft:air"));
	}

	#[test]
	fn test_invalid_bits_rejected() {
		// A corrupt bit width must produce an error instead of panicking with a
		// divide-by-zero (32 / bits == 0 for bits > 32).
		let data = vec![8u8, 1, 40u8]; // version 8, 1 layer, header bits = 20
		assert!(parse_block_layer(&data).is_err());

		// Truncated data must also be handled gracefully.
		assert!(parse_block_layer(&[8u8]).is_err());
		assert!(parse_block_layer(&[]).unwrap().is_none());
	}
}
