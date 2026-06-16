//! Lists the keys of a Bedrock Edition world's LevelDB database
//!
//! A small standalone diagnostic tool: it opens `<world>/db` and prints every
//! key, decoded into a human-readable description where possible (chunk keys are
//! shown as their dimension, chunk coordinates and record type; printable string
//! keys are shown as text; everything else is shown as hex), together with the
//! size of the stored value.
//!
//! The LevelDB open path (Mojang's extra zlib/raw-DEFLATE compressors, opening a
//! temporary copy so the save is not modified) mirrors `src/core/bedrock/db.rs`;
//! it is duplicated here so the tool stays self-contained.

#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

use std::{
	io::Read,
	path::{Path, PathBuf},
	rc::Rc,
};

use anyhow::{Context, Result};
use clap::Parser;
use flate2::read::{DeflateDecoder, DeflateEncoder, ZlibDecoder, ZlibEncoder};
use rusty_leveldb::{
	Compressor, CompressorId, CompressorList, DB, LdbIterator, Options, Status, StatusCode,
	compressor::{NoneCompressor, SnappyCompressor},
};

/// List the keys of a Bedrock Edition LevelDB database
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
	/// Path to the Bedrock world directory (containing a `db` subdirectory)
	world: PathBuf,
	/// Only list keys whose text form starts with this prefix
	#[arg(long, value_name = "PREFIX")]
	prefix: Option<String>,
	/// Also print the size of each key's value (requires reading the values)
	#[arg(long)]
	values: bool,
}

/// Maps an [io::Error](std::io::Error) to a LevelDB [Status]
fn compression_error(err: std::io::Error) -> Status {
	Status::new(StatusCode::CompressionError, &err.to_string())
}

/// Compressor for Mojang's zlib variant (compression type 2)
#[derive(Debug, Clone, Copy, Default)]
struct ZlibCompressor;

impl CompressorId for ZlibCompressor {
	const ID: u8 = 2;
}

impl Compressor for ZlibCompressor {
	fn encode(&self, block: Vec<u8>) -> rusty_leveldb::Result<Vec<u8>> {
		let mut encoder = ZlibEncoder::new(block.as_slice(), flate2::Compression::default());
		let mut out = Vec::new();
		encoder.read_to_end(&mut out).map_err(compression_error)?;
		Ok(out)
	}

	fn decode(&self, block: Vec<u8>) -> rusty_leveldb::Result<Vec<u8>> {
		let mut decoder = ZlibDecoder::new(block.as_slice());
		let mut out = Vec::new();
		decoder.read_to_end(&mut out).map_err(compression_error)?;
		Ok(out)
	}
}

/// Compressor for Mojang's raw DEFLATE variant (compression type 4)
#[derive(Debug, Clone, Copy, Default)]
struct RawZlibCompressor;

impl CompressorId for RawZlibCompressor {
	const ID: u8 = 4;
}

impl Compressor for RawZlibCompressor {
	fn encode(&self, block: Vec<u8>) -> rusty_leveldb::Result<Vec<u8>> {
		let mut encoder = DeflateEncoder::new(block.as_slice(), flate2::Compression::default());
		let mut out = Vec::new();
		encoder.read_to_end(&mut out).map_err(compression_error)?;
		Ok(out)
	}

	fn decode(&self, block: Vec<u8>) -> rusty_leveldb::Result<Vec<u8>> {
		let mut decoder = DeflateDecoder::new(block.as_slice());
		let mut out = Vec::new();
		decoder.read_to_end(&mut out).map_err(compression_error)?;
		Ok(out)
	}
}

/// Builds the [Options] for opening a Bedrock LevelDB database
fn bedrock_options() -> Options {
	let mut list = CompressorList::new();
	list.set(NoneCompressor);
	list.set(SnappyCompressor);
	list.set(ZlibCompressor);
	list.set(RawZlibCompressor);

	Options {
		compressor_list: Rc::new(list),
		compressor: RawZlibCompressor::ID,
		create_if_missing: false,
		..Options::default()
	}
}

/// Recursively copies the contents of a directory
fn copy_dir(from: &Path, to: &Path) -> Result<()> {
	std::fs::create_dir_all(to)
		.with_context(|| format!("Failed to create directory {}", to.display()))?;
	for entry in std::fs::read_dir(from)
		.with_context(|| format!("Failed to read directory {}", from.display()))?
	{
		let entry = entry?;
		let target = to.join(entry.file_name());
		if entry.file_type()?.is_dir() {
			copy_dir(&entry.path(), &target)?;
		} else {
			std::fs::copy(entry.path(), &target)?;
		}
	}
	Ok(())
}

/// Returns a name for a known Bedrock chunk record tag byte
fn chunk_tag_name(tag: u8) -> Option<&'static str> {
	Some(match tag {
		0x2b => "Data3D",
		0x2d => "Data2D",
		0x2f => "SubChunk",
		0x31 => "BlockEntity",
		0x32 => "Entity",
		0x33 => "PendingTicks",
		0x35 => "BiomeState",
		0x36 => "FinalizedState",
		0x38 => "BorderBlocks",
		0x39 => "HardcodedSpawners",
		0x3a => "RandomTicks",
		0x3b => "Checksums",
		0x76 => "LegacyVersion",
		0x78 => "Version",
		_ => return None,
	})
}

/// Maps a Bedrock dimension index to a name
fn dimension_name(index: i32) -> &'static str {
	match index {
		1 => "nether",
		2 => "end",
		_ => "overworld",
	}
}

/// Reads a little-endian `i32` from a 4-byte slice
fn read_i32(data: &[u8]) -> i32 {
	i32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

/// Describes a chunk key as `<dimension> chunk (cx, cz) <tag>`, if it is one
fn describe_chunk_key(key: &[u8]) -> Option<String> {
	let (dim, tag_off) = match key.len() {
		9 | 10 => ("overworld", 8),
		13 | 14 => (dimension_name(read_i32(&key[8..12])), 12),
		_ => return None,
	};
	let tag = key[tag_off];
	let name = chunk_tag_name(tag)?;
	let cx = read_i32(&key[0..4]);
	let cz = read_i32(&key[4..8]);
	if tag == 0x2f && key.len() == tag_off + 2 {
		let y = key[tag_off + 1] as i8;
		Some(format!("{dim} chunk ({cx}, {cz}) {name} y={y}"))
	} else if key.len() == tag_off + 1 {
		Some(format!("{dim} chunk ({cx}, {cz}) {name}"))
	} else {
		None
	}
}

/// Returns the key as text if it is entirely printable ASCII
fn printable_key(key: &[u8]) -> Option<String> {
	if !key.is_empty() && key.iter().all(|&b| b.is_ascii_graphic() || b == b' ') {
		Some(String::from_utf8_lossy(key).into_owned())
	} else {
		None
	}
}

/// Formats a key as a hex string
fn hex_key(key: &[u8]) -> String {
	key.iter().map(|b| format!("{b:02x}")).collect()
}

/// Builds a human-readable description of a key
fn describe_key(key: &[u8]) -> String {
	if let Some(text) = printable_key(key) {
		format!("\"{text}\"")
	} else if let Some(desc) = describe_chunk_key(key) {
		desc
	} else {
		format!("0x{}", hex_key(key))
	}
}

fn main() -> Result<()> {
	let args = Args::parse();

	let db_dir: PathBuf = [&args.world, Path::new("db")].iter().collect();
	if !db_dir.is_dir() {
		anyhow::bail!("Bedrock database directory {} not found", db_dir.display());
	}

	let temp_dir = std::env::temp_dir().join(format!(
		"minedmap-bedrock-keys-{}-{}",
		std::process::id(),
		std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.map(|d| d.as_nanos())
			.unwrap_or(0),
	));
	copy_dir(&db_dir, &temp_dir).context("Failed to copy database to a temporary directory")?;

	let result = (|| -> Result<usize> {
		let mut db = DB::open(&temp_dir, bedrock_options())
			.map_err(|err| anyhow::anyhow!("Failed to open Bedrock database: {err}"))?;

		let mut iter = db
			.new_iter()
			.map_err(|err| anyhow::anyhow!("Failed to create database iterator: {err}"))?;
		iter.seek_to_first();

		let mut count = 0;
		while iter.valid() {
			if let Some((key, value)) = iter.current() {
				let matches = args.prefix.as_ref().is_none_or(|prefix| {
					printable_key(&key).is_some_and(|t| t.starts_with(prefix))
				});
				if matches {
					count += 1;
					if args.values {
						println!("{} ({} bytes)", describe_key(&key), value.len());
					} else {
						println!("{}", describe_key(&key));
					}
				}
			}
			iter.advance();
		}

		let _ = db.close();
		Ok(count)
	})();

	let _ = std::fs::remove_dir_all(&temp_dir);
	let count = result?;
	eprintln!("{count} keys");

	Ok(())
}

#[cfg(test)]
mod test {
	use super::*;

	/// Builds a chunk key from coordinates, optional dimension and a tag suffix
	fn chunk_key(cx: i32, cz: i32, dim: Option<i32>, tail: &[u8]) -> Vec<u8> {
		let mut key = Vec::new();
		key.extend_from_slice(&cx.to_le_bytes());
		key.extend_from_slice(&cz.to_le_bytes());
		if let Some(d) = dim {
			key.extend_from_slice(&d.to_le_bytes());
		}
		key.extend_from_slice(tail);
		key
	}

	#[test]
	fn test_describe_chunk_key() {
		// Overworld subchunk with a Y-index
		let key = chunk_key(1, 2, None, &[0x2f, 0]);
		assert_eq!(
			describe_chunk_key(&key).as_deref(),
			Some("overworld chunk (1, 2) SubChunk y=0")
		);

		// Nether block entity
		let key = chunk_key(-1, 5, Some(1), &[0x31]);
		assert_eq!(
			describe_chunk_key(&key).as_deref(),
			Some("nether chunk (-1, 5) BlockEntity")
		);

		// End version record
		let key = chunk_key(0, 0, Some(2), &[0x78]);
		assert_eq!(
			describe_chunk_key(&key).as_deref(),
			Some("end chunk (0, 0) Version")
		);

		// Unknown tag is not treated as a chunk key
		assert_eq!(describe_chunk_key(&chunk_key(0, 0, None, &[0x01])), None);
	}

	#[test]
	fn test_describe_key() {
		assert_eq!(describe_key(b"~local_player"), "\"~local_player\"");
		assert_eq!(describe_key(b"VILLAGE_x_INFO"), "\"VILLAGE_x_INFO\"");
		// A subchunk key contains NUL bytes, so it is decoded as a chunk key
		assert_eq!(
			describe_key(&chunk_key(3, 4, None, &[0x2f, 1])),
			"overworld chunk (3, 4) SubChunk y=1"
		);
		// Anything else falls back to hex
		assert_eq!(describe_key(&[0x00, 0xff]), "0x00ff");
	}
}
