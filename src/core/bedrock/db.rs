//! Access to a Bedrock Edition LevelDB database
//!
//! Bedrock uses Mojang's fork of LevelDB, which adds two compression types on
//! top of the upstream "none" and "Snappy" variants: zlib (with header, id 2)
//! and raw DEFLATE (no header, id 4). The database is otherwise a standard
//! LevelDB using the default bytewise key comparator, so it can be read using
//! the pure-Rust [`rusty_leveldb`] implementation once the extra compressors
//! are registered.
//!
//! To avoid mutating the user's save data (opening a LevelDB performs recovery
//! that may rewrite log and manifest files), the database directory is copied
//! to a temporary location before opening.

use std::{
	io::Read,
	path::{Path, PathBuf},
	rc::Rc,
};

use anyhow::{Context, Result};
use flate2::read::{DeflateDecoder, DeflateEncoder, ZlibDecoder, ZlibEncoder};
use rusty_leveldb::{
	Compressor, CompressorId, CompressorList, DB, LdbIterator, Options, Status, StatusCode,
	compressor::{NoneCompressor, SnappyCompressor},
};

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
		// Compressor used for newly written blocks; we only read, but the id
		// must refer to a registered compressor.
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
		let file_type = entry.file_type()?;
		let target = to.join(entry.file_name());
		if file_type.is_dir() {
			copy_dir(&entry.path(), &target)?;
		} else if file_type.is_file() {
			std::fs::copy(entry.path(), &target).with_context(|| {
				format!(
					"Failed to copy {} to {}",
					entry.path().display(),
					target.display()
				)
			})?;
		}
	}
	Ok(())
}

/// A handle to an opened Bedrock LevelDB database (a temporary copy)
pub struct BedrockDb {
	/// The opened database
	db: DB,
	/// Temporary directory holding the database copy
	temp_dir: PathBuf,
}

impl BedrockDb {
	/// Opens the LevelDB database at `<input_dir>/db`
	///
	/// The database is copied to a temporary directory first so that opening it
	/// (which may trigger LevelDB recovery writes) does not modify the save.
	pub fn open(input_dir: &Path) -> Result<Self> {
		let db_dir: PathBuf = [input_dir, Path::new("db")].iter().collect();
		if !db_dir.is_dir() {
			anyhow::bail!("Bedrock database directory {} not found", db_dir.display());
		}

		let temp_dir = std::env::temp_dir().join(format!(
			"minedmap-bedrock-{}-{}",
			std::process::id(),
			std::time::SystemTime::now()
				.duration_since(std::time::UNIX_EPOCH)
				.map(|d| d.as_nanos())
				.unwrap_or(0),
		));

		copy_dir(&db_dir, &temp_dir)
			.context("Failed to copy Bedrock database to temporary directory")?;

		let db = DB::open(&temp_dir, bedrock_options()).map_err(|err| {
			let _ = std::fs::remove_dir_all(&temp_dir);
			anyhow::anyhow!("Failed to open Bedrock database: {err}")
		})?;

		Ok(BedrockDb { db, temp_dir })
	}

	/// Looks up the value for a key
	pub fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
		self.db.get(key).map(|value| value.to_vec())
	}

	/// Calls a closure for every key in the database
	///
	/// Only the keys are passed; values are fetched on demand via [Self::get]
	/// to keep memory usage bounded.
	pub fn for_each_key<F>(&mut self, mut f: F) -> Result<()>
	where
		F: FnMut(&[u8]),
	{
		let mut iter = self
			.db
			.new_iter()
			.map_err(|err| anyhow::anyhow!("Failed to create database iterator: {err}"))?;
		iter.seek_to_first();
		while iter.valid() {
			if let Some((key, _)) = iter.current() {
				f(&key);
			}
			iter.advance();
		}
		Ok(())
	}
}

impl Drop for BedrockDb {
	fn drop(&mut self) {
		let _ = self.db.close();
		let _ = std::fs::remove_dir_all(&self.temp_dir);
	}
}

/// Opens a writable Bedrock-style LevelDB database (test helper)
///
/// Uses the same compressor configuration as reading, with the raw-DEFLATE
/// compressor (id 4) as the write compressor, so that the round-trip exercises
/// the Mojang compression path.
#[cfg(test)]
pub fn open_writable(path: &Path) -> Result<DB> {
	let options = Options {
		create_if_missing: true,
		..bedrock_options()
	};
	DB::open(path, options).map_err(|err| anyhow::anyhow!("Failed to open database: {err}"))
}
