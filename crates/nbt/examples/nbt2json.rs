//! Converts a single Minecraft NBT data file (such as `level.dat` or a
//! `playerdata/<uuid>.dat`) to JSON on standard output
//!
//! Accepts both gzip-compressed files (the usual on-disk format) and raw
//! uncompressed NBT. This is a small standalone helper for inspecting NBT data
//! or feeding it into tools that work with JSON.

#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

use std::{
	fs::File,
	io::{Read, Write},
	path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use clap::Parser;
use flate2::read::GzDecoder;

/// Convert a Minecraft NBT data file to JSON
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
	/// NBT file to convert (gzip-compressed or raw)
	file: PathBuf,
	/// Pretty-print the JSON output
	#[arg(short, long)]
	pretty: bool,
}

/// Reads a file as NBT bytes, transparently decompressing gzip if present
fn read_nbt(path: &Path) -> Result<Vec<u8>> {
	let mut raw = Vec::new();
	File::open(path)
		.with_context(|| format!("Failed to open {}", path.display()))?
		.read_to_end(&mut raw)
		.with_context(|| format!("Failed to read {}", path.display()))?;

	// gzip files start with the magic bytes 0x1f 0x8b
	if raw.starts_with(&[0x1f, 0x8b]) {
		let mut decoded = Vec::new();
		GzDecoder::new(raw.as_slice())
			.read_to_end(&mut decoded)
			.with_context(|| format!("Failed to decompress {}", path.display()))?;
		Ok(decoded)
	} else {
		Ok(raw)
	}
}

fn main() -> Result<()> {
	let args = Args::parse();

	let data = read_nbt(&args.file)?;
	let value: fastnbt::Value = fastnbt::from_bytes(&data).context("Failed to decode NBT data")?;

	let stdout = std::io::stdout();
	let mut out = stdout.lock();
	if args.pretty {
		serde_json::to_writer_pretty(&mut out, &value).context("Failed to write JSON")?;
	} else {
		serde_json::to_writer(&mut out, &value).context("Failed to write JSON")?;
	}
	out.write_all(b"\n").context("Failed to write JSON")?;

	Ok(())
}
