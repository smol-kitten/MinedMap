//! Helpers and common functions for filesystem access

use std::{
	fs::{self, File},
	io::{BufReader, BufWriter, Read, Write},
	path::{Path, PathBuf},
	time::SystemTime,
};

use anyhow::{Context, Ok, Result};
use serde::{Deserialize, Serialize};

/// A file metadata version number
///
/// Deserialized metadata with non-current version number are considered invalid
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileMetaVersion(pub u32);

/// Metadata stored with generated files to track required incremental updates
#[derive(Debug, Serialize, Deserialize)]
struct FileMeta {
	/// Version of data described by the FileMeta
	version: FileMetaVersion,
	/// Timestamp stored with generated data
	///
	/// This timestamp is always the time of last modification of the inputs
	/// that were used to generate the file described by the FileMeta.
	timestamp: SystemTime,
}

/// Helper for creating suffixed file paths
fn suffix_name(path: &Path, suffix: &str) -> PathBuf {
	let mut file_name = path.file_name().unwrap_or_default().to_os_string();
	file_name.push(suffix);

	let mut ret = path.to_path_buf();
	ret.set_file_name(file_name);
	ret
}

/// Derives the filename for temporary storage of data during generation
fn tmpfile_name(path: &Path) -> PathBuf {
	suffix_name(path, ".tmp")
}

/// Derives the filename for associated metadata for generated files
fn metafile_name(path: &Path) -> PathBuf {
	suffix_name(path, ".meta")
}

/// Creates a directory including all its parents
///
/// Wrapper around [fs::create_dir_all] that adds a more descriptive error message
pub fn create_dir_all(path: &Path) -> Result<()> {
	fs::create_dir_all(path)
		.with_context(|| format!("Failed to create directory {}", path.display(),))
}

/// Renames a file or directory
///
/// Wrapper around [fs::rename] that adds a more descriptive error message
pub fn rename(from: &Path, to: &Path) -> Result<()> {
	fs::rename(from, to)
		.with_context(|| format!("Failed to rename {} to {}", from.display(), to.display()))
}

/// Creates a new file
///
/// The contents of the file are defined by the passed function.
pub fn create<T, F>(path: &Path, f: F) -> Result<T>
where
	F: FnOnce(&mut BufWriter<File>) -> Result<T>,
{
	(|| {
		let file = File::create(path)?;
		let mut writer = BufWriter::new(file);

		let ret = f(&mut writer)?;
		writer.flush()?;

		Ok(ret)
	})()
	.with_context(|| format!("Failed to write file {}", path.display()))
}

/// Checks whether the contents of two files are equal
///
/// The file sizes are compared first as a cheap shortcut, and the contents are
/// then compared in blocks rather than byte by byte, as this function runs for
/// every generated file on each (incremental) run.
pub fn equal(path1: &Path, path2: &Path) -> Result<bool> {
	let len1 = fs::metadata(path1)
		.with_context(|| format!("Failed to read metadata of {}", path1.display()))?
		.len();
	let len2 = fs::metadata(path2)
		.with_context(|| format!("Failed to read metadata of {}", path2.display()))?
		.len();
	if len1 != len2 {
		return Ok(false);
	}

	let mut file1 = BufReader::new(
		fs::File::open(path1)
			.with_context(|| format!("Failed to open file {}", path1.display()))?,
	);
	let mut file2 = BufReader::new(
		fs::File::open(path2)
			.with_context(|| format!("Failed to open file {}", path2.display()))?,
	);

	let mut buf1 = vec![0u8; 64 * 1024];
	let mut buf2 = vec![0u8; 64 * 1024];
	loop {
		let n = file1
			.read(&mut buf1)
			.with_context(|| format!("Failed to read file {}", path1.display()))?;
		if n == 0 {
			// The sizes are equal, so both files reach EOF together.
			break Ok(true);
		}
		// The equal sizes guarantee path2 has at least n more bytes.
		file2
			.read_exact(&mut buf2[..n])
			.with_context(|| format!("Failed to read file {}", path2.display()))?;
		if buf1[..n] != buf2[..n] {
			break Ok(false);
		}
	}
}

/// Creates a new file, temporarily storing its contents in a temporary file
///
/// Storing the data in a temporary file prevents leaving half-written files
/// when the function is interrupted. In addition, the old and new contents of
/// the file are compared if a file with the same name already exists, and the
/// file timestamp is only updated if the contents have changed.
pub fn create_with_tmpfile<T, F>(path: &Path, f: F) -> Result<T>
where
	F: FnOnce(&mut BufWriter<File>) -> Result<T>,
{
	let tmp_path = tmpfile_name(path);
	let mut cleanup = true;

	let ret = (|| {
		let ret = create(&tmp_path, f)?;
		if !matches!(equal(path, &tmp_path), Result::Ok(true)) {
			rename(&tmp_path, path)?;
			cleanup = false;
		}
		Ok(ret)
	})();

	if cleanup {
		let _ = fs::remove_file(&tmp_path);
	}

	ret
}

/// Returns the time of last modification for a given file path
pub fn modified_timestamp(path: &Path) -> Result<SystemTime> {
	fs::metadata(path)
		.and_then(|meta| meta.modified())
		.with_context(|| {
			format!(
				"Failed to get modified timestamp of file {}",
				path.display()
			)
		})
}

/// Returns the time of last modification for a given file path, or `Ok(None)`
/// if the file does not exist
///
/// A missing source file is an expected condition for layers that are not
/// generated for every region (e.g. the lightmap layer is never produced for
/// Bedrock worlds), so it must not be treated as an error.
pub fn modified_timestamp_opt(path: &Path) -> Result<Option<SystemTime>> {
	match fs::metadata(path).and_then(|meta| meta.modified()) {
		Result::Ok(ts) => Ok(Some(ts)),
		Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
		Err(err) => Err(err).with_context(|| {
			format!(
				"Failed to get modified timestamp of file {}",
				path.display()
			)
		}),
	}
}

/// Reads the stored timestamp from file metadata for a file previously written
/// using [create_with_timestamp]
pub fn read_timestamp(path: &Path, version: FileMetaVersion) -> Option<SystemTime> {
	let meta_path = metafile_name(path);
	let mut file = BufReader::new(fs::File::open(meta_path).ok()?);

	let meta: FileMeta = serde_json::from_reader(&mut file).ok()?;
	if meta.version != version {
		return None;
	}

	Some(meta.timestamp)
}

/// Creates a new file, temporarily storing its contents in a temporary file
/// like [create_with_tmpfile], and storing a timestamp in a metadata file
/// if successful
///
/// The timestamp can be retrieved later using [read_timestamp].
pub fn create_with_timestamp<T, F>(
	path: &Path,
	version: FileMetaVersion,
	timestamp: SystemTime,
	f: F,
) -> Result<T>
where
	F: FnOnce(&mut BufWriter<File>) -> Result<T>,
{
	let ret = create_with_tmpfile(path, f)?;

	let meta_path = metafile_name(path);
	create(&meta_path, |file| {
		serde_json::to_writer(file, &FileMeta { version, timestamp })?;
		Ok(())
	})?;

	Ok(ret)
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_equal() {
		let dir = std::env::temp_dir().join(format!(
			"minedmap-fs-test-{}-{:?}",
			std::process::id(),
			std::thread::current().id()
		));
		fs::create_dir_all(&dir).unwrap();
		let write = |name: &str, data: &[u8]| {
			let path = dir.join(name);
			fs::write(&path, data).unwrap();
			path
		};

		// Multi-block content (larger than the 64 KiB comparison buffer)
		let big: Vec<u8> = (0..200_000u32).map(|i| i as u8).collect();
		let mut big_diff = big.clone();
		*big_diff.last_mut().unwrap() ^= 0xff;

		let a = write("a", &big);
		let b = write("b", &big);
		let c = write("c", &big_diff); // same length, last byte differs
		let d = write("d", b"short");
		let empty1 = write("e1", b"");
		let empty2 = write("e2", b"");

		assert!(equal(&a, &b).unwrap());
		assert!(!equal(&a, &c).unwrap());
		assert!(!equal(&a, &d).unwrap());
		assert!(equal(&empty1, &empty2).unwrap());

		let _ = fs::remove_dir_all(&dir);
	}
}
