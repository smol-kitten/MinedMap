//! Minimal little-endian NBT reader for Bedrock Edition data
//!
//! Bedrock Edition stores NBT in little-endian byte order (as opposed to the
//! big-endian format used by Java Edition and handled by the `fastnbt` crate).
//! Only the subset of functionality needed to read block palette entries and
//! block entities is implemented here.

use anyhow::{Result, bail};

/// A parsed NBT value
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
	/// `TAG_Byte`
	Byte(i8),
	/// `TAG_Short`
	Short(i16),
	/// `TAG_Int`
	Int(i32),
	/// `TAG_Long`
	Long(i64),
	/// `TAG_Float`
	Float(f32),
	/// `TAG_Double`
	Double(f64),
	/// `TAG_Byte_Array`
	ByteArray(Vec<u8>),
	/// `TAG_String`
	String(String),
	/// `TAG_List`
	List(Vec<Value>),
	/// `TAG_Compound`
	Compound(Vec<(String, Value)>),
	/// `TAG_Int_Array`
	IntArray(Vec<i32>),
	/// `TAG_Long_Array`
	LongArray(Vec<i64>),
}

impl Value {
	/// Returns the value of a named entry of a compound
	pub fn get(&self, key: &str) -> Option<&Value> {
		match self {
			Value::Compound(entries) => {
				entries.iter().find(|(name, _)| name == key).map(|(_, v)| v)
			}
			_ => None,
		}
	}

	/// Returns the contained string, if the value is a [Value::String]
	pub fn as_str(&self) -> Option<&str> {
		match self {
			Value::String(s) => Some(s),
			_ => None,
		}
	}
}

/// Cursor-based reader over little-endian NBT data
pub struct Reader<'a> {
	/// The remaining input data
	data: &'a [u8],
	/// Current read position
	pos: usize,
}

impl<'a> Reader<'a> {
	/// Creates a new reader over the given data
	pub fn new(data: &'a [u8]) -> Self {
		Reader { data, pos: 0 }
	}

	/// Returns whether the whole input has been consumed
	pub fn at_end(&self) -> bool {
		self.pos >= self.data.len()
	}

	/// Returns the number of bytes consumed so far
	pub fn position(&self) -> usize {
		self.pos
	}

	/// Reads a fixed number of bytes
	fn bytes(&mut self, len: usize) -> Result<&'a [u8]> {
		let end = self
			.pos
			.checked_add(len)
			.filter(|&end| end <= self.data.len())
			.ok_or_else(|| anyhow::anyhow!("Unexpected end of NBT data"))?;
		let ret = &self.data[self.pos..end];
		self.pos = end;
		Ok(ret)
	}

	/// Reads a single byte
	fn u8(&mut self) -> Result<u8> {
		Ok(self.bytes(1)?[0])
	}

	/// Reads a little-endian `i16`
	fn i16(&mut self) -> Result<i16> {
		Ok(i16::from_le_bytes(self.bytes(2)?.try_into().unwrap()))
	}

	/// Reads a little-endian `i32`
	fn i32(&mut self) -> Result<i32> {
		Ok(i32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
	}

	/// Reads a little-endian `i64`
	fn i64(&mut self) -> Result<i64> {
		Ok(i64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
	}

	/// Reads an NBT string (`u16` length prefix followed by UTF-8 bytes)
	fn string(&mut self) -> Result<String> {
		let len = u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()) as usize;
		let bytes = self.bytes(len)?;
		Ok(String::from_utf8_lossy(bytes).into_owned())
	}

	/// Reads the payload of a value of the given tag type
	fn payload(&mut self, tag: u8) -> Result<Value> {
		Ok(match tag {
			1 => Value::Byte(self.u8()? as i8),
			2 => Value::Short(self.i16()?),
			3 => Value::Int(self.i32()?),
			4 => Value::Long(self.i64()?),
			5 => Value::Float(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap())),
			6 => Value::Double(f64::from_le_bytes(self.bytes(8)?.try_into().unwrap())),
			7 => {
				let len = self.i32()?;
				if len < 0 {
					bail!("Negative byte array length");
				}
				Value::ByteArray(self.bytes(len as usize)?.to_vec())
			}
			8 => Value::String(self.string()?),
			9 => {
				let elem_tag = self.u8()?;
				let len = self.i32()?;
				if len < 0 {
					bail!("Negative list length");
				}
				let mut list = Vec::with_capacity(len as usize);
				for _ in 0..len {
					// An empty list may have an element tag of 0 (End)
					if elem_tag == 0 {
						break;
					}
					list.push(self.payload(elem_tag)?);
				}
				Value::List(list)
			}
			10 => Value::Compound(self.compound()?),
			11 => {
				let len = self.i32()?;
				if len < 0 {
					bail!("Negative int array length");
				}
				let mut arr = Vec::with_capacity(len as usize);
				for _ in 0..len {
					arr.push(self.i32()?);
				}
				Value::IntArray(arr)
			}
			12 => {
				let len = self.i32()?;
				if len < 0 {
					bail!("Negative long array length");
				}
				let mut arr = Vec::with_capacity(len as usize);
				for _ in 0..len {
					arr.push(self.i64()?);
				}
				Value::LongArray(arr)
			}
			_ => bail!("Unknown NBT tag type {tag}"),
		})
	}

	/// Reads the named entries of a compound until the closing `TAG_End`
	fn compound(&mut self) -> Result<Vec<(String, Value)>> {
		let mut entries = Vec::new();
		loop {
			let tag = self.u8()?;
			if tag == 0 {
				break;
			}
			let name = self.string()?;
			let value = self.payload(tag)?;
			entries.push((name, value));
		}
		Ok(entries)
	}

	/// Reads a single root (named) NBT value
	///
	/// Returns `None` if the input is fully consumed.
	pub fn read_value(&mut self) -> Result<Option<Value>> {
		if self.at_end() {
			return Ok(None);
		}
		let tag = self.u8()?;
		if tag == 0 {
			// Trailing padding
			return Ok(None);
		}
		// Skip the (usually empty) root name
		let _name = self.string()?;
		Ok(Some(self.payload(tag)?))
	}
}

/// Reads all concatenated root NBT values from a buffer
///
/// Bedrock stores block palettes and block entity lists as several NBT
/// compounds written back to back.
pub fn read_all(data: &[u8]) -> Result<Vec<Value>> {
	let mut reader = Reader::new(data);
	let mut values = Vec::new();
	while let Some(value) = reader.read_value()? {
		values.push(value);
	}
	Ok(values)
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_read_compound() {
		// TAG_Compound (10), name "" (len 0)
		//   TAG_String (8) name "name" -> "minecraft:stone"
		//   TAG_Int (3) name "version" -> 17879555
		// TAG_End (0)
		let mut data = vec![10u8, 0, 0];
		// name string
		data.push(8);
		data.extend_from_slice(&4u16.to_le_bytes());
		data.extend_from_slice(b"name");
		data.extend_from_slice(&15u16.to_le_bytes());
		data.extend_from_slice(b"minecraft:stone");
		// version int
		data.push(3);
		data.extend_from_slice(&7u16.to_le_bytes());
		data.extend_from_slice(b"version");
		data.extend_from_slice(&17879555i32.to_le_bytes());
		data.push(0); // end

		let values = read_all(&data).unwrap();
		assert_eq!(values.len(), 1);
		assert_eq!(
			values[0].get("name").and_then(Value::as_str),
			Some("minecraft:stone")
		);
		assert_eq!(values[0].get("version"), Some(&Value::Int(17879555)));
	}
}
