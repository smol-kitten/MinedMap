//! Collection of Bedrock Edition player data for `--emit-player-data`
//!
//! Bedrock stores each player as a little-endian NBT value in the LevelDB
//! database: the single-player local player under the `~local_player` key and
//! multiplayer players under `player_server_<id>` keys. The data is mapped into
//! the same [Player] shape as Java Edition so the `players.json` schema is
//! identical, although Bedrock has no player statistics or cached names, and
//! stores health and food as entity attributes rather than top-level fields.

use std::collections::BTreeMap;

use tracing::warn;

use super::db::BedrockDb;
use super::nbt::{self, Value};
use crate::core::player::{Item, Player, Xp};

/// LevelDB key of the single-player local player
const LOCAL_PLAYER_KEY: &[u8] = b"~local_player";
/// LevelDB key prefix of server (multiplayer) players
const SERVER_PLAYER_PREFIX: &[u8] = b"player_server_";

/// Interprets any NBT integer tag as an `i64`
fn as_i64(value: &Value) -> Option<i64> {
	Some(match value {
		Value::Byte(v) => *v as i64,
		Value::Short(v) => *v as i64,
		Value::Int(v) => *v as i64,
		Value::Long(v) => *v,
		_ => return None,
	})
}

/// Interprets any NBT number tag as an `f64`
fn as_f64(value: &Value) -> Option<f64> {
	Some(match value {
		Value::Float(v) => *v as f64,
		Value::Double(v) => *v,
		other => as_i64(other)? as f64,
	})
}

/// Returns the elements of a list tag, if it is a list
fn as_list(value: &Value) -> Option<&[Value]> {
	match value {
		Value::List(list) => Some(list),
		_ => None,
	}
}

/// Reads an `[f64; 3]` position list, defaulting missing entries to zero
fn read_f64_3(value: Option<&Value>) -> [f64; 3] {
	let mut out = [0.0; 3];
	if let Some(list) = value.and_then(as_list) {
		for (i, slot) in out.iter_mut().enumerate() {
			if let Some(v) = list.get(i).and_then(as_f64) {
				*slot = v;
			}
		}
	}
	out
}

/// Reads an `[f32; 2]` rotation list, defaulting missing entries to zero
fn read_f32_2(value: Option<&Value>) -> [f32; 2] {
	let mut out = [0.0; 2];
	if let Some(list) = value.and_then(as_list) {
		for (i, slot) in out.iter_mut().enumerate() {
			if let Some(v) = list.get(i).and_then(as_f64) {
				*slot = v as f32;
			}
		}
	}
	out
}

/// Maps a Bedrock numeric dimension id to its identifier
fn dimension_name(value: Option<&Value>) -> String {
	match value.and_then(as_i64) {
		Some(1) => "minecraft:the_nether",
		Some(2) => "minecraft:the_end",
		_ => "minecraft:overworld",
	}
	.to_string()
}

/// Reads the respawn position from `SpawnX`/`SpawnY`/`SpawnZ`, if present
fn read_spawn(root: &Value) -> Option<[i32; 3]> {
	let x = root.get("SpawnX").and_then(as_i64)?;
	let y = root.get("SpawnY").and_then(as_i64).unwrap_or(0);
	let z = root.get("SpawnZ").and_then(as_i64)?;
	Some([x as i32, y as i32, z as i32])
}

/// Reads a named entity attribute's `Current` value from the `Attributes` list
fn attribute(root: &Value, name: &str) -> Option<f64> {
	let list = as_list(root.get("Attributes")?)?;
	list.iter()
		.find(|entry| entry.get("Name").and_then(Value::as_str) == Some(name))
		.and_then(|entry| entry.get("Current"))
		.and_then(as_f64)
}

/// Parses a single Bedrock item compound, skipping empty slots
fn parse_item(value: &Value) -> Option<Item> {
	let id = value.get("Name").and_then(Value::as_str)?.to_string();
	if id.is_empty() || id == "minecraft:air" {
		return None;
	}
	let slot = value.get("Slot").and_then(as_i64).map(|v| v as i32);
	let count = value.get("Count").and_then(as_i64).unwrap_or(1) as i32;
	Some(Item { slot, id, count })
}

/// Parses an item list (inventory or ender chest)
fn parse_items(value: Option<&Value>) -> Vec<Item> {
	value
		.and_then(as_list)
		.map(|list| list.iter().filter_map(parse_item).collect())
		.unwrap_or_default()
}

/// Builds a player UUID/identifier from its LevelDB key
fn player_uuid(key: &[u8]) -> String {
	if key == LOCAL_PLAYER_KEY {
		return "~local_player".to_string();
	}
	String::from_utf8_lossy(&key[SERVER_PLAYER_PREFIX.len()..]).into_owned()
}

/// Maps a parsed Bedrock player NBT value into a [Player]
fn parse_player(uuid: String, root: &Value) -> Player {
	Player {
		uuid,
		name: None,
		pos: read_f64_3(root.get("Pos")),
		dimension: dimension_name(root.get("DimensionId")),
		rotation: read_f32_2(root.get("Rotation")),
		spawn: read_spawn(root),
		xp: Xp {
			level: root.get("PlayerLevel").and_then(as_i64).unwrap_or(0) as i32,
			// Bedrock does not store the cumulative XP total.
			total: 0,
			progress: root
				.get("PlayerLevelProgress")
				.and_then(as_f64)
				.unwrap_or(0.0) as f32,
		},
		health: attribute(root, "minecraft:health").unwrap_or(0.0) as f32,
		food: attribute(root, "minecraft:player.hunger").unwrap_or(0.0) as i32,
		// Bedrock has no Java-style statistics.
		stats: None,
		killed_by: BTreeMap::new(),
		crafted: BTreeMap::new(),
		inventory: parse_items(root.get("Inventory")),
		ender_items: parse_items(root.get("EnderChestInventory")),
	}
}

/// Collects the player data of a Bedrock Edition world from its LevelDB database
pub fn collect(db: &mut BedrockDb) -> Vec<Player> {
	let mut keys = Vec::new();
	if let Err(err) = db.for_each_key(|key| {
		if key == LOCAL_PLAYER_KEY || key.starts_with(SERVER_PLAYER_PREFIX) {
			keys.push(key.to_vec());
		}
	}) {
		warn!("Failed to scan LevelDB keys for players: {err:?}");
	}

	let mut players = Vec::new();
	for key in keys {
		let Some(data) = db.get(&key) else {
			continue;
		};
		let value = match nbt::read_all(&data) {
			Ok(mut values) if !values.is_empty() => values.remove(0),
			Ok(_) => continue,
			Err(err) => {
				warn!(
					"Failed to parse player data for {:?}: {:?}",
					player_uuid(&key),
					err
				);
				continue;
			}
		};
		players.push(parse_player(player_uuid(&key), &value));
	}

	players.sort_by(|a, b| a.uuid.cmp(&b.uuid));
	players
}

#[cfg(test)]
mod test {
	use super::*;

	/// Builds an NBT compound from name/value pairs
	fn compound(entries: Vec<(&str, Value)>) -> Value {
		Value::Compound(
			entries
				.into_iter()
				.map(|(k, v)| (k.to_string(), v))
				.collect(),
		)
	}

	#[test]
	fn test_parse_player() {
		let root = compound(vec![
			(
				"Pos",
				Value::List(vec![
					Value::Float(10.5),
					Value::Float(64.0),
					Value::Float(-20.5),
				]),
			),
			(
				"Rotation",
				Value::List(vec![Value::Float(90.0), Value::Float(0.0)]),
			),
			("DimensionId", Value::Int(1)),
			("SpawnX", Value::Int(100)),
			("SpawnY", Value::Int(70)),
			("SpawnZ", Value::Int(-50)),
			("PlayerLevel", Value::Int(30)),
			("PlayerLevelProgress", Value::Float(0.5)),
			(
				"Attributes",
				Value::List(vec![
					compound(vec![
						("Name", Value::String("minecraft:health".to_string())),
						("Current", Value::Float(18.0)),
					]),
					compound(vec![
						("Name", Value::String("minecraft:player.hunger".to_string())),
						("Current", Value::Float(17.0)),
					]),
				]),
			),
			(
				"Inventory",
				Value::List(vec![
					compound(vec![
						("Slot", Value::Byte(0)),
						("Name", Value::String("minecraft:diamond_sword".to_string())),
						("Count", Value::Byte(1)),
					]),
					compound(vec![
						("Slot", Value::Byte(1)),
						("Name", Value::String("minecraft:air".to_string())),
						("Count", Value::Byte(0)),
					]),
				]),
			),
		]);

		let player = parse_player("local".to_string(), &root);
		assert_eq!(player.pos, [10.5, 64.0, -20.5]);
		assert_eq!(player.rotation, [90.0, 0.0]);
		assert_eq!(player.dimension, "minecraft:the_nether");
		assert_eq!(player.spawn, Some([100, 70, -50]));
		assert_eq!(player.xp.level, 30);
		assert_eq!(player.health, 18.0);
		assert_eq!(player.food, 17);
		// minecraft:air must be filtered out
		assert_eq!(player.inventory.len(), 1);
		assert_eq!(player.inventory[0].id, "minecraft:diamond_sword");
		assert!(player.stats.is_none());
	}

	#[test]
	fn test_player_uuid() {
		assert_eq!(player_uuid(b"~local_player"), "~local_player");
		assert_eq!(player_uuid(b"player_server_abc-123"), "abc-123");
	}
}
