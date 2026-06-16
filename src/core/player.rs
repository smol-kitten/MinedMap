//! Collection of per-player data for the `--emit-player-data` output
//!
//! Java Edition stores each player's live state (position, inventory, XP, …) as
//! a gzip-compressed NBT file `playerdata/<uuid>.dat`, the accumulated
//! statistics as JSON in `stats/<uuid>.json`, and a UUID→name cache in
//! `usercache.json` / `usernamecache.json` (which live in the server directory,
//! i.e. the world directory or its parent). This module reads all of these and
//! produces a single `players.json`.

use std::{
	collections::{BTreeMap, HashMap},
	ffi::OsStr,
	path::{Path, PathBuf},
};

use fastnbt::Value;
use rayon::prelude::*;
use serde::Serialize;
use tracing::warn;

/// A single inventory or ender-chest item
#[derive(Debug, Clone, Serialize)]
pub struct Item {
	/// Inventory slot, if present
	#[serde(skip_serializing_if = "Option::is_none")]
	pub slot: Option<i32>,
	/// Item identifier (for example `minecraft:diamond`)
	pub id: String,
	/// Item count
	pub count: i32,
}

/// Experience state of a player
#[derive(Debug, Clone, Serialize)]
pub struct Xp {
	/// Experience level
	pub level: i32,
	/// Total collected experience points
	pub total: i32,
	/// Progress towards the next level (0.0 – 1.0)
	pub progress: f32,
}

/// Accumulated statistics of a player (from `stats/<uuid>.json`)
#[derive(Debug, Clone, Default, Serialize)]
pub struct Stats {
	/// Ticks played (`minecraft:play_time`)
	pub play_time: i64,
	/// Number of deaths
	pub deaths: i64,
	/// Number of mobs killed
	pub mob_kills: i64,
	/// Number of players killed
	pub player_kills: i64,
	/// Distance walked, in centimeters
	pub walk_cm: i64,
	/// Distance sprinted, in centimeters
	pub sprint_cm: i64,
	/// Distance swum, in centimeters
	pub swim_cm: i64,
	/// Distance flown with an elytra, in centimeters
	pub fly_cm: i64,
	/// Total number of blocks mined (sum of `minecraft:mined`)
	pub blocks_mined: i64,
	/// Total number of items used (sum of `minecraft:used`); approximates the
	/// number of blocks placed, as Minecraft has no dedicated placement counter
	pub blocks_placed: i64,
}

/// Collected data of a single player
#[derive(Debug, Clone, Serialize)]
pub struct Player {
	/// Player UUID (from the `playerdata` file name)
	pub uuid: String,
	/// Player name, if it could be resolved from a name cache
	#[serde(skip_serializing_if = "Option::is_none")]
	pub name: Option<String>,
	/// Position as `[x, y, z]`
	pub pos: [f64; 3],
	/// Dimension identifier (for example `minecraft:overworld`)
	pub dimension: String,
	/// Rotation as `[yaw, pitch]`
	pub rotation: [f32; 2],
	/// Respawn position as `[x, y, z]`, if set
	#[serde(skip_serializing_if = "Option::is_none")]
	pub spawn: Option<[i32; 3]>,
	/// Experience state
	pub xp: Xp,
	/// Current health
	pub health: f32,
	/// Current food level
	pub food: i32,
	/// Accumulated statistics, if a stats file was present
	#[serde(skip_serializing_if = "Option::is_none")]
	pub stats: Option<Stats>,
	/// Causes of death (`minecraft:killed_by`), keyed by source entity id
	#[serde(skip_serializing_if = "BTreeMap::is_empty")]
	pub killed_by: BTreeMap<String, i64>,
	/// Crafted items (`minecraft:crafted`), keyed by item id
	#[serde(skip_serializing_if = "BTreeMap::is_empty")]
	pub crafted: BTreeMap<String, i64>,
	/// Main inventory contents
	pub inventory: Vec<Item>,
	/// Ender chest contents
	pub ender_items: Vec<Item>,
}

/// Returns the compound entry for a key, if `value` is a compound
fn get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
	match value {
		Value::Compound(map) => map.get(key),
		_ => None,
	}
}

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

/// Returns the string value of a tag, if it is a string
fn as_str(value: &Value) -> Option<&str> {
	match value {
		Value::String(s) => Some(s),
		_ => None,
	}
}

/// Returns the elements of a list tag, if it is a list
fn as_list(value: &Value) -> Option<&[Value]> {
	match value {
		Value::List(list) => Some(list),
		_ => None,
	}
}

/// Reads an `[f64; 3]` position-style list, defaulting missing entries to zero
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

/// Parses a single item compound, skipping empty (`minecraft:air`) slots
fn parse_item(value: &Value) -> Option<Item> {
	let id = as_str(get(value, "id")?)?.to_string();
	if id == "minecraft:air" {
		return None;
	}
	let slot = get(value, "Slot").and_then(as_i64).map(|v| v as i32);
	// Item count was a byte `Count` before 1.20.5 and an int `count` after.
	let count = get(value, "count")
		.or_else(|| get(value, "Count"))
		.and_then(as_i64)
		.unwrap_or(1) as i32;
	Some(Item { slot, id, count })
}

/// Parses an item list (inventory or ender chest)
fn parse_items(value: Option<&Value>) -> Vec<Item> {
	value
		.and_then(as_list)
		.map(|list| list.iter().filter_map(parse_item).collect())
		.unwrap_or_default()
}

/// Maps a legacy numeric dimension id to its modern identifier
fn dimension_name(value: Option<&Value>) -> String {
	match value {
		Some(Value::String(s)) => s.clone(),
		Some(other) => match as_i64(other) {
			Some(-1) => "minecraft:the_nether".to_string(),
			Some(1) => "minecraft:the_end".to_string(),
			_ => "minecraft:overworld".to_string(),
		},
		None => "minecraft:overworld".to_string(),
	}
}

/// Reads the respawn position from `SpawnX`/`SpawnY`/`SpawnZ`, if present
fn read_spawn(root: &Value) -> Option<[i32; 3]> {
	let x = get(root, "SpawnX").and_then(as_i64)?;
	let y = get(root, "SpawnY").and_then(as_i64).unwrap_or(0);
	let z = get(root, "SpawnZ").and_then(as_i64)?;
	Some([x as i32, y as i32, z as i32])
}

/// Parses a `playerdata/<uuid>.dat` file into a [Player]
fn parse_player(uuid: String, root: &Value) -> Player {
	Player {
		uuid,
		name: None,
		pos: read_f64_3(get(root, "Pos")),
		dimension: dimension_name(get(root, "Dimension")),
		rotation: read_f32_2(get(root, "Rotation")),
		spawn: read_spawn(root),
		xp: Xp {
			level: get(root, "XpLevel").and_then(as_i64).unwrap_or(0) as i32,
			total: get(root, "XpTotal").and_then(as_i64).unwrap_or(0) as i32,
			progress: get(root, "XpP").and_then(as_f64).unwrap_or(0.0) as f32,
		},
		health: get(root, "Health").and_then(as_f64).unwrap_or(0.0) as f32,
		food: get(root, "foodLevel").and_then(as_i64).unwrap_or(0) as i32,
		stats: None,
		killed_by: BTreeMap::new(),
		crafted: BTreeMap::new(),
		inventory: parse_items(get(root, "Inventory")),
		ender_items: parse_items(get(root, "EnderItems")),
	}
}

/// Returns whether a file name matches `<uuid>.dat`
fn dat_uuid(file_name: &OsStr) -> Option<String> {
	let name = file_name.to_str()?;
	let stem = name.strip_suffix(".dat")?;
	// A UUID is 36 characters with hyphens; accept anything non-empty to be
	// lenient towards offline-mode / unusual UUIDs.
	if stem.is_empty() {
		return None;
	}
	Some(stem.to_string())
}

/// Sums the integer values of a JSON object (a stats category)
fn sum_category(stats: &serde_json::Value, category: &str) -> i64 {
	stats
		.get(category)
		.and_then(|v| v.as_object())
		.map(|obj| obj.values().filter_map(serde_json::Value::as_i64).sum())
		.unwrap_or(0)
}

/// Reads a single custom stat value
fn custom_stat(custom: &serde_json::Value, key: &str) -> i64 {
	custom
		.get(key)
		.and_then(serde_json::Value::as_i64)
		.unwrap_or(0)
}

/// Converts a JSON stats object into a [BTreeMap]
fn stat_map(stats: &serde_json::Value, category: &str) -> BTreeMap<String, i64> {
	stats
		.get(category)
		.and_then(|v| v.as_object())
		.map(|obj| {
			obj.iter()
				.filter_map(|(k, v)| v.as_i64().map(|n| (k.clone(), n)))
				.collect()
		})
		.unwrap_or_default()
}

/// Reads and applies `stats/<uuid>.json` to a player, if the file exists
fn apply_stats(player: &mut Player, stats_dir: &Path) {
	let path = stats_dir.join(format!("{}.json", player.uuid));
	let data = match std::fs::read(&path) {
		Ok(data) => data,
		Err(_) => return,
	};
	let root: serde_json::Value = match serde_json::from_slice(&data) {
		Ok(root) => root,
		Err(err) => {
			warn!("Failed to parse stats file {}: {}", path.display(), err);
			return;
		}
	};
	let stats = root.get("stats").unwrap_or(&serde_json::Value::Null);
	let custom = stats
		.get("minecraft:custom")
		.cloned()
		.unwrap_or(serde_json::Value::Null);

	player.stats = Some(Stats {
		// `play_time` replaced the older `play_one_minute` key in 1.17.
		play_time: custom_stat(&custom, "minecraft:play_time")
			.max(custom_stat(&custom, "minecraft:play_one_minute")),
		deaths: custom_stat(&custom, "minecraft:deaths"),
		mob_kills: custom_stat(&custom, "minecraft:mob_kills"),
		player_kills: custom_stat(&custom, "minecraft:player_kills"),
		walk_cm: custom_stat(&custom, "minecraft:walk_one_cm"),
		sprint_cm: custom_stat(&custom, "minecraft:sprint_one_cm"),
		swim_cm: custom_stat(&custom, "minecraft:swim_one_cm"),
		fly_cm: custom_stat(&custom, "minecraft:aviate_one_cm"),
		blocks_mined: sum_category(stats, "minecraft:mined"),
		blocks_placed: sum_category(stats, "minecraft:used"),
	});
	player.killed_by = stat_map(stats, "minecraft:killed_by");
	player.crafted = stat_map(stats, "minecraft:crafted");
}

/// A UUID→name cache built from `usercache.json` / `usernamecache.json`
type NameCache = HashMap<String, String>;

/// Single entry of a vanilla `usercache.json`
#[derive(serde::Deserialize)]
struct UserCacheEntry {
	/// Player name
	name: String,
	/// Player UUID
	uuid: String,
}

/// Loads name caches, searching the input directory and its parent
///
/// On a dedicated server the world directory is a subdirectory of the server
/// directory, and `usercache.json` / `usernamecache.json` live next to the
/// world directory rather than inside it.
fn load_name_cache(input_dir: &Path) -> NameCache {
	let mut cache = NameCache::new();
	let mut dirs = vec![input_dir.to_path_buf()];
	if let Some(parent) = input_dir.parent() {
		dirs.push(parent.to_path_buf());
	}

	for dir in dirs {
		// Forge/NeoForge `usernamecache.json`: a plain {uuid: name} object.
		if let Ok(data) = std::fs::read(dir.join("usernamecache.json"))
			&& let Ok(map) = serde_json::from_slice::<HashMap<String, String>>(&data)
		{
			for (uuid, name) in map {
				cache.entry(uuid.to_lowercase()).or_insert(name);
			}
		}
		// Vanilla `usercache.json`: an array of {name, uuid, expiresOn}.
		if let Ok(data) = std::fs::read(dir.join("usercache.json"))
			&& let Ok(entries) = serde_json::from_slice::<Vec<UserCacheEntry>>(&data)
		{
			for entry in entries {
				cache.entry(entry.uuid.to_lowercase()).or_insert(entry.name);
			}
		}
	}

	cache
}

/// Collects the player data of a Java Edition world
pub fn collect(input_dir: &Path) -> Vec<Player> {
	let player_dir: PathBuf = [input_dir, Path::new("playerdata")].iter().collect();
	let stats_dir: PathBuf = [input_dir, Path::new("stats")].iter().collect();

	let mut files = Vec::new();
	if let Ok(dir) = player_dir.read_dir() {
		for entry in dir.filter_map(Result::ok) {
			if entry.file_type().map(|t| t.is_file()).unwrap_or(false)
				&& let Some(uuid) = dat_uuid(&entry.file_name())
			{
				files.push((uuid, entry.path()));
			}
		}
	}

	let name_cache = load_name_cache(input_dir);

	let mut players: Vec<Player> = files
		.par_iter()
		.filter_map(|(uuid, path)| {
			let root: Value = match crate::nbt::data::from_file(path) {
				Ok(root) => root,
				Err(err) => {
					warn!("Failed to read player file {}: {:?}", path.display(), err);
					return None;
				}
			};
			let mut player = parse_player(uuid.clone(), &root);
			player.name = name_cache.get(&uuid.to_lowercase()).cloned();
			apply_stats(&mut player, &stats_dir);
			Some(player)
		})
		.collect();

	players.sort_by(|a, b| a.uuid.cmp(&b.uuid));
	players
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_parse_player() {
		let value = fastnbt::nbt!({
			"Pos": [10.5f64, 64.0f64, -20.5f64],
			"Rotation": [90.0f32, 0.0f32],
			"Dimension": "minecraft:the_nether",
			"SpawnX": 100i32,
			"SpawnY": 70i32,
			"SpawnZ": -50i32,
			"XpLevel": 30i32,
			"XpTotal": 1395i32,
			"XpP": 0.5f32,
			"Health": 18.0f32,
			"foodLevel": 17i32,
			"Inventory": [
				{ "Slot": 0i8, "id": "minecraft:diamond_sword", "count": 1i32 },
				{ "Slot": 1i8, "id": "minecraft:air", "count": 0i32 },
				{ "Slot": 2i8, "id": "minecraft:cobblestone", "count": 64i32 },
			],
			"EnderItems": [
				{ "Slot": 0i8, "id": "minecraft:gold_ingot", "count": 5i32 },
			],
		});

		let player = parse_player("069a79f4-44e9-4726-a5be-fc90d2a28159".to_string(), &value);
		assert_eq!(player.pos, [10.5, 64.0, -20.5]);
		assert_eq!(player.rotation, [90.0, 0.0]);
		assert_eq!(player.dimension, "minecraft:the_nether");
		assert_eq!(player.spawn, Some([100, 70, -50]));
		assert_eq!(player.xp.level, 30);
		assert_eq!(player.xp.total, 1395);
		assert_eq!(player.health, 18.0);
		assert_eq!(player.food, 17);
		// minecraft:air slot must be filtered out
		assert_eq!(player.inventory.len(), 2);
		assert_eq!(player.inventory[0].id, "minecraft:diamond_sword");
		assert_eq!(player.inventory[1].id, "minecraft:cobblestone");
		assert_eq!(player.inventory[1].count, 64);
		assert_eq!(player.ender_items.len(), 1);
		assert_eq!(player.ender_items[0].id, "minecraft:gold_ingot");
	}

	#[test]
	fn test_legacy_count_and_dimension() {
		// Pre-1.20.5 byte `Count` and legacy numeric dimension
		let value = fastnbt::nbt!({
			"Pos": [0.0f64, 0.0f64, 0.0f64],
			"Dimension": -1i32,
			"Inventory": [
				{ "Slot": 0i8, "id": "minecraft:stone", "Count": 32i8 },
			],
		});
		let player = parse_player("abc".to_string(), &value);
		assert_eq!(player.dimension, "minecraft:the_nether");
		assert_eq!(player.inventory[0].count, 32);
	}

	#[test]
	fn test_collect_end_to_end() {
		use std::io::Write;

		let uuid = "069a79f4-44e9-4726-a5be-fc90d2a28159";
		let base = std::env::temp_dir().join(format!(
			"minedmap-player-test-{}-{:?}",
			std::process::id(),
			std::thread::current().id()
		));
		let world = base.join("world");
		std::fs::create_dir_all(world.join("playerdata")).unwrap();
		std::fs::create_dir_all(world.join("stats")).unwrap();

		// gzip-compressed player NBT
		let value = fastnbt::nbt!({
			"Pos": [1.0f64, 2.0f64, 3.0f64],
			"Dimension": "minecraft:overworld",
			"Health": 20.0f32,
			"foodLevel": 20i32,
			"Inventory": [
				{ "Slot": 0i8, "id": "minecraft:apple", "count": 3i32 },
			],
		});
		let nbt = fastnbt::to_bytes(&value).unwrap();
		let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
		encoder.write_all(&nbt).unwrap();
		let gz = encoder.finish().unwrap();
		std::fs::write(world.join(format!("playerdata/{uuid}.dat")), gz).unwrap();

		// stats file
		let stats = serde_json::json!({
			"stats": { "minecraft:custom": { "minecraft:deaths": 7 } },
		});
		std::fs::write(
			world.join(format!("stats/{uuid}.json")),
			serde_json::to_vec(&stats).unwrap(),
		)
		.unwrap();

		// usercache.json lives in the server dir (the parent of the world dir)
		let usercache = serde_json::json!([
			{ "name": "Steve", "uuid": uuid, "expiresOn": "2099-01-01 00:00:00 +0000" },
		]);
		std::fs::write(
			base.join("usercache.json"),
			serde_json::to_vec(&usercache).unwrap(),
		)
		.unwrap();

		let players = collect(&world);
		assert_eq!(players.len(), 1);
		let player = &players[0];
		assert_eq!(player.uuid, uuid);
		assert_eq!(player.name.as_deref(), Some("Steve"));
		assert_eq!(player.pos, [1.0, 2.0, 3.0]);
		assert_eq!(player.inventory.len(), 1);
		assert_eq!(player.inventory[0].id, "minecraft:apple");
		assert_eq!(player.stats.as_ref().unwrap().deaths, 7);

		let _ = std::fs::remove_dir_all(&base);
	}

	#[test]
	fn test_stats_parsing() {
		let stats = serde_json::json!({
			"stats": {
				"minecraft:custom": {
					"minecraft:play_time": 12000,
					"minecraft:deaths": 3,
					"minecraft:mob_kills": 42,
					"minecraft:walk_one_cm": 150000,
				},
				"minecraft:mined": {
					"minecraft:stone": 100,
					"minecraft:dirt": 50,
				},
				"minecraft:used": {
					"minecraft:cobblestone": 30,
				},
				"minecraft:killed_by": {
					"minecraft:creeper": 2,
				},
				"minecraft:crafted": {
					"minecraft:torch": 64,
				},
			},
		});

		let dir = std::env::temp_dir().join(format!("minedmap-stats-test-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let uuid = "test-uuid";
		std::fs::write(
			dir.join(format!("{uuid}.json")),
			serde_json::to_vec(&stats).unwrap(),
		)
		.unwrap();

		let mut player = Player {
			uuid: uuid.to_string(),
			name: None,
			pos: [0.0; 3],
			dimension: "minecraft:overworld".to_string(),
			rotation: [0.0; 2],
			spawn: None,
			xp: Xp {
				level: 0,
				total: 0,
				progress: 0.0,
			},
			health: 0.0,
			food: 0,
			stats: None,
			killed_by: BTreeMap::new(),
			crafted: BTreeMap::new(),
			inventory: Vec::new(),
			ender_items: Vec::new(),
		};
		apply_stats(&mut player, &dir);

		let s = player.stats.expect("stats should be present");
		assert_eq!(s.play_time, 12000);
		assert_eq!(s.deaths, 3);
		assert_eq!(s.mob_kills, 42);
		assert_eq!(s.walk_cm, 150000);
		assert_eq!(s.blocks_mined, 150);
		assert_eq!(s.blocks_placed, 30);
		assert_eq!(player.killed_by.get("minecraft:creeper"), Some(&2));
		assert_eq!(player.crafted.get("minecraft:torch"), Some(&64));

		let _ = std::fs::remove_dir_all(&dir);
	}
}
