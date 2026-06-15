//! Translation of Bedrock Edition block names to Java Edition block identifiers
//!
//! Most blocks share the same identifier between editions (since the "flattening"
//! both editions use `minecraft:<name>`), so identity mapping handles the bulk of
//! cases. This table covers the common surface blocks whose names differ; unknown
//! blocks fall back to a neutral color in the renderer.

use minedmap_resource::{BlockColor, BlockType, BlockTypes};

/// Translates a Bedrock block name to the equivalent Java Edition identifier
///
/// The `minecraft:` namespace prefix is stripped; the returned identifier can be
/// looked up directly in [BlockTypes].
pub fn translate_block_name(name: &str) -> &str {
	let name = name.strip_prefix("minecraft:").unwrap_or(name);
	match name {
		// Bedrock keeps the legacy "grass" name for the grass block
		"grass" => "grass_block",
		// Snow: Bedrock and Java swap the "snow" / "snow_block" meanings
		"snow_layer" => "snow",
		"snow" => "snow_block",
		// Fluids
		"flowing_water" => "water",
		"flowing_lava" => "lava",
		// Renamed blocks
		"grass_path" => "dirt_path",
		"hardened_clay" => "terracotta",
		"stained_hardened_clay" => "terracotta",
		"podzol" => "podzol",
		// Legacy combined log / leaf blocks (best effort: default variant)
		"log" => "oak_log",
		"log2" => "acacia_log",
		"wood" => "oak_wood",
		"leaves" => "oak_leaves",
		"leaves2" => "acacia_leaves",
		// Ground cover (non-opaque, but mapped for completeness)
		"tallgrass" => "short_grass",
		"yellow_flower" => "dandelion",
		other => other,
	}
}

/// Resolves the [BlockColor] for a Bedrock block name
///
/// Returns the neutral fallback color and `true` for the unknown flag if the
/// block could not be mapped to a known Java block type.
pub fn block_color(name: &str, block_types: &BlockTypes) -> (BlockColor, bool) {
	let java_name = translate_block_name(name);
	match block_types.get(java_name) {
		Some(block_type) => {
			let _: &BlockType = block_type;
			(block_type.block_color, false)
		}
		None => (BlockColor::NEUTRAL, true),
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_translate() {
		assert_eq!(translate_block_name("minecraft:stone"), "stone");
		assert_eq!(translate_block_name("minecraft:grass"), "grass_block");
		assert_eq!(translate_block_name("minecraft:snow_layer"), "snow");
		assert_eq!(translate_block_name("snow"), "snow_block");
		assert_eq!(translate_block_name("minecraft:dirt"), "dirt");
	}

	#[test]
	fn test_block_color_known_and_unknown() {
		let block_types = BlockTypes::default();
		let (_, unknown) = block_color("minecraft:stone", &block_types);
		assert!(!unknown);
		let (color, unknown) = block_color("minecraft:totally_made_up_block", &block_types);
		assert!(unknown);
		assert_eq!(color.color, BlockColor::NEUTRAL.color);
		assert!(color.is(minedmap_resource::BlockFlag::Opaque));
	}
}
