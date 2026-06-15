//! Translation of Bedrock Edition block names to Java Edition block identifiers
//!
//! Most blocks share the same identifier between editions (since the "flattening"
//! both editions use `minecraft:<name>`), so identity mapping handles the bulk of
//! cases. This table covers the common surface blocks whose names differ; unknown
//! blocks fall back to a neutral color in the renderer.

use minedmap_resource::{BlockColor, BlockType, BlockTypes, UnknownBlockMode};

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
/// Returns `None` for unknown blocks when [UnknownBlockMode::Hide] is in effect,
/// and `true` for the second tuple element if the block could not be mapped to a
/// known Java block type.
pub fn block_color(
	name: &str,
	block_types: &BlockTypes,
	unknown: UnknownBlockMode,
) -> (Option<BlockColor>, bool) {
	let java_name = translate_block_name(name);
	match block_types.get(java_name) {
		Some(block_type) => {
			let _: &BlockType = block_type;
			(Some(block_type.block_color), false)
		}
		None => (BlockColor::unknown(name, unknown), true),
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
		let (color, unknown) = block_color("minecraft:stone", &block_types, UnknownBlockMode::Gray);
		assert!(!unknown);
		assert!(color.is_some());

		// Unknown blocks are hidden by default, visible in gray/color modes
		let (hidden, unknown) = block_color(
			"minecraft:made_up_block",
			&block_types,
			UnknownBlockMode::Hide,
		);
		assert!(unknown);
		assert!(hidden.is_none());

		let (gray, _) = block_color(
			"minecraft:made_up_block",
			&block_types,
			UnknownBlockMode::Gray,
		);
		assert_eq!(gray.unwrap().color, BlockColor::NEUTRAL.color);
		assert!(gray.unwrap().is(minedmap_resource::BlockFlag::Opaque));
	}
}
