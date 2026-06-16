//! Core functions of the MinedMap CLI

mod bedrock;
mod common;
mod entity_collector;
mod flat;
mod heightmap;
mod java_random;
mod metadata_writer;
mod mob;
mod overlay;
mod player;
mod poi;
mod region_group;
mod region_processor;
mod texture;
mod tile_collector;
mod tile_merger;
mod tile_mipmapper;
mod tile_renderer;

use std::{
	path::PathBuf,
	sync::mpsc::{self, Receiver},
	thread,
	time::Duration,
};

use anyhow::{Context, Result};
use clap::Parser;
use git_version::git_version;

use common::{Config, Edition, ImageFormat, TileCoordMap, UnknownBlocks};
use metadata_writer::MetadataWriter;
use notify::{RecommendedWatcher, RecursiveMode, Watcher as _};
use rayon::ThreadPool;
use region_processor::RegionProcessor;
use tile_mipmapper::TileMipmapper;
use tile_renderer::TileRenderer;
use tokio::runtime::Runtime;
use tracing::{info, warn};

use self::entity_collector::EntityCollector;

/// Returns the MinedMap version number
fn version() -> &'static str {
	option_env!("MINEDMAP_VERSION").unwrap_or(
		git_version!(
			args = ["--abbrev=7", "--match=v*", "--dirty=-modified"],
			cargo_prefix = "v",
		)
		.strip_prefix("v")
		.unwrap(),
	)
}

/// Command line arguments for minedmap CLI
#[derive(Debug, Parser)]
#[command(
	about,
	version = version(),
	max_term_width = 100,
)]
pub struct Args {
	/// Number of parallel threads to use for processing
	///
	/// If not given, only a single thread is used. Pass 0 to
	/// use one thread per logical CPU core.
	#[arg(short, long)]
	pub jobs: Option<usize>,
	/// Number of parallel threads to use for initial processing
	///
	/// Passing this option only makes sense with --watch. The first run after
	/// starting MinedMap will use as many parallel jobs as configured using
	/// --job-initial, while subsequent regenerations of tiles will use the
	/// the number configured using --jobs.
	///
	/// If not given, the value from the --jobs option is used.
	#[arg(long)]
	pub jobs_initial: Option<usize>,
	/// Enable verbose messages
	#[arg(short, long)]
	pub verbose: bool,
	/// Watch for file changes and regenerate tiles automatically instead of
	/// exiting after generation
	#[arg(long)]
	pub watch: bool,
	/// Minimum delay between map generation cycles in watch mode
	#[arg(long, value_parser = humantime::parse_duration, default_value = "30s")]
	pub watch_delay: Duration,
	/// Format of generated map tiles
	#[arg(long, value_enum, default_value_t)]
	pub image_format: ImageFormat,
	/// Edition of the input Minecraft save data
	///
	/// In `auto` mode (the default), Bedrock Edition is detected by the
	/// presence of a `db/CURRENT` file in the input directory; otherwise the
	/// input is treated as Java Edition.
	#[arg(long, value_enum, default_value_t)]
	pub edition: Edition,
	/// Also render the nether dimension
	///
	/// Nether tiles are written under a `nether/` subdirectory and selectable in
	/// the viewer with the dimension switcher.
	#[arg(long)]
	pub nether: bool,
	/// Also render the end dimension
	///
	/// End tiles are written under an `end/` subdirectory and selectable in the
	/// viewer with the dimension switcher.
	#[arg(long)]
	pub end: bool,
	/// Emit derived per-chunk and marker data to the given directory
	///
	/// Writes a consolidated set of JSON files into the directory for consumption
	/// by downstream tools: `inhabited_heatmap.json`, `block_features.json`,
	/// `structures.json`, `pois.json` and `mobs.json` (the latter three are
	/// dimension-keyed; Bedrock Edition emits only the first three). All files
	/// are written atomically and their absolute paths are printed to stdout.
	/// Does not affect the generated map tiles.
	#[arg(long, value_name = "DIR")]
	pub emit_overlays: Option<PathBuf>,
	/// Emit per-player data to the given directory (Java Edition)
	///
	/// Writes `players.json` into the directory, containing each player's position,
	/// dimension, rotation, respawn point, XP, health, food, inventory, ender
	/// chest, and (from `stats/<uuid>.json`) accumulated statistics. Player names
	/// are resolved from `usercache.json` / `usernamecache.json` in the input
	/// directory or its parent. The file is written atomically and its absolute
	/// path is printed to stdout.
	#[arg(long, value_name = "DIR")]
	pub emit_player_data: Option<PathBuf>,
	/// Generate viewer overlay layers from the per-chunk overlay data
	///
	/// Writes the overlay data into the output directory and exposes it in the
	/// viewer as toggleable layers (inhabited-time heatmap, built-up areas,
	/// rails, farmland and portals).
	#[arg(long)]
	pub overlay_layers: bool,
	/// Generate an additional topographic (height) map layer
	///
	/// Renders a `height` tile layer that shades the map by terrain elevation,
	/// selectable in the viewer. Does not affect the regular map tiles.
	#[arg(long)]
	pub height_layer: bool,
	/// Generate an additional biome/climate map layer
	///
	/// Renders a `biome` tile layer coloring the map by biome, selectable in the
	/// viewer. Does not affect the regular map tiles.
	#[arg(long)]
	pub biome_layer: bool,
	/// Generate an additional cave/underground map layer
	///
	/// Renders a `cave` tile layer showing the floor of the topmost cave under
	/// the surface in each column, selectable in the viewer. Does not affect the
	/// regular map tiles.
	#[arg(long)]
	pub cave_layer: bool,
	/// Generate a mob-spawn (spawn-proofing) map layer (Java Edition)
	///
	/// Highlights surface blocks where hostile mobs can spawn at night (block
	/// light level 0 on a solid surface), selectable in the viewer. Does not
	/// affect the regular map tiles.
	#[arg(long)]
	pub mob_spawn: bool,
	/// Generate a contour (elevation lines) map layer
	///
	/// Renders a `contour` tile layer drawing elevation isolines every 8 blocks,
	/// selectable in the viewer. Does not affect the regular map tiles.
	#[arg(long)]
	pub contour_layer: bool,
	/// Generate viewer overlays for generated structure bounding boxes (Java Edition)
	///
	/// Reads each chunk's structure data and writes the bounding boxes of
	/// generated structures (villages, fortresses, monuments, …) as rectangles
	/// shown in the viewer.
	#[arg(long)]
	pub structures: bool,
	/// Generate viewer marker layers for points of interest (Java Edition)
	///
	/// Reads the world's POI data (village meeting points, villager beds and job
	/// sites, nether portals, lodestones) and writes markers shown in the viewer.
	#[arg(long)]
	pub poi_markers: bool,
	/// Generate viewer marker layers for mobs (Java Edition)
	///
	/// Reads the world's entity data and writes markers for hostile and passive
	/// mobs, shown in the viewer.
	#[arg(long)]
	pub mob_markers: bool,
	/// Generate a high-resolution textured map layer from a resource pack
	///
	/// Samples the top-face block textures from the given resource pack
	/// directory to render a detailed `textured` layer, selectable in the
	/// viewer. No textures are bundled with MinedMap; point this at a Minecraft
	/// resource pack you have the rights to use.
	#[arg(long, value_name = "DIR")]
	pub block_textures: Option<PathBuf>,
	/// Per-block texture size in pixels for the textured layer
	#[arg(long, value_name = "PIXELS", default_value_t = 8, value_parser = clap::value_parser!(u32).range(1..=16))]
	pub texture_scale: u32,
	/// How to render unrecognized (for example modded) blocks
	///
	/// By default unknown blocks are hidden (rendered as transparent). Use
	/// `gray` or `color` to make them visible, which is useful for modded worlds.
	#[arg(long, value_enum, default_value_t)]
	pub unknown_blocks: UnknownBlocks,
	/// Prefix for text of signs to show on the map
	#[arg(long)]
	pub sign_prefix: Vec<String>,
	/// Regular expression for text of signs to show on the map
	///
	/// --sign-prefix and --sign-filter allow to filter for signs to display;
	/// by default, none are visible. The options may be passed multiple times,
	/// and a sign will be visible if it matches any pattern.
	///
	/// To make all signs visible, pass an empty string to either option.
	#[arg(long)]
	pub sign_filter: Vec<String>,
	/// Regular expression replacement pattern for sign texts
	///
	/// Accepts patterns of the form 's/regexp/replacement/'. Transforms
	/// are applied to each line of sign texts separately.
	#[arg(long)]
	pub sign_transform: Vec<String>,
	/// Minecraft save directory
	pub input_dir: PathBuf,
	/// MinedMap data directory
	pub output_dir: PathBuf,
}

/// Configures a Rayon thread pool for parallel processing
fn setup_threads(num_threads: usize) -> Result<ThreadPool> {
	rayon::ThreadPoolBuilder::new()
		.num_threads(num_threads)
		.build()
		.context("Failed to configure thread pool")
}

/// Runs all MinedMap generation steps, updating all tiles as needed
fn generate(args: &Args, rt: &Runtime) -> Result<()> {
	let config = Config::new(args)?;

	match config.edition {
		// Config::new resolves Auto to a concrete edition.
		Edition::Bedrock => bedrock::generate(&config, rt),
		Edition::Java | Edition::Auto => generate_java(&config, rt),
	}
}

/// Runs all MinedMap generation steps for a Java Edition world
fn generate_java(config: &Config, rt: &Runtime) -> Result<()> {
	use overlay::Dimension;
	use std::collections::BTreeMap;

	let mut combined_overlays = overlay::OverlayData::default();
	let mut dimension_tiles: Vec<(Dimension, Vec<TileCoordMap>)> = Vec::new();
	let mut pois: BTreeMap<&'static str, poi::PoiData> = BTreeMap::new();
	let mut mobs: BTreeMap<&'static str, mob::MobData> = BTreeMap::new();

	for (dimension, region_dir) in config.dimensions_to_render() {
		let dim_config = config.for_dimension(dimension, region_dir);

		let (regions, overlays) = RegionProcessor::new(&dim_config).run()?;
		TileRenderer::new(&dim_config, rt, &regions).run()?;
		let tiles = TileMipmapper::new(&dim_config, &regions).run()?;
		// Signs are collected per dimension (the metadata writer reads each
		// dimension's processed entity data).
		EntityCollector::new(&dim_config, &regions).run()?;

		// The region processor accumulates overlay data in the overworld field;
		// route it into the dimension that was actually processed.
		let dim_overlay = overlays.overworld;
		*combined_overlays.dimension_mut(dimension) = dim_overlay;

		if config.collect_pois() {
			pois.insert(dimension.key(), poi::collect(&dim_config));
		}
		if config.collect_mobs() {
			mobs.insert(dimension.key(), mob::collect(&dim_config));
		}

		dimension_tiles.push((dimension, tiles));
	}

	MetadataWriter::new(config, &dimension_tiles).run()?;

	let structures = structures_by_dimension(&combined_overlays);

	// Viewer marker/overlay files at the output root, written only when their
	// layers are enabled, so the viewer behavior is unchanged.
	if config.structures {
		write_json(&config.viewer_structures_path, &structures)?;
	}
	if config.poi_markers {
		write_json(&config.viewer_pois_path, &pois)?;
	}
	if config.mob_markers {
		write_json(&config.viewer_mobs_path, &mobs)?;
	}

	write_overlays(config, combined_overlays)?;

	// Consolidated derived data for downstream tools: --emit-overlays <dir>
	// writes all five JSON files into one documented directory and prints the
	// absolute path of each file as it is written.
	if let Some(dir) = &config.emit_overlays {
		emit_overlay_files(dir);
		write_json_emit(&dir.join("structures.json"), &structures)?;
		write_json_emit(&dir.join("pois.json"), &pois)?;
		write_json_emit(&dir.join("mobs.json"), &mobs)?;
	}

	if let Some(dir) = &config.emit_player_data {
		crate::io::fs::create_dir_all(dir)?;
		let players = player::collect(&config.input_dir);
		write_json_emit(&dir.join("players.json"), &players)?;
	}

	Ok(())
}

/// Writes a value as JSON to a viewer output file
fn write_json<T: serde::Serialize>(path: &std::path::Path, value: &T) -> Result<()> {
	crate::io::fs::create_with_tmpfile(path, |file| {
		serde_json::to_writer(file, value).context("Failed to write viewer JSON")
	})
}

/// Writes a value as JSON to a file and prints its absolute path to stdout
fn write_json_emit<T: serde::Serialize>(path: &std::path::Path, value: &T) -> Result<()> {
	write_json(path, value)?;
	print_emitted(path);
	Ok(())
}

/// Prints the absolute path of an emitted file to stdout
fn print_emitted(path: &std::path::Path) {
	let display = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
	println!("{}", display.display());
}

/// Prints the absolute paths of the two overlay-data files already written into
/// the consolidated `--emit-overlays` directory by [write_overlays]
fn emit_overlay_files(dir: &std::path::Path) {
	print_emitted(&dir.join("inhabited_heatmap.json"));
	print_emitted(&dir.join("block_features.json"));
}

/// Builds the sorted per-dimension structure map from collected overlay data
fn structures_by_dimension(
	overlays: &overlay::OverlayData,
) -> std::collections::BTreeMap<&'static str, Vec<overlay::Structure>> {
	use overlay::Dimension;
	let mut by_dimension = std::collections::BTreeMap::new();
	for dim in Dimension::ALL {
		let mut structures = overlays.dimension(dim).structures.clone();
		structures.sort_by(|a, b| (a.bb, &a.structure_type).cmp(&(b.bb, &b.structure_type)));
		by_dimension.insert(dim.key(), structures);
	}
	by_dimension
}

/// Writes collected overlay data to all configured destinations
fn write_overlays(config: &Config, overlays: overlay::OverlayData) -> Result<()> {
	let dirs = config.overlay_output_dirs();
	if dirs.is_empty() {
		return Ok(());
	}
	let dir_refs: Vec<&std::path::Path> = dirs.iter().map(PathBuf::as_path).collect();
	overlays.write(&dir_refs)
}

/// Creates a file watcher for the
fn create_watcher(args: &Args) -> Result<(RecommendedWatcher, Receiver<()>)> {
	let (tx, rx) = mpsc::sync_channel::<()>(1);
	let mut watcher = notify::recommended_watcher(move |res| {
		// Ignore errors - we already have a watch trigger queued if try_send() fails
		let event: notify::Event = match res {
			Ok(event) => event,
			Err(err) => {
				warn!("Watch error: {err}");
				return;
			}
		};
		let notify::EventKind::Modify(modify_kind) = event.kind else {
			return;
		};
		if !matches!(
			modify_kind,
			notify::event::ModifyKind::Data(_)
				| notify::event::ModifyKind::Name(notify::event::RenameMode::To)
		) {
			return;
		}
		if !event
			.paths
			.iter()
			.any(|path| path.ends_with("level.dat") || path.extension() == Some("mcu".as_ref()))
		{
			return;
		}
		let _ = tx.try_send(());
	})?;
	watcher.watch(&args.input_dir, RecursiveMode::Recursive)?;
	Ok((watcher, rx))
}

/// Watches the data directory for changes, returning when a change has happened
fn wait_watcher(args: &Args, watch_channel: &Receiver<()>) -> Result<()> {
	info!("Watching for changes...");
	let () = watch_channel
		.recv()
		.context("Failed to read watch event channel")?;
	info!("Change detected.");

	thread::sleep(args.watch_delay);

	let _ = watch_channel.try_recv();

	Ok(())
}

/// MinedMap CLI main function
pub fn cli() -> Result<()> {
	let args = Args::parse();

	tracing_subscriber::fmt()
		.with_max_level(if args.verbose {
			tracing::Level::DEBUG
		} else {
			tracing::Level::INFO
		})
		.with_target(false)
		.init();

	let num_threads = match args.jobs {
		Some(0) => num_cpus::get(),
		Some(threads) => threads,
		None => 1,
	};
	let num_threads_initial = args.jobs_initial.unwrap_or(num_threads);

	let mut pool = setup_threads(num_threads_initial)?;

	let rt = tokio::runtime::Builder::new_current_thread()
		.build()
		.unwrap();

	let watch = args.watch.then(|| create_watcher(&args)).transpose()?;

	pool.install(|| generate(&args, &rt))?;

	let Some((_watcher, watch_channel)) = watch else {
		// watch mode disabled
		return Ok(());
	};

	if num_threads != num_threads_initial {
		pool = setup_threads(num_threads)?;
	}
	pool.install(move || {
		loop {
			wait_watcher(&args, &watch_channel)?;
			generate(&args, &rt)?;
		}
	})
}
