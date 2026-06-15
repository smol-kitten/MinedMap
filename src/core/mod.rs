//! Core functions of the MinedMap CLI

mod bedrock;
mod common;
mod entity_collector;
mod heightmap;
mod java_random;
mod metadata_writer;
mod overlay;
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

use common::{Config, Edition, ImageFormat, UnknownBlocks};
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
	/// Emit per-chunk overlay data to the given directory
	///
	/// Writes `inhabited_heatmap.json` and `block_features.json` describing the
	/// `InhabitedTime` and notable blocks of each chunk, collected during the
	/// regular render pass. Does not affect the generated map tiles.
	#[arg(long, value_name = "DIR")]
	pub emit_overlays: Option<PathBuf>,
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
	let (regions, overlays) = RegionProcessor::new(config).run()?;
	TileRenderer::new(config, rt, &regions).run()?;
	let tiles = TileMipmapper::new(config, &regions).run()?;
	EntityCollector::new(config, &regions).run()?;
	MetadataWriter::new(config, &tiles).run()?;

	write_overlays(config, overlays)?;

	Ok(())
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
