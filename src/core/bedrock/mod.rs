//! Bedrock Edition (LevelDB) support

use anyhow::{Result, bail};

use super::common::Config;

/// Runs all MinedMap generation steps for a Bedrock Edition world
pub fn generate(_config: &Config) -> Result<()> {
	bail!("Bedrock Edition support is not yet implemented");
}
