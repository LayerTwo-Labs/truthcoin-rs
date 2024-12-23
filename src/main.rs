//! Truthcoin-rs: A Rust implementation of the Truthcoin protocol
//! 
//! This implementation provides a decentralized oracle and prediction market system
//! using custom mathematics and cryptographic primitives.

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

use anyhow::Result;
use env_logger;
use log::{info, debug};

mod chain_objects;
mod crypto;
mod custom_math;

use custom_math::nice_matrix::{LabeledMatrix, BinaryLabeledMatrix};
use custom_math::get_weight;
use custom_math::outcome_consensus::*;

/// Main entry point for the Truthcoin protocol implementation
fn main() -> Result<()> {
    // Initialize logging with timestamp
    env_logger::Builder::from_default_env()
        .format_timestamp_secs()
        .init();
    
    info!("Starting Truthcoin protocol...");
    
    // Generate sample matrix for testing
    debug!("Generating sample matrix");
    let test_matrix = generate_sample_matrix(1);
    
    // Calculate binary scaling
    debug!("Computing binary scaling");
    let scaling = all_binary(&test_matrix);
    println!("Binary Scaling Result:\n{:#}", scaling);
    
    // Example of reputation-weighted consensus
    let reputation_weights = fast_rep(&[4.0, 3.0, 4.0, 1.0, 1.0, 1.0]);
    debug!("Using reputation weights: {:?}", reputation_weights);
    
    // Run consensus factory with parameters
    let factory_result = factory(
        &test_matrix,
        &scaling,
        Some(&reputation_weights),
        None,  // Default alpha
        None,  // Default beta
        Some(true),  // Include participation
        None,  // Default smoothing
    );
    
    info!("Consensus computation complete");
    debug!("Factory results: {:#?}", factory_result);
    
    Ok(())
}

/// Generates documentation for the project
/// 
/// This can be run with:
/// ```bash
/// cargo doc --no-deps --open
/// ```
#[cfg(doc)]
pub mod documentation {
    #![doc = include_str!("../README.md")]
}
