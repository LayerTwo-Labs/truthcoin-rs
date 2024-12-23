//! Cryptographic hash functions for the truthcoin protocol.
//! 
//! This module provides cryptographic hash functions used throughout the protocol
//! for creating unique identifiers and ensuring data integrity.

use sha2::{Sha256, Digest};
use log::trace;
use thiserror::Error;

/// Errors that can occur during hashing operations
#[derive(Error, Debug)]
pub enum HashError {
    #[error("Failed to compute hash: {0}")]
    ComputationError(String),
}

/// Computes the SHA-256 hash of the input string and returns it as a hexadecimal string.
///
/// # Arguments
///
/// * `input` - The string to hash
///
/// # Returns
///
/// Returns a hex-encoded string of the SHA-256 hash.
///
/// # Example
///
/// ```
/// use truthcoin_rs::crypto::hash::sha256;
///
/// let hash = sha256("Hello, world!").unwrap();
/// assert_eq!(hash, "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3");
/// ```
pub fn sha256(input: &str) -> Result<String, HashError> {
    trace!("Computing SHA-256 hash for input of length {}", input.len());
    
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_sha256_basic() {
        let input = "Hello, world!";
        let hash = sha256(input).unwrap();
        assert_eq!(hash, "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3");
    }

    #[test]
    fn test_sha256_empty_string() {
        let hash = sha256("").unwrap();
        assert_eq!(hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn test_sha256_long_input() {
        let input = "a".repeat(1000000);
        let hash = sha256(&input).unwrap();
        assert_eq!(hash.len(), 64); // SHA-256 hash is always 64 hex characters
    }

    #[test]
    fn test_sha256_special_characters() {
        let input = "Hello, 世界! 🌍";
        let hash = sha256(input).unwrap();
        assert_eq!(hash.len(), 64);
    }
}
