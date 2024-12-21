pub mod nice_matrix;
pub mod outcome_consensus;

// Re-export everything from custom_math.rs
mod custom_math;
pub use self::custom_math::{weighted_median, weighted_prin_comp, re_weight, get_weight};
