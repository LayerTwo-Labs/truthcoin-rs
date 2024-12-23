use ndarray::Array1;
use std::collections::{HashMap, BTreeMap};
use rand::Rng;
use chrono::{DateTime, Utc, Duration};
use chrono::Datelike;

const SLOTS_PER_ERA: i32 = 500; // Maximum number of decisions that can be made in each era
const SLOTS_PER_FUTURE_ERA_DECREASE: i32 = 25; // How many fewer slots each future era gets
const FUTURE_ERAS_TO_TRACK: i32 = 20; // Number of future eras to maintain slots for
const BASE_LISTING_FEE: f64 = 0.00097; // Base listing fee for standard slots (approximately $97 USD)
const BASE_LISTING_FEE_OVERFLOW: f64 = 0.0; // Base listing fee for overflow slots (free)

struct Chain {
    name: String,
    max_slots_per_era: i32, // fixed at 500 slots per era (maximum number of decisions that can be made in this era)
    claimed_slots: BTreeMap<String, i32>, // number of slots claimed in each era
    available_slots: BTreeMap<String, i32>, // number of slots available in each era
    overflow_slots: BTreeMap<String, i32>, // number of overflow slots claimed in each era (unlimited)
    base_listing_fee: f64,
    block_number: u64,
    timestamp: DateTime<Utc>,
    era_code: String, // format: YYYY.qN where N is the quarter (1-4)
}

impl Chain {
    // Get the next N era codes starting from a given era
    fn get_next_n_eras(start_era: &str, n: i32) -> Vec<String> {
        let mut result = Vec::with_capacity(n as usize);
        let mut parts = start_era.split('.');
        let year: i32 = parts.next().unwrap().parse().unwrap();
        let quarter: i32 = parts.next().unwrap()[1..].parse().unwrap();
        
        let mut current_year = year;
        let mut current_quarter = quarter;
        
        for _ in 0..n as usize {
            result.push(format!("{}.q{}", current_year, current_quarter));
            current_quarter += 1;
            if current_quarter > 4 {
                current_quarter = 1;
                current_year += 1;
            }
        }
        result
    }

    // Initialize slots for the next 20 eras with decreasing availability
    fn initialize_slot_availability(&mut self) {
        let eras = Self::get_next_n_eras(&self.era_code, FUTURE_ERAS_TO_TRACK);
        for (i, era) in eras.iter().enumerate() {
            let available = SLOTS_PER_ERA - ((i as i32) * SLOTS_PER_FUTURE_ERA_DECREASE);
            self.available_slots.insert(era.clone(), available);
            self.claimed_slots.insert(era.clone(), 0);
            self.overflow_slots.insert(era.clone(), 0);
        }
    }

    // Handle era transition by redistributing slots
    fn handle_era_transition(&mut self, old_era: &str) {
        // Remove old era's slots
        self.available_slots.remove(old_era);
        self.claimed_slots.remove(old_era);
        self.overflow_slots.remove(old_era);

        // Get the next 20 eras after the current one
        let next_eras = Self::get_next_n_eras(&self.era_code, FUTURE_ERAS_TO_TRACK);
        
        // Add SLOTS_PER_FUTURE_ERA_DECREASE new slots to each future era
        for era in next_eras {
            *self.available_slots.entry(era.clone()).or_insert(0) += SLOTS_PER_FUTURE_ERA_DECREASE;
            self.overflow_slots.entry(era.clone()).or_insert(0);
        }
    }

    // Attempt to claim a standard slot in a specific era
    fn claim_slot(&mut self, era: &str) -> Result<(), &'static str> {
        let available = self.available_slots.get(era).copied().unwrap_or(0);
        let claimed = self.claimed_slots.get(era).copied().unwrap_or(0);
        
        if claimed >= available {
            Err("No slots available in specified era")
        } else {
            *self.claimed_slots.entry(era.to_string()).or_insert(0) += 1;
            Ok(())
        }
    }

    // Attempt to claim an overflow slot in a specific era with optional fee
    fn claim_overflow_slot(&mut self, era: &str, _fee: Option<f64>) -> Result<(), &'static str> {
        *self.overflow_slots.entry(era.to_string()).or_insert(0) += 1;
        Ok(())
    }

    // Get number of available (unclaimed) slots for a specific era
    fn get_available_slots(&self, era: &str) -> i32 {
        let available = self.available_slots.get(era).copied().unwrap_or(0);
        let claimed = self.claimed_slots.get(era).copied().unwrap_or(0);
        available - claimed
    }

    // Get total number of available slots across all eras
    fn total_available_slots(&self) -> i32 {
        self.available_slots.values().sum()
    }

    // For testing purposes
    #[cfg(test)]
    fn with_timestamp(name: String, timestamp: DateTime<Utc>) -> Self {
        let era_code = Self::calculate_era_code(timestamp);
        let mut chain = Self { 
            name,
            max_slots_per_era: SLOTS_PER_ERA,
            claimed_slots: BTreeMap::new(),
            available_slots: BTreeMap::new(),
            overflow_slots: BTreeMap::new(),
            base_listing_fee: BASE_LISTING_FEE,
            block_number: 1,
            timestamp,
            era_code,
        };
        chain.initialize_slot_availability();
        chain
    }

    // Calculate era code from a timestamp
    fn calculate_era_code(timestamp: DateTime<Utc>) -> String {
        let year = timestamp.year();
        let month = timestamp.month();
        let quarter = ((month - 1) / 3) + 1;
        format!("{}.q{}", year, quarter)
    }

    fn new(name: String) -> Self {
        let timestamp = Utc::now();
        let era_code = Self::calculate_era_code(timestamp);
        let mut chain = Self { 
            name,
            max_slots_per_era: SLOTS_PER_ERA,
            claimed_slots: BTreeMap::new(),
            available_slots: BTreeMap::new(),
            overflow_slots: BTreeMap::new(),
            base_listing_fee: BASE_LISTING_FEE,
            block_number: 1,
            timestamp,
            era_code,
        };
        chain.initialize_slot_availability();
        chain
    }

    fn increment_block(&mut self) {
        self.block_number += 1;
        // Random duration between 7 and 13 minutes
        let minutes = rand::thread_rng().gen_range(7..=13);
        let old_era = self.era_code.clone();
        self.timestamp = self.timestamp + Duration::minutes(minutes as i64);
        
        // Check if we've crossed into a new quarter
        let new_era = Self::calculate_era_code(self.timestamp);
        if new_era != old_era {
            self.era_code = new_era;
            self.handle_era_transition(&old_era);
        }
    }
}

struct Decision {
    chain: String,
    tau_from_now: i32,
    standard: bool, // true for standard slots, false for overflow slots
}

fn get_q_standard_decisions(chain_to_add_on: &str, tau_until_after_event: i32, decisions: &[Decision]) -> i32 {
    decisions.iter()
        .filter(|d| d.chain == chain_to_add_on && d.tau_from_now == tau_until_after_event && d.standard)
        .count() as i32
}

fn get_q_overflow_decisions(chain_to_add_on: &str, tau_until_after_event: i32, decisions: &[Decision]) -> i32 {
    decisions.iter()
        .filter(|d| d.chain == chain_to_add_on && d.tau_from_now == tau_until_after_event && !d.standard)
        .count() as i32
}

fn get_slot_info(previous_base_fee: f64, max_slots: i32, width: f64, log: bool) -> (Array1<f64>, f64, f64, f64) {
    
    // length of each "pentile" (ie, one-fifth)
    let pent_length = max_slots as f64 / 5.0;
    
    // two ways of calculating the distribution of prices
    let slot_prices = if log {
        Array1::from(vec![
            previous_base_fee / width,
            previous_base_fee / (width.ln() / 2.0).exp(),
            previous_base_fee,
            previous_base_fee * (width.ln() / 2.0).exp(),
            previous_base_fee * width,
        ])
    } else {
        Array1::from(vec![
            previous_base_fee / width,
            previous_base_fee * (1.0 + 1.0 / width) / 2.0,
            previous_base_fee,
            previous_base_fee * (1.0 + width) / 2.0,
            previous_base_fee * width,
        ])
    };

    let log_half = (width.ln() / 2.0).exp();
    let multiple = if log {
        log_half / (log_half + 1.0)
    } else {
        4.0 / 7.0
    };

    // this is how many slots we expected people to have bought, by now
    let expected_slot_usage = Array1::from(vec![1.0, 1.0, 0.5, 0.0, 0.0]) * pent_length;
    let expected_spend = slot_prices.dot(&expected_slot_usage);

    // this is how many slots there are total , and how much could be spent total
    let max_slot_usage = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0]) * pent_length;
    let max_spend = slot_prices.dot(&max_slot_usage);

    (slot_prices, multiple, expected_spend, max_spend)
}

fn query_cost_to_author_standard(chain_to_add_on: &str, tau_until_after_event: i32, chains: &HashMap<String, Chain>, decisions: &[Decision]) -> f64 {
    let chain = chains.get(chain_to_add_on).unwrap();
    let existing_decisions = get_q_standard_decisions(chain_to_add_on, tau_until_after_event, decisions);

    if existing_decisions == chain.max_slots_per_era {
        println!("Too many Standard Decisions already...you will have to use Overflow Decisions or select a different time (or different Chain).");
        return -1.0;
    }

    let pent_length = chain.max_slots_per_era as f64 / 5.0;
    let pent = 1 + (existing_decisions as f64 / pent_length).floor() as usize;

    let (slot_prices, _, _, _) = get_slot_info(chain.base_listing_fee, chain.max_slots_per_era, 2.0, true);
    let fee = slot_prices[pent - 1];

    println!("You are number {} of {}, and are in Pentile {}.", existing_decisions + 1, chain.max_slots_per_era, pent);
    println!("This period's base fee is {}, so you must pay {}.", chain.base_listing_fee, fee);

    fee
}

fn query_cost_to_author_overflow(_chain_to_add_on: &str, _tau_until_after_event: i32, _chains: &HashMap<String, Chain>, _decisions: &[Decision]) -> f64 {
    // Overflow slots are free by default (BASE_LISTING_FEE_OVERFLOW = 0.0)
    // Users can optionally provide additional fee when claiming the slot
    BASE_LISTING_FEE_OVERFLOW
}

fn calc_new_base_fee(previous_base_fee: f64, max_slots: i32, used_slots: i32) -> f64 {
    // This is global information on the whole chain
    // At any given point in time, these values are fixed and shared across the whole chain
    // eg whole chain is: 500 max slots per era, 100 slots per pent
    let pent_length = max_slots as f64 / 5.0;
    let (slot_costs, multiple, _, _) = get_slot_info(previous_base_fee, max_slots, 2.0, true);

    let marginal_index = Array1::from((0..5).map(|i| used_slots as f64 - (pent_length * i as f64)).collect::<Vec<f64>>());
    let full_slots = marginal_index.mapv(|x| (pent_length < x) as i32);
    let marginal_slot = full_slots.sum() as usize;

    let mut actual_slot_usage = full_slots.mapv(|x| x as f64 * pent_length);
    actual_slot_usage[marginal_slot] = marginal_index[marginal_slot];

    let actual_spend = slot_costs.dot(&actual_slot_usage);
    let new_fee = (actual_spend / pent_length) * multiple;

    new_fee
}

fn update_base_fee(chain: &mut Chain, decisions: &[Decision]) {
    let used_slots = get_q_standard_decisions(&chain.name, 0, decisions);
    let new_fee = calc_new_base_fee(chain.base_listing_fee, chain.max_slots_per_era, used_slots);
    chain.base_listing_fee = new_fee;
    chain.increment_block();
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chrono::TimeZone;
    use chrono::Datelike;

    #[test]
    fn test_slot_availability() {
        let chain = Chain::new("test".to_string());
        
        // Initially should have decreasing slots over next 20 eras
        let current_era = chain.era_code.clone();
        let next_eras = Chain::get_next_n_eras(&current_era, FUTURE_ERAS_TO_TRACK);
        
        // Verify slot distribution
        assert_eq!(chain.get_available_slots(&next_eras[0]), 500); // Current era
        assert_eq!(chain.get_available_slots(&next_eras[1]), 475); // Next era
        assert_eq!(chain.get_available_slots(&next_eras[2]), 450); // Era after next
        assert_eq!(chain.get_available_slots(&next_eras[19]), 25); // Last era
        
        // Total should be 5250 slots
        assert_eq!(chain.total_available_slots(), 5250);
    }

    #[test]
    fn test_slot_claiming() {
        let mut chain = Chain::new("test".to_string());
        let current_era = chain.era_code.clone();
        
        // Initially should have 500 slots in current era
        assert_eq!(chain.get_available_slots(&current_era), 500);
        
        // Claim some slots
        for _ in 0..23 {
            assert!(chain.claim_slot(&current_era).is_ok());
        }
        assert_eq!(chain.get_available_slots(&current_era), 477);
        
        // Try claiming in future era
        let next_era = Chain::get_next_n_eras(&current_era, 1_i32)[0].clone();
        for _ in 0..23 {
            assert!(chain.claim_slot(&next_era).is_ok());
        }
        assert_eq!(chain.get_available_slots(&next_era), 452); // 475 - 23
    }

    #[test]
    fn test_overflow_slots() {
        let mut chain = Chain::new("test".to_string());
        let current_era = chain.era_code.clone();
        
        // Claim some overflow slots
        for _ in 0..100 {
            assert!(chain.claim_overflow_slot(&current_era, Some(0.1)).is_ok());
        }
        assert_eq!(*chain.overflow_slots.get(&current_era).unwrap(), 100);
        
        // Claim more overflow slots with different fees
        assert!(chain.claim_overflow_slot(&current_era, None).is_ok());
        assert!(chain.claim_overflow_slot(&current_era, Some(1.0)).is_ok());
        assert_eq!(*chain.overflow_slots.get(&current_era).unwrap(), 102);
    }

    #[test]
    fn test_era_transition() {
        let mut chain = Chain::with_timestamp(
            "test".to_string(),
            Utc.with_ymd_and_hms(2024, 12, 31, 23, 59, 59).unwrap()
        );
        let old_era = chain.era_code.clone();
        
        // Force era transition
        chain.increment_block();
        
        // Old era should be removed
        assert_eq!(chain.get_available_slots(&old_era), 0);
        
        // Each future era should have gained 25 slots
        let next_eras = Chain::get_next_n_eras(&chain.era_code, FUTURE_ERAS_TO_TRACK);
        for era in next_eras {
            assert!(chain.get_available_slots(&era) > 0);
        }
    }

    #[test]
    fn test_era_code_calculation() {
        // Test different quarters
        let timestamp = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        assert_eq!(Chain::calculate_era_code(timestamp), "2024.q1");

        let timestamp = Utc.with_ymd_and_hms(2024, 4, 1, 0, 0, 0).unwrap();
        assert_eq!(Chain::calculate_era_code(timestamp), "2024.q2");

        let timestamp = Utc.with_ymd_and_hms(2024, 7, 1, 0, 0, 0).unwrap();
        assert_eq!(Chain::calculate_era_code(timestamp), "2024.q3");

        let timestamp = Utc.with_ymd_and_hms(2024, 10, 1, 0, 0, 0).unwrap();
        assert_eq!(Chain::calculate_era_code(timestamp), "2024.q4");

        // Test year transition
        let timestamp = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        assert_eq!(Chain::calculate_era_code(timestamp), "2025.q1");

        // Test era code updates when crossing quarters
        let mut chain = Chain::with_timestamp("test".to_string(), 
            Utc.with_ymd_and_hms(2024, 3, 31, 23, 59, 59).unwrap());
        assert_eq!(chain.era_code, "2024.q1");
        
        // Add enough minutes to cross into Q2
        for _ in 0..10 {
            chain.increment_block();
        }
        assert_eq!(chain.era_code, "2024.q2");
    }

    #[test]
    fn test_calc_new_base_fee() {
        // Below: Many slots were used, fee increases significantly
        let new_fee = calc_new_base_fee(BASE_LISTING_FEE, SLOTS_PER_ERA, 490);
        assert_relative_eq!(new_fee, 0.003080965, epsilon = 1e-6);

        // Sanity check: Used Slots = Target Slots, so price doesn't change
        let new_fee = calc_new_base_fee(BASE_LISTING_FEE, SLOTS_PER_ERA, 250);
        assert_relative_eq!(new_fee, 0.00097, epsilon = 1e-6);

        // Few slots used, fee greatly decreases
        let new_fee = calc_new_base_fee(BASE_LISTING_FEE, SLOTS_PER_ERA, 70);
        assert_relative_eq!(new_fee, 0.000198774, epsilon = 1e-6);

        // Sanity check again
        let new_fee = calc_new_base_fee(0.000198774, SLOTS_PER_ERA, 250);
        assert_relative_eq!(new_fee, 0.000198774, epsilon = 1e-6);

        // Fee increases marginally
        let new_fee = calc_new_base_fee(BASE_LISTING_FEE, SLOTS_PER_ERA, 280);
        assert_relative_eq!(new_fee, 0.001140246, epsilon = 1e-6);
    }
}
