use ndarray::Array1;
use std::collections::HashMap;

struct Branch {
    name: String,
    max_decisions: i32,
    base_listing_fee: f64,
}

impl Branch {
    fn new(name: String, max_decisions: i32, base_listing_fee: f64) -> Self {
        Self { name, max_decisions, base_listing_fee }
    }
}

struct Decision {
    branch: String,
    tau_from_now: i32,
    standard: bool,
}

fn get_q_standard_decisions(branch_to_add_on: &str, tau_until_after_event: i32, decisions: &[Decision]) -> i32 {
    decisions.iter()
        .filter(|d| d.branch == branch_to_add_on && d.tau_from_now == tau_until_after_event && d.standard)
        .count() as i32
}

fn get_slot_info(previous_base_fee: f64, max_slots: i32, width: f64, log: bool) -> (Array1<f64>, f64, f64, f64) {
    let pent_length = max_slots as f64 / 5.0;
    
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

    let expected_slot_usage = Array1::from(vec![1.0, 1.0, 0.5, 0.0, 0.0]) * pent_length;
    let expected_spend = slot_prices.dot(&expected_slot_usage);

    let max_slot_usage = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0]) * pent_length;
    let max_spend = slot_prices.dot(&max_slot_usage);

    (slot_prices, multiple, expected_spend, max_spend)
}

fn query_cost_to_author_standard(branch_to_add_on: &str, tau_until_after_event: i32, branches: &HashMap<String, Branch>, decisions: &[Decision]) -> f64 {
    let branch = branches.get(branch_to_add_on).unwrap();
    let existing_decisions = get_q_standard_decisions(branch_to_add_on, tau_until_after_event, decisions);

    if existing_decisions == branch.max_decisions {
        println!("Too many Standard Decisions already...you will have to use Overflow Decisions or select a different time (or different Branch).");
        return -1.0;
    }

    let pent_length = branch.max_decisions as f64 / 5.0;
    let pent = 1 + (existing_decisions as f64 / pent_length).floor() as usize;

    let (slot_prices, _, _, _) = get_slot_info(branch.base_listing_fee, branch.max_decisions, 2.0, true);
    let fee = slot_prices[pent - 1];

    println!("You are number {} of {}, and are in Pentile {}.", existing_decisions + 1, branch.max_decisions, pent);
    println!("This period's base fee is {}, so you must pay {}.", branch.base_listing_fee, fee);

    fee
}

fn calc_new_base_fee(previous_base_fee: f64, max_slots: i32, used_slots: i32) -> f64 {
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

fn update_base_fee(branch: &mut Branch, decisions: &[Decision]) {
    let used_slots = get_q_standard_decisions(&branch.name, 0, decisions);
    let new_fee = calc_new_base_fee(branch.base_listing_fee, branch.max_decisions, used_slots);
    branch.base_listing_fee = new_fee;
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // We don't need a setup function for these tests
    // as they're testing a single function with different inputs

    #[test]
    fn test_calc_new_base_fee() {
        // Many slots used, fee increases
        let new_fee = calc_new_base_fee(0.02, 500, 490);
        assert_relative_eq!(new_fee, 0.06351472, epsilon = 1e-6);

        // Sanity check: Used Slots = Target Slots, price doesn't change
        let new_fee = calc_new_base_fee(0.0222, 500, 250);
        assert_relative_eq!(new_fee, 0.0222, epsilon = 1e-6);

        // Few slots used, fee decreases
        let new_fee = calc_new_base_fee(0.02, 500, 70);
        assert_relative_eq!(new_fee, 0.004100505, epsilon = 1e-6);

        // Sanity check again
        let new_fee = calc_new_base_fee(0.004100505, 500, 250);
        assert_relative_eq!(new_fee, 0.004100505, epsilon = 1e-6);

        // Fee increases marginally
        let new_fee = calc_new_base_fee(0.02, 500, 280);
        assert_relative_eq!(new_fee, 0.02351472, epsilon = 1e-6);
    }
}

