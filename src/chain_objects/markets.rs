// We'll need these...
use ndarray::{Array, IxDyn, s};
use crate::crypto::hash::sha256;
use std::{collections::HashMap, vec};


// use crate::market;
//use crate::{Decision, Market, add_decision_to_database, add_market_to_database};



// this is what a Decision looks like
#[derive(Debug, Clone)]
pub struct Decision {
    state: i32,                       // resolution status (-1 = Resolved, 0 = No Attempts, N = N failed resolution attempts)
    resolved_outcome: f64,            // post-judgement result (0-1)
    size: usize,                      // size in bytes
    prompt: String,                   // the decision text
    owner_ad: String,                 // wallet address
    tau_from_now: u8,                 // time until resolution in the vote-matrix 
    scaled: bool,                     // true = min/max will not be 0/1
    min: f64,                         // only meaningful for scaled
    max: f64,                         // only meaningful for scaled
    standard: bool,                   // 1 = standard, 0 = overflow
}

// this is what a Market looks like
#[derive(Debug, Clone)]
pub struct Market {
    title: String,                         // title of market
    state: u8,                             // resolution status (1 = Trading, 2 = Resolved)
    b: f64,                                // lmsr beta parameter
    trading_fee: f64,                      // charge on every trade
    description: String,                   // human-readable description of market
    tags: Vec<String>,                     // tags to make the market easy to find
    size: usize,                           // size in bytes
    tau_from_now: u8,                      // time until resolution in the vote-matrix 
    owner_ad: String,                      // wallet address
    d_axis: Vec<String>,                   // hashes of the decisions, as the axes of an array
    cube_f: Array<u8, IxDyn>,              // functional transformations, eg 1/x, log(), x^2 etc 
    shares: Array<f64, IxDyn>,             // shares at this point in time
    f_prices: Array<f64, IxDyn>,           // the final prices, after resolution is finished
    treasury: f64,                         // cash that the MSR contains
}

// Measures the dimensionality of a market, from a properly formatted string
// This is pretty hacky so probably it can be improved ...
fn get_dim(input: &[String]) -> Vec<usize> {
    // locate all instances of "+" (which is: when a dimension has more than one decision)
    let mut d_vector: Vec<usize> = input.iter().map(|s| s.split('+').count()).collect();
    // add 1 to the whole vector, to account for the null state
    for num in d_vector.iter_mut() {
        *num += 1;
    }
    d_vector
}




// here is "new()", for adding a decision, it will autofill the size for us
impl Decision {
    fn new(prompt: String, scaled: bool, min: Option<f64>, max: Option<f64>, tau_from_now: u8, owner_ad: String) -> Self {
        let mut decision = Decision {
            state: 0,
            resolved_outcome: 0.5,
            size: 0, // We'll calculate this later
            prompt,
            owner_ad,
            tau_from_now,
            scaled,
            min: if scaled { min.unwrap_or(0.0) } else { 0.0 },
            max: if scaled { max.unwrap_or(1.0) } else { 1.0 },
            standard: true,
        };
        decision.calculate_size();
        decision
    }

    fn calculate_size(&mut self) {
        self.size = std::mem::size_of_val(&self.state) +
                    std::mem::size_of_val(&self.resolved_outcome) +
                    self.prompt.len() +
                    self.owner_ad.len() +
                    std::mem::size_of_val(&self.tau_from_now) +
                    std::mem::size_of_val(&self.scaled) +
                    std::mem::size_of_val(&self.min) +
                    std::mem::size_of_val(&self.max) +
                    std::mem::size_of_val(&self.standard);
    }

    // this is Sha256() of the Decision -- we will use this to fill the hashmap 
    fn generate_hash(&self) -> String {
        let decision_string = format!("{}{}{:?}{}{}{}{}{}{}{}", 
            self.state, self.resolved_outcome, self.size, self.prompt, 
            self.owner_ad, self.tau_from_now, self.scaled, 
            self.min, self.max, self.standard);
        sha256(&decision_string)
    }

}


// We will define the market.add() now -- it will automatically calculate the size, array of shares, etc
// the hash only covers fields that do NOT change, while trading
// here we go...
impl Market {
    fn new(title: String, b: f64, trading_fee: f64, description: String,
        tags: Vec<String>, owner_ad: String,
        d_axis: Vec<String>) -> Self {
            let market_dim: Vec<usize> = get_dim(&d_axis);
            let mut market = Market {
                title,
                state: 1,
                b,
                trading_fee,
                description,
                tags,
                size: 0, // we'll fill this in later
                tau_from_now: 5, // TODO: this should be the max() of all the tfn of the market's decisions
                owner_ad,
                d_axis,
                cube_f: Array::default(IxDyn(&market_dim)),
                shares: Array::zeros(IxDyn(&market_dim)),
                f_prices: Array::zeros(IxDyn(&market_dim)),
                treasury: 0.0, // we'll fill this in later
            };
            market.calculate_size();
            market.treasury = market.calc_treasury();
            market
        }

    fn calculate_size(&mut self) {
        let cumulative_size  = std::mem::size_of_val(&self.state) +
                    std::mem::size_of_val(&self.b) +
                    std::mem::size_of_val(&self.trading_fee) +
                    self.title.len() +
                    self.description.len() +
                    self.tags.iter().map(|tag| tag.len()).sum::<usize>() +
                    std::mem::size_of_val(&self.tau_from_now) +
                    self.owner_ad.len() +
                    self.d_axis.len() * std::mem::size_of::<String>() +
                    self.cube_f.len() * std::mem::size_of::<u8>() +
                    self.shares.len() * std::mem::size_of::<f64>() +
                    self.f_prices.len() * std::mem::size_of::<f64>() +
                    std::mem::size_of_val(&self.treasury);
            self.size = cumulative_size;
    }
    
    // take the hash of the market -- omit fields that change during trading (b, shares, p_final, treasury)
    fn generate_hash(&self) -> String {
        let market_string = format!("{}{}{}{}{:?}{}{}{:?}{:?}", 
            self.title, self.state, self.trading_fee, self.description, 
            self.tags, self.size, self.tau_from_now, self.owner_ad, 
            self.d_axis);
        sha256(&market_string)
    }

    // now finally something important...
    // calculate the market treasury, given the array of shares
    fn calc_treasury(&self) -> f64 {
        // our starting ingredients
        let b =self.b;
        let s= self.shares.clone();
        
        // divide array by b, then e^x whole array
        let divided_array = s.mapv(|x| x / b);
        let exp_array = divided_array.mapv(|x| x.exp());

        // sum up the whole array, then natural log that, finally multiply by b
        let sum: f64 = exp_array.iter().sum();
        let ln_sum = sum.ln();
        let result=ln_sum*b;

        result
    }

    // now we will show the instantaneous prices, given the shares and b
    fn show_inst_prices(&self) -> Array<f64, IxDyn> {
        // our starting ingredients
        let b =self.b;
        let s= self.shares.clone();
        
        // divide array by b, then e^x whole array
        let divided_array = s.mapv(|x| x / b);
        let exp_array = divided_array.mapv(|x| x.exp());

        // sum up the whole array
        let sum: f64 = exp_array.iter().sum();

        // divide whole array by the sum
        let result = exp_array.mapv(|x| x / sum);
        result
    }

    // every time someone buys or sells, we must update the market,
    // here's how to do it -- just change the shares array!
    fn update(&mut self, new_share_array: Array<f64, IxDyn>) {

        self.shares = new_share_array;        // set every share value to the value provided
        self.treasury = self.calc_treasury(); // calculate the new treasury and store it

    }

    // how much does it cost, to update the market?
    // in other words: how much it costs, to buy shares -- or how much you get, for selling your shares
    fn query_update_cost(&self, new_share_array: Array<f64, IxDyn>) -> f64 {

        // clone the market, and update *that* one...
        let mut new = self.clone();
        new.update(new_share_array);

        // compare treasury costs
        let cost = new.treasury - self.treasury;
        cost
    }


    // obscure but we might as well get it out of the way
    fn query_amp_b_cost(&self, new_b: f64) -> f64 {

        // we can only increase b -- no stealing!
        if self.b > new_b {
            panic!("b can only be increased, not decreased")
        }

        // current treasury
        let current_treasury = self.treasury;

        // copy the market, and see what the treasury would be (with the higher b)
        let mut new = self.clone();
        new.b = new_b;
        let asprational_treasury = new.calc_treasury();

        // subtract to get the cost the user must pay
        let result = asprational_treasury - current_treasury;
        result
    }


}


// Database I/O

fn add_decision_to_database(decision: &Decision, database: &mut HashMap<String, Decision>) {
    let hash = decision.generate_hash();
    database.insert(hash, decision.clone());
}

fn add_market_to_database(market: &Market, database: &mut HashMap<String, Market>) {
    let hash = market.generate_hash();
    database.insert(hash, market.clone());
}



pub fn generate_sample_markets() -> (HashMap<String, Decision>, HashMap<String, Market>) {
    let mut decision_database: HashMap<String, Decision> = HashMap::new();
    let mut market_database: HashMap<String, Market> = HashMap::new();

    // Example Decisions
    let new_decision_1 = Decision::new(
        "Will JD Vance win in 2028?".to_string(),
        false,
        None,
        None,
        3,
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa".to_string(),
    );

    let new_decision_2 = Decision::new(
        "Republican electoral college votes in 2028?".to_string(),
        true,
        Some(0.0),
        Some(538.0),
        3,
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa".to_string(),
    );

    let new_decision_3 = Decision::new(
        "Good economy in year 2029?".to_string(),
        false,
        None,
        None,
        3,
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa".to_string(),
    );

    let new_decision_4 = Decision::new(
        "What will the S&P 500 be, on March 1, 2029?".to_string(),
        true,
        Some(4000.0),
        Some(26000.0),
        3,
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa".to_string(),
    );

    add_decision_to_database(&new_decision_1, &mut decision_database);
    add_decision_to_database(&new_decision_2, &mut decision_database);
    add_decision_to_database(&new_decision_3, &mut decision_database);
    add_decision_to_database(&new_decision_4, &mut decision_database);

    println!(" --- Printing Decision Database --- ");
    println!(" ");
    println!("{:?}", decision_database);
    println!(" ");
    println!("{:?}", decision_database.len());
    println!(" ");
    

    let mut market_1 = Market::new(
        "Vance.2028".to_string(),
        7.001,
        0.005,
        "Will JD Vance win the presidency in 2028?".to_string(),
        vec!["politics".to_string(), "elections".to_string(), "usa".to_string(), "president".to_string()],
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa".to_string(),
        vec!["f3b197085705c7f879616178debf308336f3df1237c25bef1ba5988f9bb987ca".to_string()],
    );

    let mut market_2 = Market::new(
        "MultiDim-demo".to_string(),
        3.2,
        0.005,
        "Will JD Vance win the presidency in 2028?".to_string(),
        vec!["politics".to_string(), "elections".to_string(), "usa".to_string(), "president".to_string()],
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa".to_string(),
        vec!["f3b197085705c7f879616178debf308336f3df1237c25bef1ba5988f9bb987ca".to_string(),
        "e674cf4e6decbcb090542c2cce9984f4ffeafaa1f351d1cec56c2cf2aa188d62+ddfa2d27c2028cefd5a343cceafe2bf0e5d8b659b0d9980b973404b6012bf8bf".to_string(),
        "917b6d9f57d1e1a19ba0d8ac15f652def9929c25c86234e695022db7b665583f".to_string()],
    );

    let market_hash_1 = "7304df381d951a6edf08f769f9812d23369a5a8744cb7c390c41149fd859842e";
    let market_hash_2 = "4ef8eeb89996bf3b40d04c28e1495c506417f4a3526fe96c829e882fbce1c8f7";

    add_market_to_database(&market_1, &mut market_database);
    add_market_to_database(&market_2, &mut market_database);

    println!(" ");
    println!(" --- Printing Market Database --- ");
    println!(" ");
    println!("{:?}", market_database);
    println!(" ");
    println!(" ");
    println!("{:?}", market_database.len());
    println!(" ");

    let prices1 = market_1.show_inst_prices();
    let prices2 = market_2.show_inst_prices();

    (decision_database, market_database)


}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (HashMap<String, Decision>, HashMap<String, Market>) {
        generate_sample_markets()
    }

    #[test]
    fn test_database_creation() {
        let (decision_database, market_database) = setup();
        assert_eq!(decision_database.len(), 4);
        assert_eq!(market_database.len(), 2);
    }
    

    #[test]
    fn test_market_prices() {
        let (_, market_database) = setup();

        let market_1: String = "7304df381d951a6edf08f769f9812d23369a5a8744cb7c390c41149fd859842e".to_string(); // known by me!
        let market_2: String = "4ef8eeb89996bf3b40d04c28e1495c506417f4a3526fe96c829e882fbce1c8f7".to_string();

        let prices1 = market_database.get(&market_1).unwrap().show_inst_prices();
        assert!((prices1.slice(s![1]).into_scalar() - 0.50).abs() < 1e-10);

        let prices2 = market_database.get(&market_2).unwrap().show_inst_prices();
        assert!((prices2.slice(s![1,2,1]).into_scalar() - 0.08333333333333333).abs() < 1e-10);
    }

    // Test how market updates during trading 
    #[test]
    fn test_market_update() {
        let (_, mut market_database) = setup();

        let market_1: String = "7304df381d951a6edf08f769f9812d23369a5a8744cb7c390c41149fd859842e".to_string(); // known by me!

        // the updates we will make...
        let desired_shares_1 = Array::from_vec(vec![01.0, 0.0]).into_shape(IxDyn(&[2])).unwrap();  // purchase 1 share of State 1
        let desired_shares_2 = Array::from_vec(vec![01.0, 6.0]).into_shape(IxDyn(&[2])).unwrap();  // then, purchase 6 shares of State 2
        let desired_shares_3 = Array::from_vec(vec![18.0, 6.0]).into_shape(IxDyn(&[2])).unwrap(); // last, purchase 17 shares of State 1

        let cost1 = market_database.get(&market_1).unwrap().query_update_cost(desired_shares_1);  //  0.517839434674392 , cost of one share of 1 , if starting at 0,0
        if let Some(market) = market_database.get_mut(&market_1) {                                        // the ai added this line, idk what it does
            market.update(desired_shares_2);                                                      // skip ahead to after the 2nd purchase
        }                 
        let prices3 = market_database.get(&market_1).unwrap().show_inst_prices();                 // [0.3286750579857619, 0.6713249420142381] .into_raw_vec()
        let cost3 = market_database.get(&market_1).unwrap().query_update_cost(desired_shares_3);  // 10.369663437505675 , cost of 17 shares of 1, if starting at 1,6

        // test for accuracy...
        let expected_prices3 = vec![0.3286750579857619, 0.6713249420142381];
        assert!(prices3.iter().zip(expected_prices3.iter()).all(|(a, b)| (a - b).abs() < 1e-10));
        assert!((cost1 - 0.517839434674392).abs() < 1e-10);
        assert!((cost3 - 10.369663437505675).abs() < 1e-10);
    }

    // Test identical market database 
    #[test]
    fn test_market_database_contents() {
        let (_, mut market_database) = setup();

        let expected_market_database = "{\"7304df381d951a6edf08f769f9812d23369a5a8744cb7c390c41149fd859842e\": Market { title: \"Vance.2028\", state: 1, b: 7.001, trading_fee: 0.005, description: \"Will JD Vance win the presidency in 2028?\", tags: [\"politics\", \"elections\", \"usa\", \"president\"], size: 198, tau_from_now: 5, owner_ad: \"1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa\", d_axis: [\"f3b197085705c7f879616178debf308336f3df1237c25bef1ba5988f9bb987ca\"], cube_f: [0, 0], shape=[2], strides=[1], layout=CFcf (0xf), dynamic ndim=1, shares: [0.0, 0.0], shape=[2], strides=[1], layout=CFcf (0xf), dynamic ndim=1, f_prices: [0.0, 0.0], shape=[2], strides=[1], layout=CFcf (0xf), dynamic ndim=1, treasury: 4.852723411100177 }, \"4ef8eeb89996bf3b40d04c28e1495c506417f4a3526fe96c829e882fbce1c8f7\": Market { title: \"MultiDim-demo\", state: 1, b: 3.2, trading_fee: 0.005, description: \"Will JD Vance win the presidency in 2028?\", tags: [\"politics\", \"elections\", \"usa\", \"president\"], size: 419, tau_from_now: 5, owner_ad: \"1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa\", d_axis: [\"f3b197085705c7f879616178debf308336f3df1237c25bef1ba5988f9bb987ca\", \"e674cf4e6decbcb090542c2cce9984f4ffeafaa1f351d1cec56c2cf2aa188d62+ddfa2d27c2028cefd5a343cceafe2bf0e5d8b659b0d9980b973404b6012bf8bf\", \"917b6d9f57d1e1a19ba0d8ac15f652def9929c25c86234e695022db7b665583f\"], cube_f: [[[0, 0],\n  [0, 0],\n  [0, 0]],\n\n [[0, 0],\n  [0, 0],\n  [0, 0]]], shape=[2, 3, 2], strides=[6, 2, 1], layout=Cc (0x5), dynamic ndim=3, shares: [[[0.0, 0.0],\n  [0.0, 0.0],\n  [0.0, 0.0]],\n\n [[0.0, 0.0],\n  [0.0, 0.0],\n  [0.0, 0.0]]], shape=[2, 3, 2], strides=[6, 2, 1], layout=Cc (0x5), dynamic ndim=3, f_prices: [[[0.0, 0.0],\n  [0.0, 0.0],\n  [0.0, 0.0]],\n\n [[0.0, 0.0],\n  [0.0, 0.0],\n  [0.0, 0.0]]], shape=[2, 3, 2], strides=[6, 2, 1], layout=Cc (0x5), dynamic ndim=3, treasury: 7.951701279321601 }}";
        let actual_market_database = format!("{:?}", market_database);

        // println!("{:?}", actual_market_database); // for debugging

        assert_eq!(actual_market_database, expected_market_database, "Market database does not match expected content");
    }
}
