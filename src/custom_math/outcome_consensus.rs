use nalgebra::{DMatrix, DVector};
use std::vec;

use super::nice_matrix::{LabeledMatrix, BinaryLabeledMatrix};
use super::{weighted_median, weighted_prin_comp, re_weight, get_weight};

pub fn generate_sample_matrix(size: i32) -> LabeledMatrix {
    match size {
        1 => generate_sample_matrix_6x4(),
       // 2 => generate_sample_matrix_24x4(),
        _ => panic!("Invalid size parameter. Use 1 for 6x4 matrix or 2 for 12x4 matrix"),
    }
}

fn generate_sample_matrix_6x4() -> LabeledMatrix {
    let m1 = DMatrix::from_row_slice(6, 4, &[
        1.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0,
        0.0, 0.0, 1.0, 1.0
    ]);

    let row_labels = vec!["pure truth 1".to_string(),
                          "half-liar 1".to_string(),
                          "pure truth 2".to_string(),
                          "half-liar 2".to_string(),
                          "pure liar 1".to_string(),
                          "pure liar 2".to_string() ];

    let col_labels = vec!["D.1".to_string(), "D.2".to_string(), "D.3".to_string(), "D.4".to_string()];

    LabeledMatrix::new(m1, row_labels, col_labels)
}


pub fn democracy_rep(vote_matrix: &LabeledMatrix) -> LabeledMatrix {
    // a fast way to generate the reputation vector
    // (not an endorsement or critique of any political system)

    let names = vote_matrix.row_labels.clone();
    let q_people = names.len();
    let share: f64 = 1.0 / q_people as f64;

    let m1 = DMatrix::from_element(q_people, 1, share);

    let col_labels = vec!["Rep".to_string()]; // only 1 column

    LabeledMatrix::new(m1, names, col_labels)
}

pub fn fast_rep(input: &[f64]) -> LabeledMatrix {
    // a fast way to generate the reputation vector
    // Takes a slice of f64 values and normalizes them to sum to 1

    let sum: f64 = input.iter().sum();
    let normalized: Vec<f64> = input.iter().map(|x| x / sum).collect();
    
    // Create matrix from normalized values (as column vector)
    let rep_out = DMatrix::from_column_slice(normalized.len(), 1, &normalized);

    // Generate row labels like "person.1", "person.2", etc.
    let names: Vec<String> = (1..=input.len()).map(|i| format!("person.{}", i)).collect();
    let col_labels = vec!["Rep".to_string()]; // only 1 column

    LabeledMatrix::new(rep_out, names, col_labels)
}


pub fn all_binary(vote_matrix: &LabeledMatrix) -> BinaryLabeledMatrix {
    // a fast way to declare all decisions binary, not scaled

    let dec_names = vote_matrix.col_labels.clone();

    let out_matrix = DMatrix::from_element(1, dec_names.len(), false); // 

    let row_label = vec!["Scaled".to_string()]; // only 1 column

    BinaryLabeledMatrix::new(out_matrix, row_label, dec_names)
}

pub fn get_decision_outcomes(
    vote_matrix: &LabeledMatrix,
    rep: &LabeledMatrix,
    scaled_index: &BinaryLabeledMatrix,
    verbose: &Option<bool>
) -> LabeledMatrix {
    // this is RAW and pre-caught, it is useful for the votematrix calculations

    let verbose = verbose.unwrap_or(false);

    let rep_row = &rep.data.transpose();

    let mut decision_outcomes = rep_row.clone_owned() * &vote_matrix.data; // basic multiplication, gets us binaries

    // for the scaled decisions we need to take the weighted median
    for i in 0..vote_matrix.data.ncols() {
        if scaled_index.data[(0, i)] {
            let col = vote_matrix.data.column(i);
            let col_dyn = DVector::from_column_slice(col.as_slice());
            let rep_row_dyn = DVector::from_row_slice(rep.data.as_slice());
            decision_outcomes[(0, i)] = weighted_median(&col_dyn, &rep_row_dyn);
        }
    }

    if verbose {
        println!("****************************************************");
        println!("Decision Outcomes (Raw):");
        println!("{}", decision_outcomes);
    }

    // Convert decision_outcomes to a DMatrix with dynamic row dimensions
    let decision_outcomes_dyn = DMatrix::from_row_slice(1, decision_outcomes.len(), decision_outcomes.as_slice());

    LabeledMatrix {
        data: decision_outcomes_dyn,
        row_labels: vec!["Outcomes".to_string()],
        col_labels: vote_matrix.col_labels.clone(),
    }
}

pub fn catch_decision_outcomes(
    decision_row: &LabeledMatrix,
    scaled_index: &BinaryLabeledMatrix,
    tol: &f64
) -> LabeledMatrix {
    let c = decision_row.data.ncols();

    let mut outs = DMatrix::zeros(1, c);

    for i in 0..c {
        if !scaled_index.data[(0, i)] {
            outs[(0, i)] = catch_tl(decision_row.data[(0, i)], tol);
        } else {
            outs[(0, i)] = decision_row.data[(0, i)];
        }
    }

    LabeledMatrix {
        data: outs,
        row_labels: decision_row.row_labels.clone(),
        col_labels: decision_row.col_labels.clone(), 
    }
}

fn catch_tl(x: f64, tolerance: &f64) -> f64 {
    // x is the ConoutRAW numeric, Tolerance is the length of the midpoint corresponding to .5 
    // The purpose here is to handle rounding for Binary Decisions (Scaled Decisions use weighted median)
    if x < (0.5 - (tolerance / 2.0)) {
        0.0
    } else if x > (0.5 + (tolerance / 2.0)) {
        1.0
    } else {
        0.5
    }
}

fn score_reflect(component: &DVector<f64>, previous_reputation: &DVector<f64>) -> DVector<f64> {
    // Takes an adjusted score, and breaks it at a median point.

    // Only if there actually is a median point.
    let unique_count = component.len();
    if unique_count <= 2 {
        return component.clone();
    }

    // If the Reps are calculated directly, there is an exploit in generating "sacrificial variance" and passing it to a teammate.
    // Although initially the sac-var is zero-sum (pointless), after reweighting it can become helpful to attackers.

    // Instead, a simple adjustment
    let median_factor = weighted_median(component, previous_reputation);

    // By how much does the component exceed the Median?
    let reflection = component - DVector::from_element(component.len(), median_factor);
    // Where does the Component exceed the Median?
    let excessive = reflection.map(|x| x > 0.0);

    let mut adj_prin_comp = component.clone();

    for i in 0..adj_prin_comp.len() {
        if excessive[i] {
            adj_prin_comp[i] -= reflection[i] * 0.5;  // mixes in some of the old method
        }
    }
    
    adj_prin_comp
}

#[derive(Debug)]
pub struct RewardWeightsResult {
    pub first_l: LabeledMatrix,
    pub old_rep: LabeledMatrix,
    pub this_rep: LabeledMatrix,
    pub smooth_rep: LabeledMatrix,
}

pub fn fill_na(
    m_na: &LabeledMatrix,
    rep: Option<&LabeledMatrix>,
    scaled_index: &BinaryLabeledMatrix,
    catch_p: f64,
    verbose: bool,
) -> LabeledMatrix {
    // Uses existing data and reputations to fill missing observations
    // Essentially a weighted average using all available non-NA data
    
    let mut m_new = m_na.clone();
    
    // Check if there are any NaN values
    let has_nan = m_na.data.iter().any(|&x| x.is_nan());
    
    if has_nan {
        if verbose {
            println!("Missing Values Detected. Beginning presolve using available values.");
        }
        
        // Get reputation vector if not provided
        let dem_rep = democracy_rep(m_na);
        let rep = rep.unwrap_or(&dem_rep);
        
        // Get initial decision outcomes using available data
        let decision_outcomes_raw = get_decision_outcomes(m_na, rep, scaled_index, &Some(verbose));
        
        // Catch the raw outcomes to get final values for filling
        let decision_outcomes_caught = catch_decision_outcomes(&decision_outcomes_raw, scaled_index, &catch_p);
        
        // Create a mask for NaN values
        let nan_mask = m_na.data.map(|x| x.is_nan());
        
        // Replace NaN values with the corresponding caught decision outcomes
        for i in 0..m_new.data.nrows() {
            for j in 0..m_new.data.ncols() {
                if nan_mask[(i, j)] {
                    m_new.data[(i, j)] = decision_outcomes_caught.data[(0, j)];
                }
            }
        }
        
        if verbose {
            println!("Missing Values Filled");
            println!("Raw Results:");
            println!("{}", m_new);
        }
    }
    
    m_new
}

pub fn get_reward_weights(
    vote_matrix: &LabeledMatrix,
    rep: Option<&LabeledMatrix>,
    scaled_index: &BinaryLabeledMatrix,
    alpha: f64,
    tol: f64,
    verbose: bool,
) -> RewardWeightsResult {
    let r = vote_matrix.row_labels.len();
    let c = vote_matrix.col_labels.len();
    
    // if no rep vector is given, divide rep equally
    let dem_rep = democracy_rep(vote_matrix);
    let old_rep = rep.unwrap_or(&dem_rep);  

    if verbose {
        println!("****************************************************");
        println!("Begin 'GetRewardWeights'");
        println!("Inputs...");
        println!("Matrix:");
        println!("{}", vote_matrix);
        println!("Reputation:");
        println!("{}", old_rep);
    }

    // ignore rep if there is only one person
    // otherwise, transpose the matrix (so we can use it for multiplication)
    let rep_row = if old_rep.data.nrows() == 1 {  
        old_rep.data.clone()
    } else {
        old_rep.data.transpose()
    };
    
    // also: add a copy in 2nd type, to feed it to wPCA
    let rep_vec = DVector::from(old_rep.data.column(0));

    // weighted Principal Components Analysis
    let results = weighted_prin_comp(&vote_matrix.data, Some(&rep_vec), verbose);
    let first_score = results.0;
    let first_loading = results.1;

    if verbose {
        println!("First Loading:");
        println!("{}", first_loading);
        println!("First Score:");
        println!("{}", first_score);
    }

    let mut new_rep = old_rep.clone();

    // if there was 100% agreement among everyone, then we are done: reputations do not change
    if first_score.abs().sum() != 0.0 {

        // Choosing among two score perspectives

        // PCA has two interpretations -- each opposite of each other.
        // We need to pick the correct interpretation.

        // Temporarily we will proceed with both.
        let score_min = DVector::from_element(first_score.len(), first_score.min());
        let score_max = DVector::from_element(first_score.len(), first_score.max());

        let option1 =  &first_score + score_min;
        let option2 = (&first_score - score_max).abs();

        let new_score_1 = score_reflect(&option1, &rep_vec);
        let new_score_2 = score_reflect(&option2, &rep_vec);

            if verbose {
                println!("****************************************************");
                println!("Score Analysis");
                println!("Option 1:");
                println!("{}", LabeledMatrix::new(
                    DMatrix::from_column_slice(option1.len(), 1, option1.as_slice()),
                    vote_matrix.row_labels.clone(),
                    vec!["Score".to_string()]
                ));
                println!("Reputation Vector:");
                println!("{}", LabeledMatrix::new(
                    DMatrix::from_column_slice(rep_vec.len(), 1, rep_vec.as_slice()),
                    vote_matrix.row_labels.clone(),
                    vec!["Rep".to_string()]
                ));
                println!("Score Options:");
                println!("Score 1:");
                println!("{}", LabeledMatrix::new(
                    DMatrix::from_column_slice(new_score_1.len(), 1, new_score_1.as_slice()),
                    vote_matrix.row_labels.clone(),
                    vec!["Score".to_string()]
                ));
                println!("Score 2:");
                println!("{}", LabeledMatrix::new(
                    DMatrix::from_column_slice(new_score_2.len(), 1, new_score_2.as_slice()),
                    vote_matrix.row_labels.clone(),
                    vec!["Score".to_string()]
                ));
            }

        // Now... how do we choose between Rep1 and Rep2.

        // This is method 6 , in the R code , my favorite.
        //    Use Old Reputation to calculate outcomes.
        //    Examine each voter's agreement with these outcomes.
        //    Collapse this agreement using this period's Loading.

        // Outcomes according to the previous reputation vector
        let oldrep_out_raw = get_decision_outcomes(vote_matrix, old_rep, scaled_index, &Some(verbose));  // matrix multiplication
        let oldrep_outcomes = catch_decision_outcomes( &oldrep_out_raw, scaled_index, &tol );              // Bin non-scaled modes into < 0, .5, 1 > , using Catch()

        // Create a matrix repeating the oldrep outcomes
        // in R this is so easy ... Distance <- abs(  M - ( array(1, dim(M)) %*% diag(Old_f) )  ) .. does in 1 line what it takes rust 25 lines to do
        let oldrep_outcomes_row = oldrep_outcomes.data.row(0);
        let mut outcome_matrix = DMatrix::zeros(r, c);
        for i in 0..r {
            for j in 0..c {
                outcome_matrix[(i, j)] = oldrep_outcomes_row[j];   // just stack the c-wide outcomes vector on itself, r times
            }
        }

        if verbose {
            println!("****************************************************");
            println!("Old Reputation Outcomes:");
            println!("{}", oldrep_outcomes);
            println!("Outcome Matrix:");
            println!("{}", outcome_matrix);
        }

        // Build distance matrix: absolute value of difference between "Group Outcomes (using Old)" and "Individual Outcomes"  
        let difference = vote_matrix.data.clone() - outcome_matrix;  // Calculate the difference between m.data and repeated_old_caught
        let distance = difference.abs();                             // Calculate the absolute values of the difference
        
        if verbose {
            println!("****************************************************");
            println!("Difference Matrix:");
            println!("{}", difference);
            println!("Distance Matrix:");
            println!("{}", distance);
        }

        // Separately, build index of question-cohesion
        let mut dissent = first_loading.abs();     // FirstLoading is higher when people disagree, lower when people agree, zero for perfect agreement.
        // mainstream is the opposite of dissent, but with zeroes removed first (zeros = irrelevant, not infinitely important)
        let mainstream_preweight = dissent.map(|x| if x != 0.0 { 1.0 / x } else { f64::NAN });
        let mainstream = re_weight(&mainstream_preweight);

        if verbose {
            println!("****************************************************");
            println!("Dissent and Mainstream Analysis");
            println!("Dissent Matrix:");
            println!("{}", dissent);
            println!("Mainstream Matrix:");
            println!("{}", mainstream);
        }

        // integrate the two
        let non_compliance = &distance * &mainstream;
        let max_non_compliance = non_compliance.max();
        let compliance = non_compliance.map(|x| (x - max_non_compliance).abs());

        println!("****************************************************");
        println!("Non-Compliance Matrix:");
        println!("{}", non_compliance);
        println!("Compliance Matrix:");
        println!("{}", compliance);

        // Compare - Which score has more error when it is matched up against compliance?
        // in R this is just:  Choice1 <- sum( (GetWeight(NewScore_1) - Compliance)^2 )

        let weighted_score_matrix_1 = get_weight(&new_score_1); 
        let difference_1 = &weighted_score_matrix_1 - &compliance;
        let choice1 = difference_1.iter().map(|&x| x.powi(2)).sum::<f64>(); // now repeat...
        let weighted_score_matrix_2 = get_weight(&new_score_2);
        let difference_2 = &weighted_score_matrix_2 - &compliance;
        let choice2 = difference_2.iter().map(|&x| x.powi(2)).sum::<f64>(); 

        // // When RefInd becomes positive, switch to Set2.
        let ref_ind = choice1 - choice2; // When choice1 has greater error, this will be positive.

        //  = (new_score_2 * (old_rep)/ mean(old_rep)) // for some reason, in rust this takes 20 lines of code
        let rep_mean = old_rep.data.mean();
        let scaled_rep = old_rep.data.map(|x| x / rep_mean); 
        // ^  Rep/mean(Rep) is a correction ensuring Reputation is additive. Therefore, no advantage can be obtained by splitting/combining Reputation into single/multiple accounts.
        
        println!("****************************************************");
        println!("Scaled Reputation and New Scores");
        println!("Scaled Reputation:");
        println!("{}", scaled_rep);
        println!("New Score Option 1:");
        println!("{}", new_score_1);
        println!("New Score Option 2:");
        println!("{}", new_score_2);

        let rep1 = get_weight(&new_score_1.component_mul(&scaled_rep));
        let rep2 = get_weight(&new_score_2.component_mul(&scaled_rep));

        let new_rep_dyn = if ref_ind < 0.0 {
            DMatrix::from_column_slice(r, 1, rep1.as_slice())
        } else {
            DMatrix::from_column_slice(r, 1, rep2.as_slice())
        };

        println!("****************************************************");
        println!("Reputation Options");
        println!("Reputation Option 1:");
        println!("{}", rep1);
        println!("Reputation Option 2:");
        println!("{}", rep2);
        println!("Selected New Reputation:");
        println!("{}", new_rep_dyn);

        new_rep.data = new_rep_dyn;
    }

    // We will blend the old and new, such that the effect is not too jarring
    let mut smoothed_r = new_rep.clone();
    smoothed_r.data =  &new_rep.data * alpha  + &old_rep.data * (1.0 - alpha);

    println!("****************************************************");
    println!("Final Smoothed Reputation:");
    println!("{}", smoothed_r);

    // Format to send out
    let first_loading_dyn_matrix = DMatrix::from_column_slice(c, 1, first_loading.as_slice());
    let out_first_l = LabeledMatrix::new(
        first_loading_dyn_matrix,
        vote_matrix.col_labels.clone(),
        vec!["First.Loading".to_string()]
    );

    RewardWeightsResult {
        first_l: out_first_l,
        old_rep: old_rep.clone(),
        this_rep: new_rep.clone(),
        smooth_rep: smoothed_r,
    }
}

#[derive(Debug)]
pub struct FactoryResult {
    pub original: LabeledMatrix,
    pub filled: LabeledMatrix,
    pub agents: LabeledMatrix,
    pub decisions: LabeledMatrix,
    pub participation: f64,
    pub certainty: f64,
}

pub fn factory(
    m0: &LabeledMatrix,
    scaled_index: &BinaryLabeledMatrix,
    rep: Option<&LabeledMatrix>,
    catch_p: Option<f64>,
    max_row: Option<usize>,
    verbose: Option<bool>,
    rbcr: Option<i32>
) -> FactoryResult {
    // Main Routine
    let verbose = verbose.unwrap_or(false);
    let catch_p = catch_p.unwrap_or(0.2);
    let _max_row = max_row.unwrap_or(5000);
    let _rbcr = rbcr.unwrap_or(6);

    // Handle missing values
    let filled = fill_na(m0, rep, scaled_index, catch_p, verbose);

    // Consensus - Row Players
    // New Consensus Reward
    let player_info = get_reward_weights(
        &filled,
        rep,
        scaled_index,
        0.1, // alpha
        catch_p,
        verbose
    );

    let adj_loadings = &player_info.first_l;

    // Column Players (The Decision Creators)
    // Calculation of Reward for Decision Authors
    // Consensus - "Who won?" Decision Outcome
    let decision_outcomes_raw = get_decision_outcomes(&filled, &player_info.smooth_rep, scaled_index, &Some(verbose));

    // The Outcome Itself
    let decision_outcomes_adj = catch_decision_outcomes(&decision_outcomes_raw, scaled_index, &catch_p);

    // Quality of Outcomes - is there confusion?
    let mut certainty = vec![0.0; filled.data.ncols()];
    
    // For each Decision
    for i in 0..filled.data.ncols() {
        // Sum of reputations which met the condition that they voted for the outcome selected for this Decision
        let mut sum = 0.0;
        for j in 0..filled.data.nrows() {
            if (decision_outcomes_adj.data[(0, i)] - filled.data[(j, i)]).abs() < 1e-10 {
                sum += player_info.smooth_rep.data[(j, 0)];
            }
        }
        certainty[i] = sum;
    }

    let avg_certainty = certainty.iter().sum::<f64>() / certainty.len() as f64;

    // Stability of Voter Reports
    let nrows = filled.data.nrows();
    let ncols = filled.data.ncols();
    let mut resolveable_f64 = DMatrix::zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            resolveable_f64[(i, j)] = if filled.data[(i, j)] != 0.5 { 1.0 } else { 0.0 };
        }
    }
    // Calculate max clarity directly using resolveable_f64
    let smooth_rep_data = player_info.smooth_rep.data.clone();
    let smooth_rep_slice = smooth_rep_data.as_slice();
    let smooth_rep_transposed = DMatrix::from_row_slice(1, smooth_rep_slice.len(), smooth_rep_slice);
    let max_clarity = smooth_rep_transposed * resolveable_f64;
    let certainty_matrix = DMatrix::from_row_slice(1, certainty.len(), &certainty);
    let clarity = max_clarity.component_mul(&certainty_matrix);
    let con_reward = get_weight(&DVector::from_row_slice(clarity.as_slice()));

    if verbose {
        println!("****************************************************");
        println!("Decision Outcomes Successfully Calculated");
        println!("Raw Outcomes:");
        println!("{}", decision_outcomes_raw);
        println!("Certainty Values:");
        println!("{}", LabeledMatrix::new(
            DMatrix::from_row_slice(1, certainty.len(), &certainty),
            vec!["Certainty".to_string()],
            filled.col_labels.clone()
        ));
        println!("Author Payout Factors:");
        println!("{}", LabeledMatrix::new(
            DMatrix::from_row_slice(1, con_reward.len(), con_reward.as_slice()),
            vec!["Payout".to_string()],
            filled.col_labels.clone()
        ));
    }

    // Participation
    let participation = 1.0; // TODO: Implement participation tracking

    // Format output matrices
    let mut agent_data = DMatrix::zeros(filled.data.nrows(), 7);
    for i in 0..filled.data.nrows() {
        agent_data[(i, 0)] = player_info.old_rep.data[(i, 0)];
        agent_data[(i, 1)] = player_info.this_rep.data[(i, 0)];
        agent_data[(i, 2)] = player_info.smooth_rep.data[(i, 0)];
        // Columns 3-6 would be NArow, ParticipationR, RelativePart, RowBonus
    }

    let agent_labels = filled.row_labels.clone();
    let agent_col_labels = vec![
        "OldRep".to_string(),
        "ThisRep".to_string(), 
        "SmoothRep".to_string(),
        "NArow".to_string(),
        "ParticipationR".to_string(),
        "RelativePart".to_string(),
        "RowBonus".to_string()
    ];

    let agents = LabeledMatrix::new(agent_data, agent_labels, agent_col_labels);

    let mut decision_data = DMatrix::zeros(8, filled.data.ncols());
    for i in 0..filled.data.ncols() {
        decision_data[(0, i)] = adj_loadings.data[(i, 0)];
        decision_data[(1, i)] = decision_outcomes_raw.data[(0, i)];
        decision_data[(2, i)] = con_reward[i];
        decision_data[(3, i)] = certainty[i];
        // Rows 4-7 would be NAs Filled, ParticipationC, Author Bonus, DecisionOutcome.Final
    }

    let decision_row_labels = vec![
        "First Loading".to_string(),
        "DecisionOutcomes.Raw".to_string(),
        "Consensus Reward".to_string(),
        "Certainty".to_string(),
        "NAs Filled".to_string(),
        "ParticipationC".to_string(),
        "Author Bonus".to_string(),
        "DecisionOutcome.Final".to_string()
    ];

    let decisions = LabeledMatrix::new(
        decision_data,
        decision_row_labels,
        filled.col_labels.clone()
    );

    FactoryResult {
        original: m0.clone(),
        filled,
        agents,
        decisions,
        participation,
        certainty: avg_certainty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_democracy_rep() {
        let matrix = generate_sample_matrix(1);
        let rep = democracy_rep(&matrix);
        
        assert_eq!(rep.data.nrows(), 6);
        assert_eq!(rep.data.ncols(), 1);
        assert_relative_eq!(rep.data[(0, 0)], 1.0/6.0, epsilon = 1e-10);
        assert_relative_eq!(rep.data[(1, 0)], 1.0/6.0, epsilon = 1e-10);
        assert_relative_eq!(rep.data[(2, 0)], 1.0/6.0, epsilon = 1e-10);
        assert_relative_eq!(rep.data[(3, 0)], 1.0/6.0, epsilon = 1e-10);
        assert_relative_eq!(rep.data[(4, 0)], 1.0/6.0, epsilon = 1e-10);
        assert_relative_eq!(rep.data[(5, 0)], 1.0/6.0, epsilon = 1e-10);
    }


    #[test]
    fn test_fast_rep() {
    let input = vec![0.4, 0.5, 0.6, 0.4, 0.5];
    let result = fast_rep(&input);
    
    // Check dimensions
    assert_eq!(result.data.nrows(), 5);
    assert_eq!(result.data.ncols(), 1);
    
    // Check normalization (should sum to 1)
    let sum: f64 = result.data.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    
    // Check individual values
    let total = 0.4 + 0.5 + 0.6 + 0.4 + 0.5;
    assert_relative_eq!(result.data[(0, 0)], 0.4/total, epsilon = 1e-10);
    assert_relative_eq!(result.data[(1, 0)], 0.5/total, epsilon = 1e-10);
    assert_relative_eq!(result.data[(2, 0)], 0.6/total, epsilon = 1e-10);
    assert_relative_eq!(result.data[(3, 0)], 0.4/total, epsilon = 1e-10);
    assert_relative_eq!(result.data[(4, 0)], 0.5/total, epsilon = 1e-10);
    
    // Check labels
    assert_eq!(result.row_labels, vec!["person.1", "person.2", "person.3", "person.4", "person.5"]);
    assert_eq!(result.col_labels, vec!["Rep"]);
}


    #[test]
    fn test_all_binary() {
        let matrix = generate_sample_matrix(1);
        let binary = all_binary(&matrix);
        
        assert_eq!(binary.data.nrows(), 1);
        assert_eq!(binary.data.ncols(), 4);
        assert!(!binary.data[(0, 0)]);
        assert!(!binary.data[(0, 1)]);
        assert!(!binary.data[(0, 2)]);
        assert!(!binary.data[(0, 3)]);
    }

    #[test]
    fn test_catch_tl() {
        assert_eq!(catch_tl(0.2, &0.2), 0.0);
        assert_eq!(catch_tl(0.5, &0.2), 0.5);
        assert_eq!(catch_tl(0.8, &0.2), 1.0);
        assert_eq!(catch_tl(0.14, &0.70), 0.0);
        assert_eq!(catch_tl(0.16, &0.70), 0.5);
    }

    #[test]
    fn test_get_decision_outcomes() {
        let matrix = generate_sample_matrix(1);
        let rep = democracy_rep(&matrix);
        let binary = all_binary(&matrix);
        
        let outcomes = get_decision_outcomes(&matrix, &rep, &binary, &Some(false));

        println!("****************************************************");
        println!("Test Outcomes:");
        println!("{}", outcomes);
        
        assert_eq!(outcomes.data.nrows(), 1);
        assert_eq!(outcomes.data.ncols(), 4);
        assert_relative_eq!(outcomes.data[(0, 0)], 0.66666666666666, epsilon = 1e-10);
        assert_relative_eq!(outcomes.data[(0, 1)], 0.5, epsilon = 1e-10);
        assert_relative_eq!(outcomes.data[(0, 2)], 0.5, epsilon = 1e-10);
        assert_relative_eq!(outcomes.data[(0, 3)], 0.33333333333333, epsilon = 1e-10);
    }

    #[test]
    fn test_catch_decision_outcomes() {
        let matrix = generate_sample_matrix(1);
        let rep = democracy_rep(&matrix);
        let binary = all_binary(&matrix);
        
        let outcomes = get_decision_outcomes(&matrix, &rep, &binary, &Some(false));
        let caught = catch_decision_outcomes(&outcomes, &binary, &0.2);
        
        assert_eq!(caught.data.nrows(), 1);
        assert_eq!(caught.data.ncols(), 4);
        assert_eq!(caught.data[(0, 0)], 1.0);
        assert_eq!(caught.data[(0, 1)], 0.5);
        assert_eq!(caught.data[(0, 2)], 0.5);
        assert_eq!(caught.data[(0, 3)], 0.0);
    }


    #[test]
    fn test_factory() {
        let matrix = generate_sample_matrix(1);
        let binary = all_binary(&matrix);
        let result = factory(&matrix, &binary, None, None, None, None, None);

        // Test dimensions and basic structure
        assert_eq!(result.original.data, matrix.data);
        assert_eq!(result.filled.data, matrix.data);
        assert_eq!(result.agents.data.nrows(), 6);
        assert_eq!(result.agents.data.ncols(), 7);
        assert_eq!(result.decisions.data.nrows(), 8);
        assert_eq!(result.decisions.data.ncols(), 4);
        
        // Test participation and certainty
        assert_eq!(result.participation, 1.0);
        assert!(result.certainty > 0.0 && result.certainty <= 1.0);

        // Test with custom reputation
        let rep_vec = vec![0.4, 0.1, 0.2, 0.1, 0.1, 0.1];
        let rep = fast_rep(&rep_vec);
        let result_with_rep = factory(&matrix, &binary, Some(&rep), None, None, None, None);
        
        // Verify reputation was used
        assert_relative_eq!(result_with_rep.agents.data[(0, 0)], 0.4, epsilon = 1e-10);
        assert_relative_eq!(result_with_rep.agents.data[(1, 0)], 0.1, epsilon = 1e-10);
    }
}
