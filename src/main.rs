#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]
#![allow(unused_imports)]

mod chain_objects;
mod crypto;
mod custom_math;

use custom_math::nice_matrix::{LabeledMatrix, BinaryLabeledMatrix};
use custom_math::get_weight;
use custom_math::outcome_consensus::*;

fn main() {
    let m1 = generate_sample_matrix(1);
    // println!("{:#}", m1);

    // let rep = democracy_rep(&m1);
    // println!("{:#}", rep);

    let scl = all_binary(&m1);
    println!("{:#}", scl);

    // let outcomes = get_decision_outcomes(&m1, &rep, &scl, &Some(true));
    // println!("{:#}", outcomes);

    // let r = m1.row_labels.len();
    // let c = m1.col_labels.len();
    // println!("{:?}", r);
    // println!("{:?}", c);

    // let large_outcome = get_reward_weights(&m1, None, &scl, 0.10, 0.20, true);
    // println!("{:#?}", large_outcome);

    // println!("{:?}", fast_rep(&[4.0, 5.0, 6.0, 7.0, 8.0]));

    // let large_outcome = get_reward_weights(&m1, Some(&fast_rep(&[4.0, 4.0, 4.0, 1.0, 1.0, 1.0])), &scl, 0.10, 0.20, true);
    
    // let large_outcome = get_reward_weights(&m1,
    //                         Some(&fast_rep(&[4.0, 3.0, 4.0, 1.0, 1.0, 1.0])),
    //                         &scl, 0.10, 0.20, true);

    // println!("{:?}", large_outcome);

    let factory1 = factory(&m1, &scl, Some(&fast_rep(&[4.0, 3.0, 4.0, 1.0, 1.0, 1.0])), None, None, Some(true), None);

    // println!("{:#?}", factory1.agents);
    // println!("{:#?}", factory1.decisions);
    // println!("{:#?}", factory1.original);
    // println!("{:#?}", factory1.participation);
    // println!("{:#?}", factory1.certainty);


}
