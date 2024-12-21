use nalgebra::{DMatrix, DVector, SVD};

const NA: f64 = f64::NAN;

pub fn mean_na(vec: &[f64]) -> DMatrix<f64> {
    // fills missing values with the sample average
    let valid_values: Vec<f64> = vec.iter().filter(|&&x| !x.is_nan()).cloned().collect();
    let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
    DMatrix::from_row_slice(1, vec.len(), &vec.iter().map(|&x| if x.is_nan() { mean } else { x }).collect::<Vec<f64>>())
}

pub fn get_weight_m(vec: &DVector<f64>, add_mean: bool) -> DVector<f64> {
    // Takes a Vector, absolute value, then proportional linear deviance from 0.
    let mut new = vec.abs(); // absolute value

    if add_mean {
        let mean = new.mean();        // add mean, if necessary
        new = new.map(|x| x + mean);
    }

    if new.sum() == 0.0 {          // if it is all zeros, add +1 to everything
        new.fill(1.0);
    }
    new /= new.sum();
    new
}


pub fn get_weight(vec: &DVector<f64>) -> DVector<f64> {
    get_weight_m(vec, false)
}

pub fn catch_tl(x: f64, tolerance: &f64) -> f64 {
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

pub fn catch(x: f64) -> f64 {
    // default catch function, tolerance = 0
    catch_tl(x, &0.0)
}


pub fn weighted_median(values: &DVector<f64>, weights: &DVector<f64>) -> f64 {
    assert_eq!(values.len(), weights.len(), "Values and weights must have the same length");

    let mut sorted: Vec<_> = values.iter().zip(weights.iter()).collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap());

    let total_weight: f64 = weights.iter().sum();
    let mut cumulative_weight = 0.0;
    let half_weight = total_weight / 2.0;

    for (i, (&value, &weight)) in sorted.iter().enumerate() {
        cumulative_weight += weight;
        if cumulative_weight >= half_weight {
            if cumulative_weight == half_weight && i < sorted.len() - 1 {
                return (value + sorted[i + 1].0) / 2.0;
            } else {
                return value;
            }
        }
    }

    *sorted.last().unwrap().0
}


pub fn influence(weight: &DMatrix<f64>) -> DMatrix<f64> {
    //  Takes a normalized Vector (one that sums to 1), and computes relative strength of the indicators.
    //  this is because by-default the conformity of each Author and Judge is expressed relatively.
    let len = weight.len() as f64;
    let expected = 1.0 / len;
    
    weight.map(|w| w / expected)
}


pub fn re_weight(vector_in: &DVector<f64>) -> DVector<f64> {
    let mut out = vector_in.clone();
    
    // Check if all values are positive
    if out.iter().any(|&x| x.is_sign_negative() && !x.is_nan()) {
        println!("Warning: Expected all positive.");
    }
    
    // Take absolute values and handle NaN
    out.apply(|x| {
        if x.is_nan() {
            *x = 0.0;
        } else {
            *x = x.abs();
        }
    });
    
    // Handle division by zero
    if out.sum() == 0.0 {
        out.fill(1.0);
    }
    
    // Normalize
    let sum = out.sum();
    out.apply(|x| *x = *x / sum);

    out
}



pub fn weighted_prin_comp(x: &DMatrix<f64>, weights: Option<&DVector<f64>>, verbose: bool) -> (DVector<f64>, DVector<f64>) {
    let n_rows = x.nrows();
    let n_cols = x.ncols();

    let weights = match weights {
        Some(w) => {
            if w.len() != n_rows {
                panic!("Error: Weights must be equal to nrow(X)");
            }
            w.clone()
        },
        None => re_weight(&DVector::from_element(n_rows, 1.0)),
    };

    // Compute weighted mean
    let total_weight = weights.sum();
    let weighted_mean = (x.transpose() * &weights) / total_weight;

    // Center the data
    let mut centered = x.clone();
    for mut row in centered.row_iter_mut() {
        row -= &weighted_mean.transpose();
    }

    // Compute weighted covariance
    let mut weighted_centered = DMatrix::zeros(centered.nrows(), centered.ncols());
    for (i, row) in centered.row_iter().enumerate() {
        weighted_centered.set_row(i, &(row * weights[i]));
    }
    let wcvm = (centered.transpose() * weighted_centered) / total_weight;

    // SVD
    let svd = SVD::new(wcvm, true, false);
let first_loading = svd.u.as_ref().unwrap().column(0).clone_owned();
    let first_loading = svd.u.as_ref().unwrap().column(0).clone_owned();   // extract first Loading (first column of U matrix)
    let first_score = centered * &first_loading;                            // manually calculate the first Score, using the input matrix, covariance matrix, and first loading

    if verbose {
        println!("Loadings:");
        println!("{}", svd.u.as_ref().unwrap()); // the entire u matrix
        println!();
        println!("Scores:");
        println!("{}", first_score);             // all of the scores, TODO this is wrong, it is only 1 score
        println!();
    }

    (first_score, first_loading.clone())
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_mean_na() {
        assert_eq!(mean_na(&[3.0, 4.0, 6.0, 7.0, 8.0]), DMatrix::from_row_slice(1, 5, &[3.0, 4.0, 6.0, 7.0, 8.0]));
        assert_eq!(mean_na(&[3.0,  NA, 6.0, 7.0, 8.0]), DMatrix::from_row_slice(1, 5, &[3.0, 6.0, 6.0, 7.0, 8.0]));
        assert_eq!(mean_na(&[0.0, 0.0, 0.0, 1.0,  NA]), DMatrix::from_row_slice(1, 5, &[0.0, 0.0, 0.0, 1.0, 0.25]));
        assert_relative_eq!(mean_na(&[0.0, 0.0, NA, 1.0, NA])[(0, 2)], 0.3333333333333333, epsilon = 1e-10);
    }


    #[test]
    fn test_get_weight() {
        assert_eq!(
            get_weight(&DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0])),
            DVector::from_column_slice(&[0.25, 0.25, 0.25, 0.25])
        );
        assert_eq!(
            get_weight(&DVector::from_column_slice(&[10.0, 10.0, 10.0, 10.0])),
            DVector::from_column_slice(&[0.25, 0.25, 0.25, 0.25])
        );
        assert_relative_eq!(
            get_weight(&DVector::from_column_slice(&[4.0, 5.0, 6.0, 7.0]))[0],
            0.18181818181818182,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_get_weight_m() {
        let result = get_weight_m(&DVector::from_column_slice(&[4.0, 5.0, 6.0, 7.0]), true);
        assert_relative_eq!(result[(0, 0)], 0.2159090909090909, epsilon = 1e-10);
    }

    #[test]
    fn test_weighted_median() {
        assert_eq!(   weighted_median(
                &DVector::from_row_slice(&[3.0, 4.0, 5.0]),
                &DVector::from_row_slice(&[0.2, 0.2, 0.6])),  5.0);
        assert_eq!(   weighted_median(
                &DVector::from_row_slice(&[3.0, 4.0, 5.0]),
                &DVector::from_row_slice(&[0.2, 0.2, 0.5])),  5.0);
        assert_eq!(   weighted_median(
                &DVector::from_row_slice(&[3.0, 4.0, 5.0]),
                &DVector::from_row_slice(&[0.2, 0.2, 0.4])),  4.5);
    }
    

    #[test]
    fn test_influence() {
        assert_eq!(            influence(&DMatrix::from_row_slice(1, 4, &[0.25, 0.25, 0.25, 0.25])),
                                          DMatrix::from_row_slice(1, 4, &[1.0,  1.0,  1.0,  1.0]) );
        
        assert_relative_eq!(   influence(&DMatrix::from_row_slice(1, 3, &[0.3, 0.3, 0.4]))[(0, 2)],
            1.2000000000000002,epsilon = 1e-10 );
        
        assert_relative_eq!(   influence(&DMatrix::from_row_slice(1, 5, &[0.99, 0.0025, 0.0025, 0.0025, 0.0025]))[(0, 0)],
            4.949999999999999, epsilon = 1e-10);
    }


    #[test]
    fn test_re_weight() {
        assert_eq!(re_weight(&DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0])), DVector::from_column_slice(&[0.25, 0.25, 0.25, 0.25]));
        assert_eq!(re_weight(&DVector::from_column_slice(&[1.0,  NA, 1.0,  NA])), DVector::from_column_slice(&[0.5,  0.0,  0.5,  0.0]));
        assert_relative_eq!(re_weight(&DVector::from_column_slice(&[2.0, 4.0, 6.0, 12.0]))[0], 0.0833333333333333, epsilon = 1e-10);
        assert_relative_eq!(re_weight(&DVector::from_column_slice(&[2.0, 4.0,  NA, 12.0]))[0], 0.1111111111111111, epsilon = 1e-10);
    }

    #[test]
    fn test_svd() {
        let matrix1 = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let svd1 = matrix1.svd(true, true);
        assert_relative_eq!(svd1.singular_values[0], 9.525518091565104, epsilon = 1e-10);
        assert_relative_eq!(svd1.singular_values[1], 0.514300580658644, epsilon = 1e-10);
    }

    #[test]
    fn test_weighted_prin_comp() {
        let m1 = DMatrix::from_row_slice(3, 3, &[
            0.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
            0.0, 0.0, 0.0
        ]);

        let m2 = DMatrix::from_row_slice(3, 4, &[
            0.0, 0.0, 1.0, 0.7,
            1.0, 1.0, 0.0, 0.5,
            0.0, 0.0, 0.0, 0.2
        ]);

        let wg1 = DVector::from_column_slice(&[0.33333, 0.33333, 0.33333]);
        let wg2 = DVector::from_column_slice(&[0.1, 0.1, 0.8]);

        let (score, loading) = weighted_prin_comp(&m1.clone(), None, false);
        assert_relative_eq!( score[0], -0.7251092490536917, epsilon = 1e-10);
        assert_relative_eq!(loading[0], 0.6279630301995545, epsilon = 1e-10);

        let (score, loading) = weighted_prin_comp(&m1.clone(), Some(&wg1), false);
        assert_relative_eq!( score[0], -0.7251092490536917, epsilon = 1e-10);
        assert_relative_eq!(loading[0], 0.6279630301995545, epsilon = 1e-10);

        let (score, loading) = weighted_prin_comp(&m1.clone(), Some(&wg2), false);
        assert_relative_eq!( score[0], -0.27628007046829856, epsilon = 1e-10);
        assert_relative_eq!(loading[0], 0.698927408964215, epsilon = 1e-10);

        let (score, loading) = weighted_prin_comp(&m2.clone(), None, false);
        assert_relative_eq!( score[0], -0.7385623942929247, epsilon = 1e-10);
        assert_relative_eq!(loading[0], 0.6242801367887156, epsilon = 1e-10);

        let (score, loading) = weighted_prin_comp(&m2.clone(), Some(&wg2), false);
        assert_relative_eq!( score[0], -0.12696925880631366, epsilon = 1e-10);
        assert_relative_eq!(loading[0], 0.694611214418193, epsilon = 1e-10);
    }
}
