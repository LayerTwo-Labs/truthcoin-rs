use nalgebra::DMatrix;
use std::fmt;


#[derive(Debug, Clone)]
pub struct LabeledMatrix {
    pub data: DMatrix<f64>,
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
}
#[derive(Debug, Clone)]
pub struct BinaryLabeledMatrix {
    pub data: DMatrix<bool>,
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
}

impl LabeledMatrix {
    
    pub fn new(data: DMatrix<f64>, row_labels: Vec<String>, col_labels: Vec<String>) -> Self {
        assert_eq!(data.nrows(), row_labels.len(), "Row labels must match matrix rows");
        assert_eq!(data.ncols(), col_labels.len(), "Column labels must match matrix columns");
        LabeledMatrix { data, row_labels, col_labels }
    }

    pub fn get_value(&self, row_label: &str, col_label: &str) -> Option<f64> {
        let row_index = self.row_labels.iter().position(|r| r == row_label)?;
        let col_index = self.col_labels.iter().position(|c| c == col_label)?;
        Some(self.data[(row_index, col_index)])
    }

}

impl fmt::Display for LabeledMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Find the maximum length of row labels for alignment
        let max_label_len = self.row_labels.iter().map(|s| s.len()).max().unwrap_or(0);

        // Print column labels
        write!(f, "{:width$}", "", width = max_label_len + 4)?;
        for col in &self.col_labels {
            write!(f, "{:>10}", col)?;
        }
        writeln!(f)?;

        // Print rows with labels and data
        for (i, row_label) in self.row_labels.iter().enumerate() {
            write!(f, "{:width$}", row_label, width = max_label_len + 4)?;
            for j in 0..self.data.ncols() {
                write!(f, "{:10.5}", self.data[(i, j)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}


impl BinaryLabeledMatrix {
    
    pub fn new(data: DMatrix<bool>, row_labels: Vec<String>, col_labels: Vec<String>) -> Self {
        assert_eq!(data.nrows(), row_labels.len(), "Row labels must match matrix rows");
        assert_eq!(data.ncols(), col_labels.len(), "Column labels must match matrix columns");
        BinaryLabeledMatrix { data, row_labels, col_labels }
    }

    pub fn get_value(&self, row_label: &str, col_label: &str) -> Option<bool> {
        let row_index = self.row_labels.iter().position(|r| r == row_label)?;
        let col_index = self.col_labels.iter().position(|c| c == col_label)?;
        Some(self.data[(row_index, col_index)])
    }

}

impl fmt::Display for BinaryLabeledMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Find the maximum length of row labels for alignment
        let max_label_len = self.row_labels.iter().map(|s| s.len()).max().unwrap_or(0);

        // Print column labels
        write!(f, "{:width$}", "", width = max_label_len + 4)?;
        for col in &self.col_labels {
            write!(f, "{:^8}", col)?;
        }
        writeln!(f)?;

        // Print rows with labels and data
        for (i, row_label) in self.row_labels.iter().enumerate() {
            write!(f, "{:width$}", row_label, width = max_label_len + 4)?;
            for j in 0..self.data.ncols() {
                write!(f, "{:^8}", if self.data[(i, j)] { "true" } else { "false" })?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}