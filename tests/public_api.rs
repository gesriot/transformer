use ndarray::Array2;
use transformer::predict::Predictions;

/// `Predictions` is the return type of the public `export_predictions`
/// callback. Keep a construction path available even though the report is
/// non-exhaustive to external crates.
#[test]
fn predictions_can_be_constructed_by_an_external_caller() {
    let predictions = Predictions::new(Array2::zeros((2, 1)), Vec::new());

    assert_eq!(predictions.rows(), 2);
    assert_eq!(predictions.extrapolated_rows(), 0);
}
