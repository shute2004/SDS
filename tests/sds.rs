use sds_sort::sort::{sds_sort, sds_sort_inplace};

fn assert_matches_total_cmp(mut values: Vec<f64>) {
    let mut expected = values.clone();
    expected.sort_by(|a, b| a.total_cmp(b));
    sds_sort(&mut values);
    assert_eq!(values, expected);
}

#[test]
fn orders_mixed_finite_values() {
    assert_matches_total_cmp(vec![9.0, -3.0, 1.5, 1.5, 0.0, 7.2, -8.0, 4.0]);
}

#[test]
fn handles_duplicates() {
    assert_matches_total_cmp(vec![2.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 0.0]);
}

#[test]
fn handles_non_finite_values_without_entering_predictor_path() {
    assert_matches_total_cmp(vec![
        4.0,
        f64::NAN,
        f64::INFINITY,
        -2.0,
        f64::NEG_INFINITY,
        -0.0,
        0.0,
    ]);
}

#[test]
fn inplace_variant_matches_total_cmp() {
    let mut values = vec![5.0, 1.0, 4.0, 2.0, 8.0, 0.0, 2.0, -1.0];
    let mut expected = values.clone();
    expected.sort_by(|a, b| a.total_cmp(b));
    sds_sort_inplace(&mut values);
    assert_eq!(values, expected);
}

#[test]
fn inplace_variant_handles_non_finite_values() {
    let mut values = vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0];
    let mut expected = values.clone();
    expected.sort_by(|a, b| a.total_cmp(b));
    sds_sort_inplace(&mut values);
    assert_eq!(values, expected);
}
