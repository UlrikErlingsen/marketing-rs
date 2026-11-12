use marketing::clv::{constant_margin_value, infinite_multiple};
use serde_json::Value;

#[test]
fn constant_margin_matches_exact_rational_reference() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/clv/constant_margin.json")).unwrap();
    let rows = fixture["cases"].as_array().unwrap();
    assert!(!rows.is_empty());
    for row in rows {
        let margin = row["margin"].as_f64().unwrap();
        let retention = row["retention"].as_f64().unwrap();
        let discount = row["discount"].as_f64().unwrap();
        let comparisons = [
            (
                constant_margin_value(margin, retention, discount).unwrap(),
                row["expected_value"].as_f64().unwrap(),
            ),
            (
                infinite_multiple(retention, discount).unwrap(),
                row["expected_multiple"].as_f64().unwrap(),
            ),
        ];
        for (actual, expected) in comparisons {
            assert!(
                (actual - expected).abs() <= 1e-9 + expected.abs() * 1e-9,
                "{}: {actual} != {expected}",
                row
            );
        }
    }
}

#[test]
fn rates_convergence_and_numerical_limits() {
    for retention in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(infinite_multiple(retention, 0.1).is_err());
    }
    for discount in [-1., -2., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(infinite_multiple(0.8, discount).is_err());
    }
    assert!(infinite_multiple(1., 0.).is_err());
    assert!(infinite_multiple(0.9, -0.2).is_err());
    assert_eq!(infinite_multiple(0., 0.).unwrap(), 0.);
    assert!((infinite_multiple(1., 0.1).unwrap() - 10.).abs() < 1e-12);
    assert!((infinite_multiple(0.5, -0.25).unwrap() - 2.).abs() < 1e-12);
    assert_eq!(constant_margin_value(-10., 0.5, 0.).unwrap(), -10.);
    assert_eq!(constant_margin_value(0., 0.5, 0.).unwrap(), 0.);
    assert!(constant_margin_value(0., 1., 0.).is_err());
    for margin in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(constant_margin_value(margin, 0.5, 0.1).is_err());
    }
    assert!(constant_margin_value(f64::MAX, 0.9, 0.).is_err());
}
