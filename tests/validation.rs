use marketing::rfm::{RfmScore, scores};
use sha2::{Digest, Sha256};

#[test]
fn fixtures_are_unchanged() {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/manifest.json")).unwrap();
    for (path, hash) in manifest["sha256"].as_object().unwrap() {
        let bytes =
            std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            hash.as_str().unwrap(),
            "{path}"
        );
    }
}

#[test]
fn rejects_invalid_configuration_and_empty_dimensions() {
    assert!(scores(&[1.], &[], &[1.], 5, false).is_err());
    assert!(scores(&[1.], &[1.], &[], 5, false).is_err());
    assert!(scores(&[1.], &[1.], &[1.], 0, false).is_err());
    assert!(scores(&[], &[], &[], 5, false).is_err());
    for axis in 0..3 {
        let mut values = [[1.]; 3];
        values[axis][0] = f64::NAN;
        assert!(scores(&values[0], &values[1], &values[2], 5, true).is_err());
    }
}

#[test]
fn ties_constant_columns_signed_zero_and_total_overflow_follow_documented_bands() {
    let v = [1., 1., 1., 2., 2.];
    let output = scores(&v, &v, &v, 5, false).unwrap();
    assert_eq!(
        output.iter().map(|s| s.r).collect::<Vec<_>>(),
        [Some(1), Some(1), Some(1), Some(2), Some(2)]
    );
    assert_eq!(
        output.iter().map(|s| s.f).collect::<Vec<_>>(),
        [Some(1), Some(2), Some(2), Some(1), Some(1)]
    );
    let v = [1., 2., 3., 4., 5., 6.];
    assert_eq!(
        scores(&v, &v, &v, 5, false)
            .unwrap()
            .iter()
            .map(|s| s.r)
            .collect::<Vec<_>>(),
        [Some(1), Some(1), Some(2), Some(3), Some(4), Some(5)]
    );
    for nested in [false, true] {
        assert!(
            scores(&[4.; 8], &[1.; 8], &[20.; 8], 5, nested)
                .unwrap()
                .iter()
                .all(|s| s.segment().as_deref() == Some("1-1-1"))
        );
        assert!(
            scores(&[0., -0.], &[0., -0.], &[0., -0.], 1, nested)
                .unwrap()
                .iter()
                .all(|s| s.total() == Some(3))
        );
    }
    assert_eq!(
        RfmScore {
            r: Some(100),
            f: Some(100),
            m: Some(100)
        }
        .total(),
        None
    );
    assert_eq!(RfmScore::default().segment(), None);
    assert_eq!(RfmScore::default().total(), None);
}

#[test]
fn nested_missing_cells_and_infinities() {
    let v = [f64::NEG_INFINITY, 0., f64::INFINITY];
    let out = scores(&v, &v, &v, 255, false).unwrap();
    assert_eq!(
        out[0],
        RfmScore {
            r: Some(1),
            f: Some(3),
            m: Some(3)
        }
    );
    let out = scores(
        &[f64::NAN, 1., 2.],
        &[1., f64::NAN, 1.],
        &[10., 20., 5.],
        5,
        true,
    )
    .unwrap();
    assert_eq!(out[0], RfmScore::default());
    assert_eq!(out[1].m, None);
    assert_eq!(out[2].m, Some(1));
}

#[test]
fn tied_values_can_split_across_bands_and_a_singleton_nested_cell_gets_band_one() {
    // Three equal recencies and one larger, two bands: positions 0 and 1 get band 1, position 2 gets band 2.
    let recency = [1., 1., 1., 5.];
    let other = [4., 3., 2., 1.];
    let independent = scores(&recency, &other, &other, 2, false).unwrap();
    let r: Vec<_> = independent.iter().map(|s| s.r).collect();
    assert_eq!(r, [Some(1), Some(1), Some(2), Some(2)]);
    // Nested: the row alone in its recency band is alone in its cell, so its F and M are 1.
    let nested = scores(&[1., 2., 3., 9.], &other, &other, 4, true).unwrap();
    for score in &nested {
        assert_eq!((score.f, score.m), (Some(1), Some(1)));
    }
}
