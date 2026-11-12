//! BG/NBD forecasts against two posterior-integral keys: dropout shape a near one, and large r.
//! Both keys were computed with mpmath, independently of this crate. Each bundle pins its files by hash.
use marketing::bgnbd::{self, History, Parameters};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::PathBuf};

fn dir(bundle: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(bundle)
}
fn sealed(bundle: &str, manifest_sha256: &str, files: usize) -> Value {
    let raw = fs::read(dir(bundle).join("manifest.json")).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&raw)), manifest_sha256);
    let manifest: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(manifest["files"].as_object().unwrap().len(), files);
    for (name, entry) in manifest["files"].as_object().unwrap() {
        let raw = fs::read(dir(bundle).join(name)).unwrap();
        assert_eq!(raw.len() as u64, entry["bytes"].as_u64().unwrap(), "{name}");
        assert_eq!(
            format!("{:x}", Sha256::digest(&raw)),
            entry["sha256"].as_str().unwrap(),
            "{name}"
        );
    }
    serde_json::from_slice(&fs::read(dir(bundle).join("expected.json")).unwrap()).unwrap()
}
fn inputs(row: &Value) -> (Parameters, History) {
    let number = |v: &Value| v.as_f64().unwrap();
    let (p, h) = (&row["parameters"], &row["history"]);
    (
        Parameters {
            r: number(&p["r"]),
            alpha: number(&p["alpha"]),
            a: number(&p["a"]),
            b: number(&p["b"]),
        },
        History {
            x: h["x"].as_u64().unwrap() as u32,
            tx: number(&h["tx"]),
            t: number(&h["t"]),
            weight: number(&h["weight"]),
        },
    )
}
struct Outcome {
    answered: usize,
    refused: BTreeSet<String>,
    failures: Vec<String>,
    worst: f64,
}
/// Every horizon of every case: an answer must lie within the key's own allowance on all three fields.
fn compare(data: &Value) -> Outcome {
    let rule = &data["tolerances"]["scalar"];
    let (absolute, relative) = (
        rule["absolute"].as_f64().unwrap(),
        rule["relative"].as_f64().unwrap(),
    );
    let mut out = Outcome {
        answered: 0,
        refused: BTreeSet::new(),
        failures: Vec::new(),
        worst: 0.,
    };
    for row in data["cases"].as_array().unwrap() {
        let (p, h) = inputs(row);
        let id = row["id"].as_str().unwrap();
        for expected in row["predictions"].as_array().unwrap() {
            let horizon = expected["horizon"].as_f64().unwrap();
            let Ok(actual) = bgnbd::predict(p, h, horizon) else {
                out.refused.insert(format!("{id} h={horizon}"));
                continue;
            };
            out.answered += 1;
            for (key, value) in [
                ("probability_active", actual.probability_active),
                ("expected_if_active", actual.expected_if_active),
                ("expected_purchases", actual.expected_purchases),
            ] {
                let want = expected[key].as_f64().unwrap();
                if !value.is_finite() || (value - want).abs() > absolute + relative * want.abs() {
                    out.failures.push(format!(
                        "{id} h={horizon} {key}: actual={value:.17e}, expected={want:.17e}"
                    ));
                }
                if want != 0. {
                    out.worst = out.worst.max((value - want).abs() / want.abs());
                }
                if horizon == 0. && key != "probability_active" {
                    assert_eq!(value.to_bits(), 0f64.to_bits(), "{id}");
                }
            }
        }
    }
    out
}

#[test]
fn near_one_key_all_88_horizons_answer_within_the_allowance() {
    let data = sealed(
        "bgnbd_near_one",
        "cae4c6bdaddd0a2d22d0756fcc58254ef4855361d11f4fb884fc4a98313ad0ea",
        9,
    );
    assert_eq!(data["cases"].as_array().unwrap().len(), 22);
    assert_eq!(
        data["tolerances"]["scalar"],
        serde_json::json!({"absolute":1e-12,"relative":1e-10})
    );
    let out = compare(&data);
    assert!(out.failures.is_empty(), "{}", out.failures.join("\n"));
    assert!(out.refused.is_empty(), "{:?}", out.refused);
    assert_eq!(out.answered, 88);
    println!("near-one key: largest relative error {:e}", out.worst);
    assert!(out.worst <= 1e-12, "{:e}", out.worst);
}

#[test]
fn large_r_key_413_horizons_answer_within_the_allowance_and_six_refuse_as_listed() {
    let data = sealed(
        "bgnbd_large_r",
        "5fbc312b76ba10ee77bc3d496b87e2c71c6ae65b8144e7af61650b09d7e35916",
        9,
    );
    assert_eq!(data["cases"].as_array().unwrap().len(), 87);
    assert_eq!(
        data["tolerances"]["scalar"],
        serde_json::json!({"absolute":1e-12,"relative":1e-9})
    );
    let out = compare(&data);
    assert!(out.failures.is_empty(), "{}", out.failures.join("\n"));
    // Inside the near-one band the series of that branch refuses large r.
    let listed: BTreeSet<String> = ["near_one_a0.992188", "near_one_a1.00781"]
        .iter()
        .flat_map(|id| [13, 104, 260].map(|horizon| format!("{id} h={horizon}")))
        .collect();
    assert_eq!(out.refused, listed);
    assert_eq!(out.answered, 413);
    println!("large-r key: largest relative error {:e}", out.worst);
    assert!(out.worst <= 5e-12, "{:e}", out.worst);
}

#[test]
fn reported_large_r_forecasts_are_right() {
    let p = Parameters {
        r: 200.,
        alpha: 200.,
        a: 0.8,
        b: 2.5,
    };
    let h = History {
        x: 0,
        tx: 0.,
        t: 52.,
        weight: 1.,
    };
    // 0.2.0 returned 19.89 and 2.0e14 here.
    for (horizon, expected) in [(104., 12.977220324032245), (260., 17.836268845151363)] {
        let got = bgnbd::predict(p, h, horizon).unwrap();
        assert!(
            (got.expected_if_active - expected).abs() <= 1e-11 * expected,
            "{horizon}: {}",
            got.expected_if_active
        );
        assert_eq!(got.probability_active, 1.);
    }
}

#[test]
fn tiny_dropout_shapes_keep_their_precision() {
    // With x = 0 and a tiny a + b, a + b - 1 is close to -1. Expected values: equation 10 with mpmath
    // hyp2f1 at 60 and 90 digits, the same binary64 at both.
    let h = History {
        x: 0,
        tx: 0.,
        t: 52.,
        weight: 1.,
    };
    for (r, horizon, expected) in [
        (200., 104., 41.769841269678466),
        (20., 1040., 144.9444444436861),
        (2000., 13., 6.8352810158596515),
    ] {
        let p = Parameters {
            r,
            alpha: r,
            a: 1e-12,
            b: 1e-12,
        };
        let got = bgnbd::predict(p, h, horizon).unwrap().expected_if_active;
        assert!(
            (got - expected).abs() <= 1e-11 * expected,
            "r={r} h={horizon}: {got} vs {expected}"
        );
    }
}

#[test]
fn documented_long_horizon_error_inside_the_near_one_band() {
    // The module documentation and the changelog name this input. Inside the band the series returns an
    // error when its rounding estimate exceeds 1e-11 of the expected purchases if active, here from
    // horizon 108.
    let p = Parameters {
        r: 0.1704,
        alpha: 5.92,
        a: 1.012,
        b: 7.727,
    };
    let h = History {
        x: 0,
        tx: 0.,
        t: 2.383,
        weight: 1.,
    };
    assert!(bgnbd::predict(p, h, 100.).is_ok());
    for horizon in [108., 150., 200., 1000.] {
        assert!(bgnbd::predict(p, h, horizon).is_err(), "{horizon}");
    }
    // Just outside the band the same history answers at every one of these horizons.
    let outside = Parameters {
        a: 1. + 1. / 64. + 1e-9,
        ..p
    };
    for horizon in [100., 150., 200., 1000.] {
        assert!(bgnbd::predict(outside, h, horizon).is_ok(), "{horizon}");
    }
}
