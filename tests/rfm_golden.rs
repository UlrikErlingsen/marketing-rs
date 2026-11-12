//! RFM bands checked against an independent answer key.
//!
//! `tools/reference/rfm/generate_rfm.py` computes every expected band from the
//! documented rules with exact rational ordering and integer band arithmetic.
//! It writes the synthetic inputs, the expected scores and a manifest of
//! SHA-256 hashes to `tests/fixtures/rfm/`. This test checks the hashes, runs
//! `marketing::rfm::scores` on every case and requires exact equality of every
//! band, `segment()` and `total()`.

use marketing::rfm::{RfmScore, scores};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture_dir() -> PathBuf {
    root().join("tests/fixtures/rfm")
}

fn sha256(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    format!("{:x}", Sha256::digest(bytes))
}

fn manifest() -> Value {
    let text = std::fs::read_to_string(fixture_dir().join("manifest.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn load(name: &str) -> Value {
    let text = std::fs::read_to_string(fixture_dir().join(name)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn parse_value(text: &str) -> f64 {
    text.parse::<f64>()
        .unwrap_or_else(|e| panic!("bad value {text:?}: {e}"))
}

fn column(case: &Value, key: &str) -> Vec<f64> {
    case[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| parse_value(v.as_str().unwrap()))
        .collect()
}

struct Population {
    recency: Vec<f64>,
    frequency: Vec<f64>,
    monetary: Vec<f64>,
}

fn population() -> Population {
    let mut reader = csv::Reader::from_path(fixture_dir().join("population.csv")).unwrap();
    let headers = reader.headers().unwrap().clone();
    assert_eq!(
        headers.iter().collect::<Vec<_>>(),
        ["customer_id", "recency", "frequency", "monetary"]
    );
    let mut population = Population {
        recency: Vec::new(),
        frequency: Vec::new(),
        monetary: Vec::new(),
    };
    for (i, record) in reader.records().enumerate() {
        let record = record.unwrap();
        assert_eq!(&record[0], format!("c{:04}", i + 1));
        population.recency.push(parse_value(&record[1]));
        population.frequency.push(parse_value(&record[2]));
        population.monetary.push(parse_value(&record[3]));
    }
    population
}

fn band(value: &Value) -> Option<u8> {
    value.as_u64().map(|b| u8::try_from(b).unwrap())
}

#[test]
fn manifest_hashes_match_fixture_files() {
    let manifest = manifest();
    let hashes = manifest["sha256"].as_object().unwrap();
    let mut names: Vec<&str> = hashes.keys().map(String::as_str).collect();
    names.sort_unstable();
    assert_eq!(names, ["cases.json", "errors.json", "population.csv"]);
    for (name, expected) in hashes {
        assert_eq!(
            sha256(&fixture_dir().join(name)),
            expected.as_str().unwrap(),
            "{name}"
        );
    }
    let generator = manifest["generator"].as_str().unwrap();
    assert_eq!(generator, "tools/reference/rfm/generate_rfm.py");
    assert_eq!(
        sha256(&root().join(generator)),
        manifest["generator_sha256"].as_str().unwrap(),
        "generator"
    );
}

#[test]
fn every_row_matches_the_independent_key() {
    let manifest = manifest();
    let cases = load("cases.json");
    let cases = cases.as_array().unwrap();
    assert_eq!(cases.len() as u64, manifest["cases"].as_u64().unwrap());
    let population = population();
    assert_eq!(
        population.recency.len() as u64,
        manifest["population_rows"].as_u64().unwrap()
    );

    let mut rows_checked = 0u64;
    let mut seen_groups = std::collections::BTreeSet::new();
    let mut seen_modes = std::collections::BTreeSet::new();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let groups = u8::try_from(case["groups"].as_u64().unwrap()).unwrap();
        let nested = case["nested"].as_bool().unwrap();
        seen_groups.insert(groups);
        seen_modes.insert(nested);
        let (recency, frequency, monetary) = match case.get("input") {
            Some(input) => {
                assert_eq!(input.as_str().unwrap(), "population.csv", "{id}");
                (
                    population.recency.clone(),
                    population.frequency.clone(),
                    population.monetary.clone(),
                )
            }
            None => (
                column(case, "recency"),
                column(case, "frequency"),
                column(case, "monetary"),
            ),
        };
        let actual = scores(&recency, &frequency, &monetary, groups, nested)
            .unwrap_or_else(|e| panic!("{id}: unexpected error {e}"));
        let expected = case["expected"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len(), "{id}: row count");
        for (row, (got, want)) in actual.iter().zip(expected).enumerate() {
            let want = want.as_array().unwrap();
            let want_score = RfmScore {
                r: band(&want[0]),
                f: band(&want[1]),
                m: band(&want[2]),
            };
            assert_eq!(*got, want_score, "{id}: row {row}");
            assert_eq!(
                got.segment().as_deref(),
                want[3].as_str(),
                "{id}: row {row} segment"
            );
            assert_eq!(got.total(), band(&want[4]), "{id}: row {row} total");
            rows_checked += 1;
        }
    }
    assert_eq!(rows_checked, manifest["case_rows"].as_u64().unwrap());
    assert_eq!(
        seen_groups.into_iter().collect::<Vec<_>>(),
        [1, 2, 3, 5, 7, 255]
    );
    assert_eq!(seen_modes.len(), 2);
}

#[test]
fn documented_error_cases_are_rejected() {
    let manifest = manifest();
    let errors = load("errors.json");
    let errors = errors.as_array().unwrap();
    assert_eq!(
        errors.len() as u64,
        manifest["error_cases"].as_u64().unwrap()
    );
    for case in errors {
        let id = case["id"].as_str().unwrap();
        let groups = u8::try_from(case["groups"].as_u64().unwrap()).unwrap();
        let nested = case["nested"].as_bool().unwrap();
        let result = scores(
            &column(case, "recency"),
            &column(case, "frequency"),
            &column(case, "monetary"),
            groups,
            nested,
        );
        assert!(result.is_err(), "{id}: expected an error, got {result:?}");
    }
}
