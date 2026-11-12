use marketing::{bgbb as bb, bgnbd as bg, sbg};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::Path;

fn data(name: &str) -> Value {
    serde_json::from_slice(
        &std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/lifetime")
                .join(name),
        )
        .unwrap(),
    )
    .unwrap()
}
fn n(v: &Value, key: &str) -> f64 {
    v[key].as_f64().unwrap()
}
fn bg_p(v: &Value) -> bg::Parameters {
    bg::Parameters {
        r: n(v, "r"),
        alpha: n(v, "alpha"),
        a: n(v, "a"),
        b: n(v, "b"),
    }
}
fn bg_h(v: &Value) -> bg::History {
    bg::History {
        x: v["x"].as_u64().unwrap() as u32,
        tx: n(v, "tx"),
        t: n(v, "t"),
        weight: n(v, "weight"),
    }
}
fn bb_p(v: &Value) -> bb::Parameters {
    bb::Parameters {
        alpha: n(v, "alpha"),
        beta: n(v, "beta"),
        gamma: n(v, "gamma"),
        delta: n(v, "delta"),
    }
}
fn bb_h(v: &Value) -> bb::History {
    bb::History {
        n: v["n"].as_u64().unwrap() as u32,
        tx: v["tx"].as_u64().unwrap() as u32,
        x: v["x"].as_u64().unwrap() as u32,
        weight: n(v, "weight"),
    }
}
fn close(label: &str, actual: f64, expected: f64, rule: &Value) {
    assert!(
        actual.is_finite()
            && (actual - expected).abs() <= n(rule, "atol") + n(rule, "rtol") * expected.abs(),
        "{label}: {actual} != {expected}; {rule}"
    );
}
fn stationary(point: [f64; 4], threshold: f64, f: impl Fn([f64; 4]) -> f64) {
    let h = 1e-4;
    for axis in 0..4 {
        let eval = |step: f64| {
            let mut z = point;
            z[axis] += step * h;
            f(z)
        };
        let derivative = (eval(-2.) - 8. * eval(-1.) + 8. * eval(1.) - eval(2.)) / (12. * h);
        assert!(
            derivative.abs() < threshold,
            "axis {axis}: derivative {derivative}"
        );
    }
}

#[test]
fn independent_fixture_and_generator_hashes_are_frozen() {
    let manifest = data("manifest.json");
    for (name, sha) in manifest["files"].as_object().unwrap() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/lifetime")
            .join(name);
        assert_eq!(
            format!("{:x}", Sha256::digest(std::fs::read(path).unwrap())),
            sha.as_str().unwrap(),
            "{name}"
        );
    }
    for (name, sha) in manifest["generators"].as_object().unwrap() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(name);
        assert_eq!(
            format!("{:x}", Sha256::digest(std::fs::read(path).unwrap())),
            sha.as_str().unwrap(),
            "{name}"
        );
    }
}

#[test]
fn positive_age_bgnbd_fixed_cases_match_lifetimes_and_zero_age_is_refused() {
    let raw = data("bgnbd.json");
    let rule = &raw["metadata"]["tolerances"]["fixed"];
    let (mut calculated, mut refused) = (0, 0);
    for row in raw["fixed_cases"].as_array().unwrap() {
        let p = bg_p(&row["parameters"]);
        let h = bg_h(&row["history"]);
        if h.t == 0. {
            assert!(bg::log_likelihood(p, h).is_err());
            assert!(bg::predict(p, h, n(row, "horizon")).is_err());
            refused += 1;
            continue;
        }
        let a = bg::predict(p, h, n(row, "horizon")).unwrap();
        let e = &row["expected"];
        close(
            "BG/NBD LL",
            bg::log_likelihood(p, h).unwrap(),
            n(e, "log_likelihood"),
            rule,
        );
        close(
            "BG/NBD activity",
            a.probability_active,
            n(e, "probability_active"),
            rule,
        );
        close(
            "BG/NBD conditional mean",
            a.expected_if_active,
            n(e, "expected_if_active"),
            rule,
        );
        close(
            "BG/NBD mean",
            a.expected_purchases,
            n(e, "expected_purchases"),
            rule,
        );
        calculated += 1;
    }
    assert_eq!((calculated, refused), (60, 12));
}

#[test]
fn bgnbd_synthetic_mle_matches_independent_optimum_and_is_stationary() {
    let raw = data("bgnbd.json");
    let fit = &raw["fit_case"];
    let rules = &raw["metadata"]["tolerances"];
    let rows: Vec<_> = fit["rows"].as_array().unwrap().iter().map(bg_h).collect();
    assert_eq!(rows.len(), 1000);
    let a = bg::fit(&rows).unwrap();
    let e = &fit["expected"];
    assert_eq!(a.converged, e["converged"].as_bool().unwrap());
    for (key, value) in [
        ("r", a.parameters.r),
        ("alpha", a.parameters.alpha),
        ("a", a.parameters.a),
        ("b", a.parameters.b),
    ] {
        close(
            key,
            value,
            n(&e["parameters"], key),
            &rules["mle_parameters"],
        );
    }
    close(
        "BG/NBD fit LL",
        a.log_likelihood,
        n(e, "log_likelihood"),
        &rules["mle_total_log_likelihood"],
    );
    stationary(
        [
            a.parameters.r,
            a.parameters.alpha,
            a.parameters.a,
            a.parameters.b,
        ]
        .map(f64::ln),
        n(rules, "mle_log_parameter_mean_nll_gradient_max_abs"),
        |z| {
            let [r, alpha, a, b] = z.map(f64::exp);
            -rows
                .iter()
                .map(|h| {
                    h.weight * bg::log_likelihood(bg::Parameters { r, alpha, a, b }, *h).unwrap()
                })
                .sum::<f64>()
                / 1000.
        },
    );
    let reference = bg::Fit {
        parameters: bg_p(&e["parameters"]),
        log_likelihood: n(e, "log_likelihood"),
        converged: true,
    };
    for expected in fit["diagnostics"]["predicted_purchases"]
        .as_array()
        .unwrap()
    {
        let total = rows
            .iter()
            .map(|h| {
                reference
                    .predict(*h, n(expected, "horizon"))
                    .unwrap()
                    .expected_purchases
            })
            .sum::<f64>();
        close(
            "BG/NBD aggregate fixed-parameter forecast",
            total,
            n(expected, "total_expected_purchases"),
            &rules["fixed"],
        );
    }
}

#[test]
fn bgbb_fixed_likelihoods_and_forecasts_match_lifetimes_and_forward_checks() {
    let raw = data("bgbb.json");
    let rule = &raw["metadata"]["tolerances"]["fixed"];
    assert_eq!(raw["fixed_cases"].as_array().unwrap().len(), 72);
    for row in raw["fixed_cases"].as_array().unwrap() {
        let p = bb_p(&row["parameters"]);
        let h = bb_h(&row["history"]);
        let horizon = n(row, "horizon") as u32;
        let a = bb::predict(p, h, horizon).unwrap();
        for e in [&row["expected"], &row["independent_check"]["expected"]] {
            close(
                "BG/BB LL",
                bb::log_likelihood(p, h).unwrap(),
                n(e, "log_likelihood"),
                rule,
            );
            close(
                "BG/BB activity",
                a.probability_alive,
                n(e, "probability_alive"),
                rule,
            );
            close(
                "BG/BB mean",
                a.expected_future_purchases,
                n(e, "expected_future_purchases"),
                rule,
            );
        }
    }
}

#[test]
fn fixed_probability_kernels_match_independent_finite_state_calculations() {
    let raw = data("bgbb.json");
    let rule = &raw["metadata"]["tolerances"]["fixed"];
    let rows = raw["fixed_pq_cases"].as_array().unwrap();
    assert_eq!(rows.len(), 16);
    for row in rows {
        let h = bb_h(&row["history"]);
        let p = n(&row["parameters"], "p");
        let q = n(&row["parameters"], "q");
        close(
            "fixed probability",
            bb::history_probability_no_heterogeneity(h, p, q).unwrap(),
            n(&row["expected"], "likelihood"),
            rule,
        );
        if row["status"] == "zero_likelihood" {
            assert!(bb::probability_alive_no_heterogeneity(h, p, q).is_err());
        } else {
            close(
                "fixed activity",
                bb::probability_alive_no_heterogeneity(h, p, q).unwrap(),
                n(&row["expected"], "probability_alive"),
                rule,
            );
        }
    }
}

#[test]
fn bgbb_synthetic_mle_matches_independent_optimum_and_is_stationary() {
    let raw = data("bgbb.json");
    let fit = &raw["fit_case"];
    let rules = &raw["metadata"]["tolerances"];
    let rows: Vec<_> = fit["rows"].as_array().unwrap().iter().map(bb_h).collect();
    assert_eq!(rows.len(), 1000);
    let a = bb::fit(&rows).unwrap();
    let e = &fit["expected"];
    assert_eq!(a.converged, e["converged"].as_bool().unwrap());
    assert!(!a.at_bound);
    assert_eq!(
        a.distinct_histories,
        fit["diagnostics"]["unique_summary_count"].as_u64().unwrap() as usize
    );
    for (key, value) in [
        ("alpha", a.parameters.alpha),
        ("beta", a.parameters.beta),
        ("gamma", a.parameters.gamma),
        ("delta", a.parameters.delta),
    ] {
        close(
            key,
            value,
            n(&e["parameters"], key),
            &rules["mle_parameters"],
        );
    }
    close(
        "BG/BB fit LL",
        a.log_likelihood,
        n(e, "log_likelihood"),
        &rules["mle_total_log_likelihood"],
    );
    stationary(
        [
            a.parameters.alpha,
            a.parameters.beta,
            a.parameters.gamma,
            a.parameters.delta,
        ]
        .map(f64::ln),
        n(rules, "mle_log_parameter_mean_nll_gradient_max_abs"),
        |z| {
            let [alpha, beta, gamma, delta] = z.map(f64::exp);
            -rows
                .iter()
                .map(|h| {
                    h.weight
                        * bb::log_likelihood(
                            bb::Parameters {
                                alpha,
                                beta,
                                gamma,
                                delta,
                            },
                            *h,
                        )
                        .unwrap()
                })
                .sum::<f64>()
                / 1000.
        },
    );
    let reference = bb::Fit {
        parameters: bb_p(&e["parameters"]),
        log_likelihood: n(e, "log_likelihood"),
        converged: true,
        at_bound: false,
        distinct_histories: a.distinct_histories,
    };
    for expected in fit["diagnostics"]["predicted_purchases"]
        .as_array()
        .unwrap()
    {
        let total = rows
            .iter()
            .map(|h| {
                reference
                    .predict(*h, n(expected, "horizon") as u32)
                    .unwrap()
                    .expected_future_purchases
            })
            .sum::<f64>();
        close(
            "BG/BB aggregate fixed-parameter forecast",
            total,
            n(expected, "total_expected_purchases"),
            &rules["fixed"],
        );
    }
}

fn absolute(label: &str, actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "{label}: {actual} != {expected}; absolute {tolerance}"
    );
}

#[test]
fn sbg_fixed_values_match_independent_scipy_distribution() {
    let raw = data("sbg.json");
    let tolerance = n(&raw["metadata"]["tolerances"], "fixed_absolute");
    assert_eq!(raw["fixed_cases"].as_array().unwrap().len(), 4);
    for row in raw["fixed_cases"].as_array().unwrap() {
        let p = sbg::Parameters {
            alpha: n(row, "alpha"),
            beta: n(row, "beta"),
        };
        let e = &row["expected"];
        for (period, v) in e["survival"].as_array().unwrap().iter().enumerate() {
            absolute(
                "sBG survival",
                sbg::survival(p, period as u32).unwrap(),
                v.as_f64().unwrap(),
                tolerance,
            );
            if period > 0 {
                absolute(
                    "sBG retention",
                    sbg::retention(p, period as u32).unwrap(),
                    e["retention"][period - 1].as_f64().unwrap(),
                    tolerance,
                );
                let loss = sbg::survival(p, period as u32 - 1).unwrap()
                    - sbg::survival(p, period as u32).unwrap();
                absolute(
                    "sBG churn mass",
                    loss,
                    e["churn_probability"][period - 1].as_f64().unwrap(),
                    tolerance,
                );
            }
        }
        let lifetime = sbg::expected_lifetime(p).unwrap();
        if e["mean_duration"].is_null() {
            assert!(lifetime.is_none());
        } else {
            absolute(
                "sBG expected duration",
                lifetime.unwrap(),
                n(e, "mean_duration"),
                tolerance,
            );
        }
    }
}

#[test]
fn sbg_synthetic_fits_match_independent_scipy_and_high_precision_checks() {
    let raw = data("sbg.json");
    let tolerances = &raw["metadata"]["tolerances"];
    assert_eq!(raw["fit_cases"].as_array().unwrap().len(), 2);
    for row in raw["fit_cases"].as_array().unwrap() {
        let survivors: Vec<u64> = serde_json::from_value(row["survivors"].clone()).unwrap();
        assert_eq!(survivors[0], 2000);
        assert_eq!(
            row["losses"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .sum::<u64>()
                + row["censored"].as_u64().unwrap(),
            2000
        );
        let a = sbg::fit(&survivors).unwrap();
        let e = &row["expected"];
        assert_eq!(a.converged, e["convergence"]["success"].as_bool().unwrap());
        absolute(
            "sBG fitted alpha",
            a.parameters.alpha,
            n(e, "alpha"),
            n(tolerances, "fit_parameters_absolute"),
        );
        absolute(
            "sBG fitted beta",
            a.parameters.beta,
            n(e, "beta"),
            n(tolerances, "fit_parameters_absolute"),
        );
        absolute(
            "sBG fitted LL",
            a.log_likelihood,
            n(e, "loglikelihood"),
            n(tolerances, "fit_loglikelihood_absolute"),
        );
        for (period, expected) in e["curves"]["survival"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            absolute(
                "sBG fitted survival",
                a.survival(period as u32).unwrap(),
                expected.as_f64().unwrap(),
                n(tolerances, "fit_curves_absolute"),
            );
            if period > 0 {
                absolute(
                    "sBG fitted retention",
                    a.retention(period as u32).unwrap(),
                    e["curves"]["retention"][period - 1].as_f64().unwrap(),
                    n(tolerances, "fit_curves_absolute"),
                );
            }
        }
        let lifetime = a.expected_lifetime().unwrap();
        if e["curves"]["mean_duration"].is_null() {
            assert!(lifetime.is_none());
        } else {
            absolute(
                "sBG fitted duration",
                lifetime.unwrap(),
                n(&e["curves"], "mean_duration"),
                n(tolerances, "fit_mean_duration_absolute"),
            );
        }
        for likelihood in row["loglikelihood_at_fixed_parameters"].as_array().unwrap() {
            let p = raw["fixed_cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["name"] == likelihood["fixed_case"])
                .unwrap();
            absolute(
                "sBG fixed LL",
                sbg::log_likelihood(
                    sbg::Parameters {
                        alpha: n(p, "alpha"),
                        beta: n(p, "beta"),
                    },
                    &survivors,
                )
                .unwrap(),
                n(likelihood, "loglikelihood"),
                n(tolerances, "fixed_absolute"),
            );
        }
    }
}
