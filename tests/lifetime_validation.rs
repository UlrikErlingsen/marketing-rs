use marketing::{bgbb, bgnbd, sbg};

#[test]
fn impossible_fixed_probability_histories_refuse_conditional_activity() {
    let h = bgbb::History {
        n: 3,
        tx: 2,
        x: 1,
        weight: 1.,
    };
    assert_eq!(
        bgbb::history_probability_no_heterogeneity(h, 0., 0.2).unwrap(),
        0.
    );
    assert!(bgbb::probability_alive_no_heterogeneity(h, 0., 0.2).is_err());
}

#[test]
fn bgnbd_refuses_unrepresentable_likelihoods_and_time_combinations() {
    let h = bgnbd::History {
        x: 1,
        tx: 1.,
        t: 10.,
        weight: 1.,
    };
    let p = bgnbd::Parameters {
        r: f64::MAX,
        alpha: 4.,
        a: 2.,
        b: 3.,
    };
    assert!(bgnbd::log_likelihood(p, h).is_err());
    let p = bgnbd::Parameters {
        r: 1.,
        alpha: 1e308,
        a: 2.,
        b: 3.,
    };
    let h = bgnbd::History {
        x: 1,
        tx: 1e308,
        t: 1e308,
        weight: 1.,
    };
    assert!(bgnbd::predict(p, h, 0.).is_err());
    assert!(bgnbd::predict(p, h, 1.).is_err());
}

#[test]
fn sbg_refuses_parameter_overflow_and_nonfinite_calculated_values() {
    let p = sbg::Parameters {
        alpha: 1e308,
        beta: 1e308,
    };
    assert!(sbg::survival(p, 1).is_err());
    assert!(sbg::retention(p, 1).is_err());
    let p = sbg::Parameters {
        alpha: f64::MIN_POSITIVE,
        beta: f64::MIN_POSITIVE,
    };
    assert!(sbg::survival(p, 1).is_err());
    assert!(sbg::retention(p, 1).is_err());
    assert!(sbg::log_likelihood(p, &[100, 50, 25]).is_err());
    assert!(
        sbg::expected_lifetime(sbg::Parameters {
            alpha: 1. + f64::EPSILON,
            beta: f64::MAX
        })
        .is_err()
    );
}

#[test]
fn invalid_parameters_histories_horizons_and_weights_are_errors() {
    let p = bgnbd::Parameters {
        r: 0.8,
        alpha: 7.5,
        a: 1.4,
        b: 3.2,
    };
    let h = bgnbd::History {
        x: 2,
        tx: 30.43,
        t: 38.86,
        weight: 1.,
    };
    for v in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(bgnbd::predict(bgnbd::Parameters { r: v, ..p }, h, 1.).is_err());
        assert!(sbg::survival(sbg::Parameters { alpha: v, beta: 3. }, 1).is_err());
        assert!(
            bgbb::Parameters {
                alpha: v,
                beta: 1.,
                gamma: 1.,
                delta: 3.
            }
            .validate()
            .is_err()
        );
    }
    for horizon in [-1., f64::NAN, f64::INFINITY] {
        assert!(bgnbd::predict(p, h, horizon).is_err());
    }
    // Since 0.2.1, a = 1 answers with the continuous limit of the forecast and no longer refuses.
    let at = |a: f64| {
        bgnbd::predict(bgnbd::Parameters { a, ..p }, h, 1.)
            .unwrap()
            .expected_if_active
    };
    assert!(at(1. + 1e-6) < at(1.) && at(1.) < at(1. - 1e-6));
    assert!((at(1. - 1e-6) - at(1. + 1e-6)) < 1e-5 * at(1.));
    assert_eq!(
        bgnbd::predict(bgnbd::Parameters { a: 1., ..p }, h, 0.)
            .unwrap()
            .expected_purchases,
        0.
    );
    for bad in [
        bgnbd::History { tx: -1., ..h },
        bgnbd::History { tx: 40., ..h },
        bgnbd::History { t: 0., ..h },
        bgnbd::History { weight: -1., ..h },
    ] {
        assert!(bgnbd::fit(&[bad]).is_err());
    }
    assert!(bgnbd::fit(&[]).is_err());
    assert!(bgnbd::fit(&[bgnbd::History { weight: 0., ..h }]).is_err());
    for cohort in [&[][..], &[0, 0, 0], &[10, 9], &[10, 11, 9]] {
        assert!(sbg::fit(cohort).is_err());
    }
    assert!(
        sbg::retention(
            sbg::Parameters {
                alpha: 2.,
                beta: 3.
            },
            0
        )
        .is_err()
    );
    assert_eq!(
        sbg::expected_lifetime(sbg::Parameters {
            alpha: 1.,
            beta: 3.
        })
        .unwrap(),
        None
    );
    for bad in [
        bgbb::History {
            n: 0,
            tx: 0,
            x: 0,
            weight: 1.,
        },
        bgbb::History {
            n: 3,
            tx: 2,
            x: 3,
            weight: 1.,
        },
        bgbb::History {
            n: 3,
            tx: 4,
            x: 2,
            weight: 1.,
        },
        bgbb::History {
            n: 3,
            tx: 2,
            x: 1,
            weight: f64::NAN,
        },
    ] {
        assert!(bgbb::fit(&[bad]).is_err());
    }
    assert!(bgbb::fit(&[]).is_err());
    let h = bgbb::History {
        n: 3,
        tx: 2,
        x: 1,
        weight: 1.,
    };
    for q in [-1., 1.1, f64::NAN] {
        assert!(bgbb::probability_alive_no_heterogeneity(h, 0.5, q).is_err());
    }
}

#[test]
fn checked_forecasts_refuse_failed_or_bounded_fit_attempts() {
    let p = bgnbd::Parameters {
        r: 0.8,
        alpha: 7.5,
        a: 1.4,
        b: 3.2,
    };
    let h = bgnbd::History {
        x: 2,
        tx: 30.43,
        t: 38.86,
        weight: 1.,
    };
    let fit = bgnbd::Fit {
        parameters: p,
        log_likelihood: 0.,
        converged: false,
    };
    assert!(fit.predict(h, 1.).is_err());
    let fit = sbg::Fit {
        parameters: sbg::Parameters {
            alpha: 2.,
            beta: 3.,
        },
        log_likelihood: 0.,
        converged: false,
    };
    assert!(fit.survival(1).is_err());
    assert!(fit.retention(1).is_err());
    assert!(fit.expected_lifetime().is_err());
    for (converged, at_bound) in [(false, false), (true, true)] {
        let fit = bgbb::Fit {
            parameters: bgbb::Parameters {
                alpha: 1.,
                beta: 1.,
                gamma: 1.,
                delta: 3.,
            },
            log_likelihood: 0.,
            converged,
            at_bound,
            distinct_histories: 1,
        };
        assert!(bgbb::fit_problem(&fit).is_some());
        assert!(
            fit.predict(
                bgbb::History {
                    n: 3,
                    tx: 2,
                    x: 1,
                    weight: 1.
                },
                1
            )
            .is_err()
        );
    }
}

#[test]
fn degenerate_no_purchase_or_no_churn_fits_cannot_forecast() {
    let h = bgnbd::History {
        x: 0,
        tx: 0.,
        t: 10.,
        weight: 100.,
    };
    let fit = bgnbd::fit(&[h]).unwrap();
    assert!(!fit.converged);
    assert!(fit.predict(h, 1.).is_err());
    let fit = sbg::fit(&[100, 100, 100]).unwrap();
    assert!(!fit.converged);
    assert!(fit.survival(1).is_err());
    let h = bgbb::History {
        n: 6,
        tx: 0,
        x: 0,
        weight: 100.,
    };
    let fit = bgbb::fit(&[h]).unwrap();
    assert!(bgbb::fit_problem(&fit).is_some());
    assert!(fit.predict(h, 1).is_err());
}

#[test]
fn individual_equations_ignore_frequency_weights_and_zero_repeat_activity_is_one() {
    let p = bgnbd::Parameters {
        r: 0.8,
        alpha: 7.5,
        a: 1.4,
        b: 3.2,
    };
    let h = bgnbd::History {
        x: 0,
        tx: 0.,
        t: 10.,
        weight: 1.,
    };
    let a = bgnbd::predict(p, h, 10.).unwrap();
    let b = bgnbd::predict(p, bgnbd::History { weight: 17., ..h }, 10.).unwrap();
    assert_eq!(a.probability_active, 1.);
    assert_eq!(a.expected_purchases, b.expected_purchases);
    assert_eq!(
        bgnbd::log_likelihood(p, h).unwrap(),
        bgnbd::log_likelihood(p, bgnbd::History { weight: 17., ..h }).unwrap()
    );
    let p = bgbb::Parameters {
        alpha: 1.8,
        beta: 2.4,
        gamma: 1.2,
        delta: 4.1,
    };
    let h = bgbb::History {
        n: 6,
        tx: 6,
        x: 3,
        weight: 1.,
    };
    assert!(bgbb::predict(p, h, 1).unwrap().probability_alive < 1.);
    assert_eq!(
        bgbb::predict(p, h, 0).unwrap().expected_future_purchases,
        0.
    );
    assert_eq!(
        bgbb::log_likelihood(p, h).unwrap(),
        bgbb::log_likelihood(p, bgbb::History { weight: 20., ..h }).unwrap()
    );
}

#[test]
fn overflowing_weighted_likelihood_cannot_be_a_successful_fit() {
    let h = bgnbd::History {
        x: 1,
        tx: 1.,
        t: 10.,
        weight: f64::MAX,
    };
    assert!(bgnbd::fit(&[h]).is_err());
    let h = bgbb::History {
        n: 3,
        tx: 2,
        x: 1,
        weight: f64::MAX,
    };
    assert!(bgbb::fit(&[h]).is_err());
}

#[test]
fn zero_purchase_bgbb_history_requires_zero_last_purchase_time() {
    let h = bgbb::History {
        n: 3,
        tx: 2,
        x: 0,
        weight: 1.,
    };
    let p = bgbb::Parameters {
        alpha: 1.8,
        beta: 2.4,
        gamma: 1.2,
        delta: 4.1,
    };
    assert!(h.validate().is_err());
    assert!(bgbb::log_likelihood(p, h).is_err());
    assert!(bgbb::predict(p, h, 1).is_err());
    assert!(bgbb::fit(&[h]).is_err());
    assert!(bgbb::history_probability_no_heterogeneity(h, 0.5, 0.2).is_err());
    assert!(bgbb::probability_alive_no_heterogeneity(h, 0.5, 0.2).is_err());
}

#[test]
fn nonfinite_fit_diagnostics_cannot_authorize_checked_forecasts() {
    for log_likelihood in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let fit = bgnbd::Fit {
            parameters: bgnbd::Parameters {
                r: 0.8,
                alpha: 7.5,
                a: 1.4,
                b: 3.2,
            },
            log_likelihood,
            converged: true,
        };
        assert!(
            fit.predict(
                bgnbd::History {
                    x: 1,
                    tx: 1.,
                    t: 10.,
                    weight: 1.
                },
                1.
            )
            .is_err()
        );
        let fit = sbg::Fit {
            parameters: sbg::Parameters {
                alpha: 2.,
                beta: 3.,
            },
            log_likelihood,
            converged: true,
        };
        assert!(fit.survival(1).is_err());
        assert!(fit.retention(1).is_err());
        assert!(fit.expected_lifetime().is_err());
        let fit = bgbb::Fit {
            parameters: bgbb::Parameters {
                alpha: 1.8,
                beta: 2.4,
                gamma: 1.2,
                delta: 4.1,
            },
            log_likelihood,
            converged: true,
            at_bound: false,
            distinct_histories: 1,
        };
        assert!(bgbb::fit_problem(&fit).is_some());
        assert!(
            fit.predict(
                bgbb::History {
                    n: 3,
                    tx: 2,
                    x: 1,
                    weight: 1.
                },
                1
            )
            .is_err()
        );
    }
}

#[test]
fn zero_repeat_bgnbd_history_requires_zero_last_purchase_time() {
    let p = bgnbd::Parameters {
        r: 0.8,
        alpha: 7.5,
        a: 1.4,
        b: 3.2,
    };
    let h = bgnbd::History {
        x: 0,
        tx: 1.,
        t: 10.,
        weight: 1.,
    };
    assert_eq!(
        h.validate().unwrap_err().0,
        "BG/NBD: tx must be 0 when x = 0."
    );
    assert!(bgnbd::log_likelihood(p, h).is_err());
    assert!(bgnbd::predict(p, h, 1.).is_err());
    assert!(bgnbd::fit(&[h]).is_err());
    for tx in [0., -0.] {
        let valid = bgnbd::History { tx, ..h };
        assert!(valid.validate().is_ok());
        assert_eq!(bgnbd::predict(p, valid, 1.).unwrap().probability_active, 1.);
    }
}

#[test]
fn bgbb_shape_lost_to_rounding_is_an_error_not_a_panic() {
    // beta + n - x rounds to zero when beta is below about 1.1e-16 times n and every opportunity has a
    // purchase. Before 0.2.1 the log-beta call panicked.
    let p = bgbb::Parameters {
        alpha: 1.,
        beta: 1e-17,
        gamma: 1.,
        delta: 1.,
    };
    for (n, tx, x) in [(1, 1, 1), (5, 5, 5), (5, 2, 2)] {
        let h = bgbb::History {
            n,
            tx,
            x,
            weight: 1.,
        };
        assert!(bgbb::log_likelihood(p, h).is_err(), "{n} {tx} {x}");
        assert!(bgbb::predict(p, h, 3).is_err(), "{n} {tx} {x}");
    }
    // The same shape with a purchase missing keeps beta in the sum and still answers.
    let h = bgbb::History {
        n: 5,
        tx: 4,
        x: 3,
        weight: 1.,
    };
    assert!(bgbb::log_likelihood(p, h).unwrap().is_finite());
    assert!(bgbb::predict(p, h, 3).is_ok());
}
