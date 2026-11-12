# marketing

Marketing formulas and customer lifetime models in Rust, tested against independent answer keys. Requires **Rust 1.88** or later. Licensed under **MIT OR Apache-2.0**, at your option.

- Crate: <https://crates.io/crates/marketing>
- Documentation: <https://docs.rs/marketing>
- Source, tests and answer keys: <https://github.com/UlrikErlingsen/marketing-rs>

Written with AI assistance. Each method is checked against independently computed answer keys.

```sh
cargo add marketing@0.2.2
```

```rust
use marketing::{clv, rfm};

fn main() -> Result<(), marketing::Error> {
    let scores = rfm::scores(&[1., 10., 30.], &[9., 5., 1.], &[100., 50., 10.], 3, false)?;
    assert_eq!(scores[0].segment().as_deref(), Some("1-1-1"));
    let value = clv::constant_margin_value(64., 0.81, 0.08)?;
    assert!((value - 192.).abs() < 1e-9);
    Ok(())
}
```

## Choose the model for the data

| Module | Inputs | Outputs and limits |
| --- | --- | --- |
| `rfm` | Recency, frequency and monetary columns over a common window | Relative independent or nested bands, **1 = best**. Segmentation, not lifetime value or a validated targeting policy. |
| `clv` | Constant period margin, retention and discount | Infinite-horizon discounted margin and its margin multiple, with payments beginning at period one. |
| `bgnbd` | Repeat count after acquisition, last purchase time and observation age | Likelihood, bounded fit, probability active and expected future purchases. For irregular, noncontractual repeat purchases. |
| `sbg` | One acquisition cohort's survivors at equal renewal intervals | Retention, survival, bounded fit and expected duration. For observed contractual renewals. |
| `bgbb` | Observed opportunities, purchased opportunities, last purchase opportunity and optional count weights | Likelihood, bounded fit, next-opportunity activity and expected purchased opportunities. For binary noncontractual incidence. |

The module documentation gives parameter definitions, assumptions, numerical limits and a short example for each model. Generate it with `cargo doc --open`. The caller supplies prepared histories and a model appropriate to the data.

## Lifetime example

```rust
use marketing::bgnbd::{self, History, Parameters};

fn main() -> Result<(), marketing::Error> {
    let p = Parameters { r: 0.8, alpha: 7.5, a: 1.4, b: 3.2 };
    let h = History { x: 3, tx: 8.0, t: 12.0, weight: 1.0 };
    let prediction = bgnbd::predict(p, h, 6.0)?;
    assert!(prediction.expected_purchases.is_finite());
    Ok(())
}
```

Use the same time unit for acquisition age, last purchase time and horizon. Include zero-repeat customers. BG/NBD assumes stationary purchase rates and permanent dropout after a purchase, with independent gamma purchase-rate and beta dropout heterogeneity. Its zero-repeat convention gives probability active one. Seasonality, campaign effects, reactivation and spending are outside the model.

sBG assumes each customer's churn probability is constant, with beta heterogeneity. Use actual cohort renewals, keep later acquisitions out and use equal renewal intervals. Its expected duration is unbounded when alpha is at most one. BG/BB assumes independent beta purchase and dropout probabilities. It counts opportunities containing a purchase. Its activity probability includes the dropout decision before the next opportunity.

### Fitting and refusals

`fit` returns a maximum-likelihood **attempt**, including convergence diagnostics. Poorly identified data can fail or reach a parameter bound. Keep that information when inspecting a fit. The fitted object's forecast methods refuse unsuccessful or bounded attempts:

```rust
use marketing::bgnbd::{self, History};

fn main() -> Result<(), marketing::Error> {
    let histories = [
        History { x: 3, tx: 8.0, t: 12.0, weight: 1.0 },
        History { x: 1, tx: 5.0, t: 12.0, weight: 1.0 },
        History { x: 0, tx: 0.0, t: 12.0, weight: 1.0 },
    ];
    let attempt = bgnbd::fit(&histories)?;
    if attempt.converged {
        println!("{}", attempt.predict(histories[0], 6.0)?.expected_purchases);
    }
    Ok(())
}
```

Free functions evaluate explicitly supplied parameters. A converged optimizer does not prove the model fits the business. Check observation quality, variation and held-out predictions before relying on extrapolation. Small cohorts, sparse purchasing and changes in behavior can make answers unreliable. Extreme parameter magnitudes are outside the qualified range. A BG/NBD forecast carries a rounding check. If no series bounds its rounding within its tolerance in 100,000 terms, the call returns an error. Since 0.2.1, `a = 1` answers with the continuous limit of the forecast. Within `|a - 1| <= 1/64`, a large `r` or a very long horizon can return an error. Outside that band the check does not cover the final subtraction, which loses accuracy at horizons far shorter than `alpha + T`. The `bgnbd` module documentation gives the tolerances and the measured figures. Parameter-uncertainty interfaces are deferred.

BG/NBD and sBG apply finite-range admission checks: overflowing parameter/time combinations and nonfinite calculated values return errors. These refusals have separate validation tests. BG/NBD requires a positive observation age; the independent key also includes zero-age inputs to verify that refusal. For fixed-p/q BG/BB, a zero-probability history has no defined conditional activity; that query returns an error. In BG/NBD and BG/BB, a history with no repeat purchases must have last-purchase time zero. A BG/BB shape that rounding removes from `beta + n - x` returns an error. Fits reject unrepresentable weighted likelihoods, and the optimizer cannot declare an all-nonfinite objective converged.

## RFM and constant-margin conventions

RFM sorts recency ascending and frequency and monetary value descending, breaking ties by input row order. For `n` usable rows and `D` distinct values, it uses `B = min(groups, n, D)` bands. Zero-based position `p` gets the first band `b` where `p * B <= (n - 1) * b`, or band `B`. Constant columns get band 1. The accepted scale is 1 through 255.

Independent mode ranks each column over all customers. Nested mode ranks frequency within recency bands, then monetary value within recency/frequency cells. NaN means missing; infinity is ranked. In nested mode, a row missing recency has no frequency band, and a row missing recency or frequency has no monetary band. Each dimension needs a non-NaN value. Empty dimensions, unequal lengths and zero bands return errors. These band, tie and missing-value rules are this implementation's conventions. Bands follow sorted position, so equal values can land in different bands. Value-quantile binning that keeps equal values together, such as pandas `qcut`, gives different bands on tied data. A nested cell can hold a single row, which then gets band 1.

Constant-margin value is `margin * r / (1 + d - r)`, summing payments from period one onward. Retention is between zero and one, discount is greater than minus one, and `r < 1 + d` is required for convergence. Use contribution margin after variable and retention costs, excluding VAT, with one currency and period throughout. Signed finite margin is supported. Acquisition cost is excluded. These are estimates under constant retention, permanent churn and constant margin, with no rounding to accounting units. RFM's "1 = best" convention does not apply to monetary values.

## Sources

The papers below are peer reviewed. RFM began in direct-mail practice; the cited paper provides academic context for customer selection. This module implements descriptive bands, not that paper's profit-optimized selection procedure.

- **RFM context:** Jan Roelf Bult and Tom Wansbeek (1995), "Optimal Selection for Direct Mail," *Marketing Science* 14(4), 378–394. [DOI: 10.1287/mksc.14.4.378](https://doi.org/10.1287/mksc.14.4.378). Exact band, tie and missing-value conventions remain explicit implementation choices. RFM scores alone do not establish who should be contacted.
- **Constant-margin CLV and margin multiple:** Sunil Gupta and Donald R. Lehmann (2003), "Customers as Assets," *Journal of Interactive Marketing* 17(1), 9–24, equation 3. [DOI: 10.1002/dir.10045](https://doi.org/10.1002/dir.10045).
- **BG/NBD:** Peter S. Fader, Bruce G. S. Hardie and Ka Lok Lee (2005), "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model, *Marketing Science* 24(2), 275–284. [DOI: 10.1287/mksc.1040.0098](https://doi.org/10.1287/mksc.1040.0098).
- **sBG:** Peter S. Fader and Bruce G. S. Hardie (2007), "How to Project Customer Retention," *Journal of Interactive Marketing* 21(1), 76–90. [DOI: 10.1002/dir.20074](https://doi.org/10.1002/dir.20074).
- **BG/BB:** Peter S. Fader, Bruce G. S. Hardie and Jen Shang (2010), "Customer-Base Analysis in a Discrete-Time Noncontractual Setting," *Marketing Science* 29(6), 1086–1108. [DOI: 10.1287/mksc.1100.0580](https://doi.org/10.1287/mksc.1100.0580).

## Answer keys

The RFM key uses a seeded synthetic population of 300 customers and 16 hand-built inputs, each scored with 1, 2, 3, 5, 7 and 255 groups in independent and nested mode: 204 cases and 11,076 rows. A standalone generator applies the documented rules with exact rational ordering and integer band arithmetic. Its manifest records the seed and hashes. Constant-margin CLV uses 93 new synthetic cases checked with exact rational arithmetic: for `q = r/(1+d)`, the discounted payment sum is `q/(1-q)`. The generator verifies this closed form against a finite sum plus its exact remainder before writing each reference value.

The BG/NBD and BG/BB keys use new synthetic histories checked with Lifetimes 0.11.3: 72 fixed cases and 1,000 fitting histories per model, plus 16 fixed-p/q cases checked by an independent finite-state calculation. The sBG key uses four fixed parameter sets and two newly generated 2,000-customer cohorts, checked with SciPy's beta-negative-binomial distribution at n=1 and mpmath at high precision. Their manifests record generation seeds, software versions, model sources and tolerances. BG/NBD forecasts have two further keys, each computed as a posterior integral with mpmath at 60 and 90 digits and confirmed by an independent closed-form evaluation at 80 digits: 22 histories with dropout shape `a` at or near one, and 87 histories with large `r`.

The package on crates.io holds the library source, one example and the licence notices. The tests, the answer keys and the scripts that generate them are in the [repository](https://github.com/UlrikErlingsen/marketing-rs), under `tests/` and `tools/`.

Not part of this crate today: discounted lifetime value beyond the constant-margin formula, preparation of transaction or date data, and accounting projections. The only runtime dependency is `statrs`, with default features disabled. The library needs no Python, database or network at run time.
