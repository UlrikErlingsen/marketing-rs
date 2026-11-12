//! BG/BB for binary purchase opportunities in a noncontractual setting.
//!
//! Supply `n` observed opportunities after acquisition, `x` opportunities with
//! a purchase and the last purchase opportunity `tx`. Multiple purchases in one
//! opportunity count once. All customers need a consistent opportunity definition.
//! With no purchases, `tx` must be zero.
//! Weights pool identical histories. Single-history equations ignore the weight.
//!
//! While active, each customer has constant purchase probability p. Permanent
//! dropout occurs before each opportunity with probability q. Independent beta
//! distributions describe heterogeneity in p and q. Changing opportunities,
//! seasonality, reactivation and purchase volumes/spend are outside this model.
//! Sparse or short histories can leave four parameters unidentified. A converged
//! optimizer does not establish model validity; inspect held-out performance.
//!
//! P(alive) means entering the next opportunity after its dropout decision. Even
//! a purchase in the final observed opportunity does not imply probability one.
//! Forecasts count purchase opportunities, not transactions, units or money.
//! Extreme parameter magnitudes are not qualified. Discounted extensions and
//! application-specific minimum customer/opportunity counts are not included.
//!
//! Source: Peter S. Fader, Bruce G. S. Hardie and Jen Shang (2010), Customer-Base
//! Analysis in a Discrete-Time Noncontractual Setting, *Marketing Science*
//! 29(6), 1086–1108. <https://doi.org/10.1287/mksc.1100.0580>.
//!
//! ```
//! use marketing::bgbb::{self, History, Parameters};
//! let p = Parameters { alpha: 1.8, beta: 2.4, gamma: 1.2, delta: 4.1 };
//! let h = History { n: 6, tx: 4, x: 2, weight: 1.0 };
//! let result = bgbb::predict(p, h, 5)?;
//! assert!(result.expected_future_purchases.is_finite());
//! # Ok::<(), marketing::Error>(())
//! ```
// SPDX-License-Identifier: MIT OR Apache-2.0
// MIT License
//
// Copyright (c) 2026 Ulrik Øhrn Erlingsen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use crate::optimizer;
use statrs::function::beta::checked_ln_beta;
use std::collections::BTreeMap;
fn invalid(message: &'static str) -> crate::Error {
    crate::Error(message)
}
/// Log beta function. An argument that rounds to zero gives NaN, which every caller reports as an error.
/// `beta + n - x` rounds to zero when beta is below about 1.1e-16 times n and every opportunity has a purchase.
fn ln_beta(a: f64, b: f64) -> f64 {
    checked_ln_beta(a, b).unwrap_or(f64::NAN)
}
#[derive(Debug, Clone, Copy)]
/// Positive model parameters; functions validate them before calculation.
pub struct Parameters {
    /// First beta shape for active purchase probability.
    pub alpha: f64,
    /// Second beta shape for active purchase probability.
    pub beta: f64,
    /// First shape of the beta dropout-probability distribution.
    pub gamma: f64,
    /// Second shape of the beta dropout-probability distribution.
    pub delta: f64,
}
impl Parameters {
    /// Check that model parameters and their required sums are positive and finite.
    pub fn validate(self) -> Result<(), crate::Error> {
        if [
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
            self.alpha + self.beta,
            self.gamma + self.delta,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
        {
            Err(invalid("BG/BB parameters must be positive and finite."))
        } else {
            Ok(())
        }
    }
    fn from_log(z: [f64; 4]) -> Self {
        let [alpha, beta, gamma, delta] = z.map(f64::exp);
        Self {
            alpha,
            beta,
            gamma,
            delta,
        }
    }
}
#[derive(Debug, Clone, Copy)]
/// One summarized history, with an optional frequency weight.
pub struct History {
    /// Number of observed purchase opportunities after acquisition.
    pub n: u32,
    /// Last purchase opportunity after acquisition; zero when x is zero.
    pub tx: u32,
    /// Number of opportunities containing a purchase after acquisition.
    pub x: u32,
    /// Finite nonnegative frequency weight; used only when fitting pooled histories.
    pub weight: f64,
}
impl History {
    /// Check history dimensions, consistency and finite input ranges.
    pub fn validate(self) -> Result<(), crate::Error> {
        if self.n == 0 || self.x > self.tx || self.tx > self.n || (self.x == 0 && self.tx != 0) {
            return Err(invalid(
                "Require n >= 1, 0 <= x <= tx <= n, and tx = 0 when x = 0.",
            ));
        }
        if !self.weight.is_finite() || self.weight < 0. {
            return Err(invalid("BG/BB weights must be finite and nonnegative."));
        }
        Ok(())
    }
}
#[derive(Debug, Clone)]
/// A bounded maximum-likelihood attempt. Inspect convergence; use its checked forecast methods.
pub struct Fit {
    /// Parameters at the best optimizer attempt; check convergence before using them.
    pub parameters: Parameters,
    /// Total frequency-weighted log likelihood at the returned parameters.
    pub log_likelihood: f64,
    /// Whether the optimizer converged without reaching a parameter bound.
    pub converged: bool,
    /// Whether a log parameter reached an optimization bound.
    pub at_bound: bool,
    /// Number of distinct (n, tx, x) histories after pooling.
    pub distinct_histories: usize,
}
#[derive(Debug, Clone, Copy)]
/// Conditional activity probability and expected purchases.
pub struct Prediction {
    /// Probability of entering the next opportunity after its dropout decision.
    pub probability_alive: f64,
    /// Expected number of purchased opportunities in the future horizon.
    pub expected_future_purchases: f64,
}
fn logadd(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}
// Consecutive dropout paths use beta recurrences. No allocation or gamma calls per path.
fn likelihood(p: Parameters, h: History) -> f64 {
    let (n, tx, x) = (f64::from(h.n), f64::from(h.tx), f64::from(h.x));
    let base = ln_beta(p.alpha, p.beta) + ln_beta(p.gamma, p.delta);
    let mut total = ln_beta(p.alpha + x, p.beta + n - x) + ln_beta(p.gamma, p.delta + n) - base;
    if h.n > h.tx {
        let mut term =
            ln_beta(p.alpha + x, p.beta + tx - x) + ln_beta(p.gamma + 1., p.delta + tx) - base;
        for j in 0..h.n - h.tx {
            total = logadd(total, term);
            let j = f64::from(j);
            term += ((p.beta + tx - x + j) / (p.alpha + p.beta + tx + j)).ln()
                + ((p.delta + tx + j) / (p.gamma + p.delta + tx + j + 1.)).ln();
        }
    }
    total
}
/// Log likelihood for one history, ignoring its frequency weight.
pub fn log_likelihood(p: Parameters, h: History) -> Result<f64, crate::Error> {
    p.validate()?;
    h.validate()?;
    let value = likelihood(p, h);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("Non-finite BG/BB likelihood."))
    }
}
fn log_purchase(p: Parameters, h: History) -> f64 {
    ln_beta(p.alpha + f64::from(h.x) + 1., p.beta + f64::from(h.n - h.x))
        - ln_beta(p.alpha, p.beta)
        - likelihood(p, h)
}
/// Evaluate fixed parameters over the supplied nonnegative future horizon.
pub fn predict(p: Parameters, h: History, future_periods: u32) -> Result<Prediction, crate::Error> {
    p.validate()?;
    h.validate()?;
    let n = f64::from(h.n);
    let alive = (ln_beta(p.alpha + f64::from(h.x), p.beta + f64::from(h.n - h.x))
        - ln_beta(p.alpha, p.beta)
        + ln_beta(p.gamma, p.delta + n + 1.)
        - ln_beta(p.gamma, p.delta)
        - likelihood(p, h))
    .exp();
    let mut term =
        (log_purchase(p, h) + ln_beta(p.gamma, p.delta + n + 1.) - ln_beta(p.gamma, p.delta)).exp();
    let mut expected = 0.;
    for k in 0..future_periods {
        expected += term;
        term *= (p.delta + n + 1. + f64::from(k)) / (p.gamma + p.delta + n + 1. + f64::from(k));
    }
    if !alive.is_finite() || !expected.is_finite() || !(0. ..=1. + 1e-10).contains(&alive) {
        return Err(invalid("Non-finite BG/BB forecast."));
    }
    Ok(Prediction {
        probability_alive: alive,
        expected_future_purchases: expected,
    })
}

/// Probability of this history for fixed purchase p and dropout q, each in [0, 1].
pub fn history_probability_no_heterogeneity(
    h: History,
    p: f64,
    q: f64,
) -> Result<f64, crate::Error> {
    h.validate()?;
    if !(0. ..=1.).contains(&p) || !(0. ..=1.).contains(&q) {
        return Err(invalid("p and q must be between 0 and 1."));
    }
    let stay = p.powf(f64::from(h.x))
        * (1. - p).powf(f64::from(h.n - h.x))
        * (1. - q).powf(f64::from(h.n));
    let mut total = stay;
    for j in 0..h.n - h.tx {
        total += p.powf(f64::from(h.x))
            * (1. - p).powf(f64::from(h.tx - h.x) + f64::from(j))
            * q
            * (1. - q).powf(f64::from(h.tx) + f64::from(j));
    }
    Ok(total)
}
/// Next-opportunity alive probability for fixed purchase p and dropout q.
pub fn probability_alive_no_heterogeneity(h: History, p: f64, q: f64) -> Result<f64, crate::Error> {
    let likelihood = history_probability_no_heterogeneity(h, p, q)?;
    if likelihood == 0. {
        return Err(invalid(
            "Conditional activity is undefined for a zero-probability history.",
        ));
    }
    Ok(p.powf(f64::from(h.x))
        * (1. - p).powf(f64::from(h.n - h.x))
        * (1. - q).powf(f64::from(h.n) + 1.)
        / likelihood)
}

/// Fit with bounded multistart likelihood optimization. A failed/bounded attempt has converged=false; use the checked forecast methods.
pub fn fit(histories: &[History]) -> Result<Fit, crate::Error> {
    if histories.is_empty() {
        return Err(invalid("BG/BB needs at least one history."));
    }
    let mut grouped = BTreeMap::new();
    for &h in histories {
        h.validate()?;
        *grouped.entry((h.n, h.tx, h.x)).or_insert(0.) += h.weight;
    }
    let weight: f64 = grouped.values().sum();
    if !weight.is_finite() || weight <= 0. {
        return Err(invalid("BG/BB needs positive finite total history weight."));
    }
    let starts = [
        [1.2, 0.8, 0.7, 2.8],
        [1., 1., 1., 3.],
        [2., 2., 0.5, 2.],
        [0.5, 2., 2., 5.],
    ]
    .map(|p| p.map(f64::ln));
    let solution = optimizer::minimize(&starts, (-12., 12.), |z| {
        let p = Parameters::from_log(z);
        -grouped
            .iter()
            .map(|(&(n, tx, x), &w)| {
                w * likelihood(
                    p,
                    History {
                        n,
                        tx,
                        x,
                        weight: w,
                    },
                )
            })
            .sum::<f64>()
            / weight
    });
    let at_bound = solution
        .x
        .iter()
        .any(|v| *v <= -12. + 1e-4 || *v >= 12. - 1e-4);
    let log_likelihood = -solution.cost * weight;
    if !log_likelihood.is_finite() {
        return Err(crate::Error(
            "The model likelihood could not be calculated. Check for weights or values too large to calculate with.",
        ));
    }
    Ok(Fit {
        parameters: Parameters::from_log(solution.x),
        log_likelihood,
        converged: solution.converged && !at_bound,
        at_bound,
        distinct_histories: grouped.len(),
    })
}

/// Why a fit gives no forecast: a nonfinite likelihood, a parameter bound, or no convergence.
pub fn fit_problem(f: &Fit) -> Option<&'static str> {
    if !f.log_likelihood.is_finite() {
        Some("the fitted likelihood is not finite")
    } else if f.at_bound {
        Some("a parameter sits at the optimiser bound")
    } else if !f.converged {
        Some("optimiser reported failure")
    } else {
        None
    }
}

impl Fit {
    /// Evaluate this fitted model only when its optimization converged away from bounds.
    pub fn predict(&self, history: History, horizon: u32) -> Result<Prediction, crate::Error> {
        if fit_problem(self).is_some() {
            return Err(crate::Error(
                "The model fit failed or reached a parameter bound; no forecast is available.",
            ));
        }
        predict(self.parameters, history, horizon)
    }
}
