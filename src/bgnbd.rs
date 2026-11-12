//! BG/NBD for irregular repeat purchases, without a spending model.
//!
//! Supply repeat count `x` after acquisition, time `tx` from acquisition to the
//! last purchase, and observation age `t`, in one time unit. Include customers
//! with no repeats. History weights pool identical customers; a single-history
//! likelihood or prediction does not multiply its answer by that weight.
//! Observation age must be positive, with 0 <= tx <= t. Prior-only, zero-age
//! histories are outside this API's accepted range. With no repeats, `tx` must be zero.
//!
//! Active customers buy according to a stationary Poisson process with gamma
//! heterogeneity. Permanent dropout occurs after purchases with beta-distributed
//! probability, independently of purchase rate. Seasonality, marketing effects,
//! changing customer behavior and reactivation are outside the model. Do not use
//! it for contractual cancellations. Estimates require representative histories
//! and enough repeat-purchase variation; a successful optimizer is not evidence
//! that these assumptions hold. Extreme parameter magnitudes are not qualified.
//!
//! The zero-repeat convention gives probability active = 1. A zero horizon has
//! zero expected purchases. Outputs are conditional means, not individual
//! prediction intervals or monetary values.
//!
//! # Numerical limits of the forecast
//!
//! The forecast is equation 10 of the source, evaluated in one of three ways.
//!
//! - Within `|a - 1| <= 1/64`, a series with the factor `a - 1` removed. At
//!   `a = 1` it gives the continuous limit. It returns an error when its rounding
//!   estimate exceeds 1e-11 of the expected purchases if active.
//! - Elsewhere, the Euler-transformed hypergeometric series, when a bound on its
//!   rounding error is within 1e-9 of the larger operand of the final
//!   subtraction `1 - weight * series`.
//! - Otherwise the untransformed series, which answers only when its rounding
//!   bound is within 1e-9 of the result. This happens when `r` is much larger
//!   than `a + b + x - 1`.
//!
//! Each series stops at 100,000 terms. If no series bounds its rounding within
//! its tolerance in that many terms, the call returns an error. Nonrepresentable
//! time combinations and nonfinite calculated values also return errors.
//!
//! Inside the band around `a = 1`, a large `r`, or a long horizon that brings
//! `horizon / (alpha + T + horizon)` close to one, can return an error. Version
//! 0.2.0 answered some of the long-horizon cases correctly. For example,
//! `r = 0.1704`, `alpha = 5.92`, `a = 1.012`, `b = 7.727`, `x = 0`, `T = 2.383`
//! answers at horizon 100 and returns an error from horizon 108. At horizon 200,
//! version 0.2.0 returned 2.14676511, a relative error of 1.4e-11.
//!
//! The rounding check of the second way does not cover its final subtraction,
//! which loses accuracy when `a` is close to the band or the horizon is far
//! shorter than `alpha + T`. The figures below are the largest relative errors
//! against 60-digit answers over 600 pseudo-random inputs: `r` from 0.05 to 10,
//! `alpha` from 0.5 to 100, `b` from 0.2 to 20, `T` from 4 to 300, `x` up to 20
//! and, unless stated, horizons from 1 to 520. They depend on the sample, so read
//! them as orders of magnitude.
//!
//! - At `|a - 1| = 0.02`, just outside the band: 2e-10. At `|a - 1| = 0.2`: 1e-11.
//! - At a horizon of one ten-thousandth of `alpha + T`: 1e-9. At one millionth:
//!   8e-8. At one hundred-millionth: 8e-6.
//! - Version 0.2.0 inside the band, which 0.2.1 repairs: 2e-8 at
//!   `|a - 1| = 1e-4`, 2e-4 at `1e-8`, and errors larger than the forecast itself
//!   at `1e-12`.
//!
//! At large `r`, outside the band and outside that sample, version 0.2.0 could
//! return a wrong number without an error: 19.89 where the model gives 12.977,
//! for `r = 200`, `alpha = 200`, `a = 0.8`, `b = 2.5`, `x = 0`, `T = 52` at
//! horizon 104.
//!
//! Source: Peter S. Fader, Bruce G. S. Hardie and Ka Lok Lee (2005),
//! "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model,
//! *Marketing Science* 24(2), 275–284. <https://doi.org/10.1287/mksc.1040.0098>.
//!
//! ```
//! use marketing::bgnbd::{self, History, Parameters};
//! let p = Parameters { r: 0.8, alpha: 7.5, a: 1.4, b: 3.2 };
//! let h = History { x: 3, tx: 8.0, t: 12.0, weight: 1.0 };
//! let result = bgnbd::predict(p, h, 6.0)?;
//! assert!(result.expected_purchases.is_finite());
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
use statrs::function::gamma::ln_gamma;
#[derive(Debug, Clone, Copy)]
/// Positive model parameters; functions validate them before calculation.
pub struct Parameters {
    /// Gamma shape of the active purchase-rate distribution.
    pub r: f64,
    /// Gamma rate parameter for purchase-rate heterogeneity, in the history time unit.
    pub alpha: f64,
    /// First shape of the beta dropout-probability distribution.
    pub a: f64,
    /// Second shape of the beta dropout-probability distribution.
    pub b: f64,
}
impl Parameters {
    /// Check that model parameters and their required sums are positive and finite.
    pub fn validate(self) -> Result<(), crate::Error> {
        if self.array().iter().any(|v| !v.is_finite() || *v <= 0.) || !(self.a + self.b).is_finite()
        {
            Err(crate::Error(
                "BG/NBD parameters must be positive and finite, with a finite dropout-shape sum",
            ))
        } else {
            Ok(())
        }
    }
    pub(crate) fn array(self) -> [f64; 4] {
        [self.r, self.alpha, self.a, self.b]
    }
    pub(crate) fn from_log(z: [f64; 4]) -> Self {
        let [r, alpha, a, b] = z.map(f64::exp);
        Self { r, alpha, a, b }
    }
}
#[derive(Debug, Clone, Copy)]
/// One summarized history, with an optional frequency weight.
pub struct History {
    /// Number of repeat purchases after acquisition.
    pub x: u32,
    /// Time from acquisition to the last repeat purchase; zero when x is zero.
    pub tx: f64,
    /// Observation age in the same time unit as tx and the forecast horizon.
    pub t: f64,
    /// Finite nonnegative frequency weight; used only when fitting pooled histories.
    pub weight: f64,
}
impl History {
    /// Check history dimensions, consistency and finite input ranges.
    pub fn validate(self) -> Result<(), crate::Error> {
        if !self.tx.is_finite()
            || !self.t.is_finite()
            || !self.weight.is_finite()
            || self.tx < 0.
            || self.tx > self.t
            || self.t <= 0.
            || self.weight < 0.
        {
            Err(crate::Error(
                "BG/NBD requires x>=0, 0<=tx<=T, T>0 and finite nonnegative weights",
            ))
        } else if self.x == 0 && self.tx != 0. {
            Err(crate::Error("BG/NBD: tx must be 0 when x = 0."))
        } else {
            Ok(())
        }
    }
}
#[derive(Debug, Clone, Copy)]
/// Conditional activity probability and expected purchases.
pub struct Prediction {
    /// Conditional probability the customer is still active.
    pub probability_active: f64,
    /// Expected purchases over the horizon conditional on being active.
    pub expected_if_active: f64,
    /// Expected purchases over the horizon given the history: `probability_active * expected_if_active`.
    pub expected_purchases: f64,
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
}
fn logadd(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}
/// Log likelihood for one history, ignoring its frequency weight.
pub fn log_likelihood(p: Parameters, h: History) -> Result<f64, crate::Error> {
    p.validate()?;
    h.validate()?;
    if !(p.alpha + h.t).is_finite() {
        return Err(crate::Error(
            "The BG/NBD time combination exceeds the finite numerical range.",
        ));
    }
    let x = f64::from(h.x);
    let a1 = ln_gamma(p.r + x) - ln_gamma(p.r) + p.r * p.alpha.ln();
    let a2 = ln_gamma(p.a + p.b) + ln_gamma(p.b + x) - ln_gamma(p.b) - ln_gamma(p.a + p.b + x);
    let a3 = -(p.r + x) * (p.alpha + h.t).ln();
    let a4 = if h.x > 0 {
        p.a.ln() - (p.b + x - 1.).ln() - (p.r + x) * (p.alpha + h.tx).ln()
    } else {
        f64::NEG_INFINITY
    };
    let value = a1 + a2 + logadd(a3, a4);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(crate::Error(
            "The BG/NBD likelihood exceeds the finite numerical range.",
        ))
    }
}
/// Largest accepted rounding bound of a forecast series. See the module documentation.
const SERIES_TOLERANCE: f64 = 1e-9;
/// Rounding of one series step: eight rounded operations in the term recurrence and one in the sum, each at most
/// half an epsilon, so k+1 steps lose at most 4.5*(k+1) epsilons of the absolute sum. Eight epsilons per step,
/// counted over k+2 steps, leaves room for the second-order terms.
const ROUNDING: f64 = 8. * f64::EPSILON;
/// The Euler-transformed series 2F1(a, b; c; z), with a bound on the rounding error of the returned sum.
fn hypergeometric(a: f64, b: f64, c: f64, z: f64) -> Option<(f64, f64)> {
    let (mut term, mut sum, mut absolute) = (1., 1., 1.);
    for k in 0..100000 {
        let k = k as f64;
        let denominator = (c + k) * (k + 1.);
        if denominator == 0. {
            return None;
        }
        term *= (a + k) * (b + k) / denominator * z;
        sum += term;
        absolute += f64::abs(term);
        if !sum.is_finite() {
            return None;
        }
        if term.abs() <= 2e-15 * sum.abs() {
            return Some((sum, ROUNDING * (k + 2.) * absolute));
        }
    }
    None
}
/// 2^600, an exact power of two, so rescaling by it rounds nothing.
const RESCALE: f64 = f64::from_bits(0x6570_0000_0000_0000);
const LOG_RESCALE: f64 = 600. * std::f64::consts::LN_2;
/// The direct series never runs longer than the Euler series may.
const DIRECT_TERMS: u32 = 100000;
/// `1 - w * 2F1(a, b; s-1; z)` with `w = exp(log_weight)`, from the untransformed series of Fader et al. (2005),
/// eq. 10. From the second term on, every term has the same sign, so the sum does not cancel the way the Euler series
/// does at large r. Refuses unless the rounding bound of `w * 2F1` is within the tolerance of the returned value.
///
/// The third argument is s = c+1, and the series forms c+k as s+(k-1). With x = 0 and a tiny a+b, c is close to -1
/// and c+1 would keep only the rounding of a+b, in the denominator of every term after the first.
fn direct_bracket(a: f64, b: f64, s: f64, z: f64, log_weight: f64) -> Option<f64> {
    let (mut term, mut sum, mut absolute, mut rescales) = (1.0_f64, 1.0_f64, 1.0_f64, 0.0_f64);
    for k in 0..DIRECT_TERMS {
        let k = f64::from(k);
        let denominator = (s + (k - 1.)) * (k + 1.);
        if denominator == 0. {
            return None;
        }
        term *= (a + k) * (b + k) / denominator * z;
        sum += term;
        absolute += term.abs();
        if !absolute.is_finite() {
            return None;
        }
        // The terms can exceed the binary64 range before they shrink. Keep them in range; the weight absorbs the
        // scale.
        if absolute > RESCALE {
            term /= RESCALE;
            sum /= RESCALE;
            absolute /= RESCALE;
            rescales += 1.;
        }
        // Every later term ratio is at most `bound`, because both factors fall towards one or stay below it. The
        // remaining tail is then at most term * bound / (1 - bound). The first test is a cheap necessary condition.
        let next = k + 1.;
        if term.abs() * z <= 1e-17 * absolute {
            // c + next = s + k, which is positive because s is.
            let bound = z * ((a + next) / (next + 1.)).max(1.) * ((b + next) / (s + k)).max(1.);
            if bound < 1. && term.abs() * bound <= 1e-17 * absolute * (1. - bound) {
                let exponent = rescales * LOG_RESCALE;
                let weighted = (log_weight + exponent).exp() * sum;
                let bracket = 1. - weighted;
                // Relative rounding of the sum, and of the exponent, which becomes relative once exponentiated.
                let rounding = ROUNDING * (k + 2.) * absolute / sum.abs()
                    + ROUNDING * (log_weight.abs() + exponent + 1.);
                let certified = rounding * weighted.abs() <= SERIES_TOLERANCE * bracket.abs();
                return (certified && bracket.is_finite()).then_some(bracket);
            }
        }
    }
    None
}
/// The bracket of eq. 10, `1 - (1-z)^(r+x) * 2F1(r+x, b+x; c; z)`. The Euler series answers when the rounding bound
/// of its weighted sum is within the tolerance of the larger operand of the subtraction, with the operations and
/// their order unchanged. Otherwise the direct series answers, or refuses.
fn bracket(
    p: Parameters,
    h: History,
    horizon: f64,
    z: f64,
    euler: Option<(f64, f64)>,
) -> Option<f64> {
    if let Some((hyper, rounding)) = euler {
        let weight = (1. - z).powf(p.a - 1.);
        let product = weight * hyper;
        let unstable = weight * rounding > SERIES_TOLERANCE * product.abs().max(1.);
        if !unstable {
            return Some(1. - product);
        }
    }
    let x = f64::from(h.x);
    let log_weight = -(p.r + x) * (horizon / (p.alpha + h.t)).ln_1p();
    direct_bracket(p.r + x, p.b + x, p.a + p.b + x, z, log_weight)
}
/// Evaluate fixed parameters over the supplied nonnegative future horizon.
pub fn predict(p: Parameters, h: History, horizon: f64) -> Result<Prediction, crate::Error> {
    p.validate()?;
    h.validate()?;
    if !horizon.is_finite() || horizon < 0. {
        return Err(crate::Error(
            "future horizon must be finite and nonnegative",
        ));
    }
    if !(p.alpha + h.t + horizon).is_finite() {
        return Err(crate::Error(
            "The BG/NBD time combination exceeds the finite numerical range.",
        ));
    }
    let x = f64::from(h.x);
    let log_factor = if h.x == 0 {
        f64::NEG_INFINITY
    } else {
        p.a.ln() - (p.b + x - 1.).ln() + (p.r + x) * ((p.alpha + h.t) / (p.alpha + h.tx)).ln()
    };
    let alive = if log_factor > 0. {
        let e = (-log_factor).exp();
        e / (1. + e)
    } else {
        1. / (1. + log_factor.exp())
    };
    if !alive.is_finite() {
        return Err(crate::Error(
            "The BG/NBD activity probability exceeds the finite numerical range.",
        ));
    }
    if horizon == 0. {
        return Ok(Prediction {
            probability_active: alive,
            expected_if_active: 0.,
            expected_purchases: 0.,
        });
    }
    let z = horizon / (p.alpha + h.t + horizon);
    if near_one(p) {
        let series = near_sum(p, h, z).ok_or(crate::Error(
            "unstable BG/NBD near-one series or numerical range",
        ))?;
        return near_prediction(p, h, alive, horizon, series);
    }
    let c = p.a + p.b + x - 1.;
    // Euler transform absorbs (1-z)^(r+x), avoiding enormous intermediate series for high x.
    let bracket = bracket(
        p,
        h,
        horizon,
        z,
        hypergeometric(p.a + p.b - p.r - 1., p.a - 1., c, z),
    )
    .ok_or(crate::Error("unstable BG/NBD hypergeometric series"))?;
    let conditional = c / (p.a - 1.) * bracket;
    let expected = alive * conditional;
    if !expected.is_finite() || !conditional.is_finite() || expected < -1e-8 {
        return Err(crate::Error("unstable BG/NBD forecast"));
    }
    Ok(Prediction {
        probability_active: alive,
        expected_if_active: conditional.max(0.),
        expected_purchases: expected.max(0.),
    })
}
// This local branch avoids division by q in the cancelled numerator of Fader et al. (2005), eq. 10.
// The exact binary threshold covers both sides of one while leaving ordinary forecasts unchanged.
const NEAR_ONE: f64 = 1. / 64.;
fn near_one(p: Parameters) -> bool {
    (p.a - 1.).abs() <= NEAR_ONE
}
fn near_shapes(p: Parameters, h: History) -> (f64, f64, f64) {
    let q = p.a - 1.;
    (q + p.b - p.r, q, q + (p.b + f64::from(h.x)))
}
fn near_coefficient(p: Parameters, h: History, k: usize) -> Option<f64> {
    let (a, q, c) = near_shapes(p, h);
    // c*S has no initial 1/c. Preserve refusal of the unrelated c=0 hypergeometric pole.
    if !a.is_finite() || !c.is_finite() || c == 0. {
        return None;
    }
    if k == 0 {
        return Some(a);
    }
    let k = k as f64;
    let denominator = (c + k) * (k + 1.);
    (denominator != 0. && denominator.is_finite()).then(|| (a + k) * (q + k) / denominator)
}
#[derive(Clone, Copy)]
struct NearSum {
    term: f64,
    sum: f64,
    compensation: f64,
    absolute_sum: f64,
    error: f64,
}
impl NearSum {
    fn new() -> Self {
        Self {
            term: 1.,
            sum: 0.,
            compensation: 0.,
            absolute_sum: 0.,
            error: 0.,
        }
    }
    fn step(&mut self, coefficient: f64, z: f64, k: usize, a: f64, c: f64) -> Option<bool> {
        self.term *= coefficient * z;
        let adjusted = self.term - self.compensation;
        let next = self.sum + adjusted;
        self.compensation = (next - self.sum) - adjusted;
        self.sum = next;
        self.absolute_sum += self.term.abs();
        if !self.sum.is_finite() || !self.absolute_sum.is_finite() {
            return None;
        }
        let n = (k + 1) as f64;
        // After the sign-changing factors, every subsequent ratio is in [0,z]:
        // A <= c and q < 1. Bound the entire tail, not just the last term.
        let tail = if self.term == 0. {
            0.
        } else if a + n >= 0. && c + n > 0. {
            self.term.abs() * z / (1. - z)
        } else {
            return Some(false);
        };
        self.error = tail + 8. * f64::EPSILON * (n + 1.) * self.absolute_sum;
        Some(tail <= 2e-16 * self.sum.abs())
    }
}
fn near_sum(p: Parameters, h: History, z: f64) -> Option<NearSum> {
    if !(z > 0. && z < 1.) {
        return None;
    }
    let (a, _, c) = near_shapes(p, h);
    let mut sum = NearSum::new();
    for k in 0..100000 {
        if sum.step(near_coefficient(p, h, k)?, z, k, a, c)? {
            return Some(sum);
        }
    }
    None
}
fn near_prediction(
    p: Parameters,
    h: History,
    alive: f64,
    horizon: f64,
    series: NearSum,
) -> Result<Prediction, crate::Error> {
    let (_, q, c) = near_shapes(p, h);
    // Equal to log1p(-z), without losing the small 1-z at large finite horizons.
    let log = -(horizon / (p.alpha + h.t)).ln_1p();
    let w = q * log;
    let exprel = if w == 0. { 1. } else { w.exp_m1() / w };
    // 2F1(A,q;c;z) = 1 + q*S. At q=0 the limit is -c*log(1-z) - c*S.
    // Accumulate c*S directly, and never form 1 - exp(q*log)*2F1.
    let first = -c * log * exprel;
    let second = w.exp() * series.sum;
    let conditional = first - second;
    let error = w.exp() * series.error + 16. * f64::EPSILON * (first.abs() + second.abs());
    let expected = alive * conditional;
    if !conditional.is_normal()
        || conditional <= 0.
        || !expected.is_finite()
        || (alive > 0. && expected < f64::MIN_POSITIVE)
        || !error.is_finite()
        || error > 1e-11 * conditional
    {
        return Err(crate::Error(
            "unstable BG/NBD near-one series or numerical range",
        ));
    }
    Ok(Prediction {
        probability_active: alive,
        expected_if_active: conditional,
        expected_purchases: expected,
    })
}
/// Fit with bounded multistart likelihood optimization. A failed/bounded attempt has converged=false; use the checked forecast methods.
pub fn fit(histories: &[History]) -> Result<Fit, crate::Error> {
    if histories.is_empty() {
        return Err(crate::Error("no BG/NBD histories"));
    }
    for &h in histories {
        h.validate()?;
    }
    let weight: f64 = histories.iter().map(|h| h.weight).sum();
    if !weight.is_finite() || weight <= 0. {
        return Err(crate::Error("no positive finite history weight"));
    }
    let mut times: Vec<_> = histories.iter().map(|h| h.t).collect();
    times.sort_unstable_by(f64::total_cmp);
    let n = times.len();
    let median = if n % 2 == 1 {
        times[n / 2]
    } else {
        (times[n / 2 - 1] + times[n / 2]) / 2.
    };
    let starts = [
        [0.5, median / 2., 1.5, 3.],
        [1., median, 2., 4.],
        [0.2, median / 10., 0.8, 2.],
        [2., median * 2., 3., 3.],
    ]
    .map(|p| p.map(f64::ln));
    let max_x = histories
        .iter()
        .map(|h| h.x as usize)
        .max()
        .unwrap()
        .min(512);
    let (mut r_cache, mut beta_cache) = (vec![0.; max_x + 1], vec![0.; max_x + 1]);
    let mut objective = |z: [f64; 4]| {
        let p = Parameters::from_log(z);
        r_cache[0] = 0.;
        beta_cache[0] = 0.;
        for k in 1..=max_x {
            let j = (k - 1) as f64;
            r_cache[k] = r_cache[k - 1] + (p.r + j).ln();
            beta_cache[k] = beta_cache[k - 1] + (p.b + j).ln() - (p.a + p.b + j).ln();
        }
        -histories
            .iter()
            .map(|&h| {
                let x = f64::from(h.x);
                let ll = if h.x as usize <= max_x {
                    let a3 = -(p.r + x) * (p.alpha + h.t).ln();
                    let a4 = if h.x > 0 {
                        p.a.ln() - (p.b + x - 1.).ln() - (p.r + x) * (p.alpha + h.tx).ln()
                    } else {
                        f64::NEG_INFINITY
                    };
                    r_cache[h.x as usize]
                        + p.r * p.alpha.ln()
                        + beta_cache[h.x as usize]
                        + logadd(a3, a4)
                } else {
                    log_likelihood(p, h).unwrap_or(f64::NEG_INFINITY)
                };
                h.weight * ll
            })
            .sum::<f64>()
            / weight
    };
    let solution = optimizer::minimize(&starts, (-12., 16.), &mut objective);
    let parameters = Parameters::from_log(solution.x);
    let bound = solution
        .x
        .iter()
        .any(|v| *v <= -12. + 1e-4 || *v >= 16. - 1e-4);
    let converged = solution.converged && !bound;
    let log_likelihood = -solution.cost * weight;
    if !log_likelihood.is_finite() {
        return Err(crate::Error(
            "The model likelihood could not be calculated. Check for weights or values too large to calculate with.",
        ));
    }
    Ok(Fit {
        parameters,
        log_likelihood,
        converged,
    })
}

impl Fit {
    /// Evaluate this fitted model only when its optimization converged away from bounds.
    pub fn predict(&self, history: History, horizon: f64) -> Result<Prediction, crate::Error> {
        if !self.converged || !self.log_likelihood.is_finite() {
            return Err(crate::Error(
                "The model fit failed or reached a parameter bound; no forecast is available.",
            ));
        }
        predict(self.parameters, history, horizon)
    }
}
