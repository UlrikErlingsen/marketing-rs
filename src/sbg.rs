//! Shifted beta-geometric retention for one observed acquisition cohort.
//!
//! Supply nonincreasing survivor counts at equal renewal intervals, beginning
//! with the original cohort size. Do not add later acquisitions. These must be
//! actual renewals/cancellations, not a curve inferred from irregular purchases.
//! Each customer has a constant churn probability with beta heterogeneity.
//! Aggregate retention rises through selection, not improving individual loyalty.
//! Reactivation and changing renewal opportunities are outside the model.
//!
//! Three counts (periods 0, 1 and 2) are the minimum accepted input, not a
//! guarantee of a reliable fit. Small cohorts or little churn may not identify
//! parameters. Check fit convergence before extrapolation. Extreme parameter
//! magnitudes are not qualified. Overflowing parameter sums and nonfinite
//! calculated values are refused. Lifetime includes the initial period and is
//! unbounded when alpha <= 1; that answer is `None`. Parameter confidence
//! intervals are not exposed by this module.
//!
//! Source: Peter S. Fader and Bruce G. S. Hardie (2007), How to Project Customer
//! Retention, *Journal of Interactive Marketing* 21(1), 76–90.
//! <https://doi.org/10.1002/dir.20074>.
//!
//! ```
//! use marketing::sbg::{self, Parameters};
//! let p = Parameters { alpha: 1.5, beta: 4.5 };
//! assert!((sbg::survival(p, 2)? - 0.5892857142857143).abs() < 1e-12);
//! assert_eq!(sbg::expected_lifetime(p)?, Some(10.0));
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
#[derive(Debug, Clone, Copy)]
/// Positive model parameters; functions validate them before calculation.
pub struct Parameters {
    /// First beta shape for individual churn probability.
    pub alpha: f64,
    /// Second beta shape for individual churn probability.
    pub beta: f64,
}
impl Parameters {
    fn validate(self) -> Result<(), crate::Error> {
        if !self.alpha.is_finite()
            || !self.beta.is_finite()
            || self.alpha <= 0.
            || self.beta <= 0.
            || !(self.alpha + self.beta).is_finite()
        {
            Err(crate::Error(
                "sBG parameters must be positive and finite, with a finite sum",
            ))
        } else {
            Ok(())
        }
    }
    fn from_log(z: [f64; 2]) -> Self {
        Self {
            alpha: z[0].exp(),
            beta: z[1].exp(),
        }
    }
}
#[derive(Debug, Clone)]
/// A bounded maximum-likelihood attempt. Inspect convergence; use its checked forecast methods.
pub struct Fit {
    /// Parameters at the best optimizer attempt; check convergence before using them.
    pub parameters: Parameters,
    /// Total cohort log likelihood at the returned parameters.
    pub log_likelihood: f64,
    /// Whether the optimizer converged without reaching a parameter bound.
    pub converged: bool,
}
fn finite(value: f64) -> Result<f64, crate::Error> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(crate::Error(
            "The sBG calculation exceeds the finite numerical range.",
        ))
    }
}
/// Probability of remaining in the original cohort after the supplied renewal period.
pub fn survival(p: Parameters, period: u32) -> Result<f64, crate::Error> {
    p.validate()?;
    finite(
        (1..=period)
            .map(|t| ((p.beta + f64::from(t) - 1.) / (p.alpha + p.beta + f64::from(t) - 1.)).ln())
            .sum::<f64>()
            .exp(),
    )
}
/// Conditional retention at a renewal opportunity (period starts at one).
pub fn retention(p: Parameters, period: u32) -> Result<f64, crate::Error> {
    p.validate()?;
    if period == 0 {
        return Err(crate::Error("renewal period starts at one"));
    }
    let t = f64::from(period);
    finite((p.beta + t - 1.) / (p.alpha + p.beta + t - 1.))
}
/// Expected total duration including the initial period, or None when the mean diverges.
pub fn expected_lifetime(p: Parameters) -> Result<Option<f64>, crate::Error> {
    p.validate()?;
    (p.alpha > 1.)
        .then(|| finite((p.alpha + p.beta - 1.) / (p.alpha - 1.)))
        .transpose()
}
fn validate(s: &[u64]) -> Result<(), crate::Error> {
    if s.len() < 3 || s[0] == 0 || s.windows(2).any(|w| w[1] > w[0]) {
        Err(crate::Error(
            "sBG needs periods 0..2, positive initial size and non-increasing survivors",
        ))
    } else {
        Ok(())
    }
}
/// Log likelihood of the supplied survivor cohort, including final-period censoring.
pub fn log_likelihood(p: Parameters, s: &[u64]) -> Result<f64, crate::Error> {
    p.validate()?;
    validate(s)?;
    let (mut log_survival, mut ll) = (0., 0.);
    for t in 1..s.len() {
        let denominator = p.alpha + p.beta + t as f64 - 1.;
        let failure = log_survival + (p.alpha / denominator).ln();
        ll += (s[t - 1] - s[t]) as f64 * failure;
        log_survival += ((p.beta + t as f64 - 1.) / denominator).ln();
    }
    finite(ll + s[s.len() - 1] as f64 * log_survival)
}
/// Fit with bounded multistart likelihood optimization. A failed/bounded attempt has converged=false; use the checked forecast methods.
pub fn fit(s: &[u64]) -> Result<Fit, crate::Error> {
    validate(s)?;
    let n = s[0] as f64;
    let starts = [[1f64, 1.], [0.5, 2.], [2., 5.], [0.2, 0.8]].map(|p| p.map(f64::ln));
    let objective =
        |z| -log_likelihood(Parameters::from_log(z), s).unwrap_or(f64::NEG_INFINITY) / n;
    let solution = optimizer::minimize(&starts, (-12., 12.), objective);
    let converged = solution.converged
        && !solution
            .x
            .iter()
            .any(|v| *v <= -12. + 1e-4 || *v >= 12. - 1e-4);
    let log_likelihood = -solution.cost * n;
    if !log_likelihood.is_finite() {
        return Err(crate::Error(
            "The model likelihood could not be calculated. Check for cohort counts or values too large to calculate with.",
        ));
    }
    Ok(Fit {
        parameters: Parameters::from_log(solution.x),
        log_likelihood,
        converged,
    })
}

impl Fit {
    fn check(&self) -> Result<(), crate::Error> {
        if self.converged && self.log_likelihood.is_finite() {
            Ok(())
        } else {
            Err(crate::Error(
                "The model fit failed or reached a parameter bound; no forecast is available.",
            ))
        }
    }
    /// Forecast survival only after a converged fit away from parameter bounds.
    pub fn survival(&self, period: u32) -> Result<f64, crate::Error> {
        self.check()?;
        survival(self.parameters, period)
    }
    /// Forecast conditional renewal retention only after a converged fit.
    pub fn retention(&self, period: u32) -> Result<f64, crate::Error> {
        self.check()?;
        retention(self.parameters, period)
    }
    /// Evaluate fitted expected duration only after a converged fit.
    pub fn expected_lifetime(&self) -> Result<Option<f64>, crate::Error> {
        self.check()?;
        expected_lifetime(self.parameters)
    }
}
