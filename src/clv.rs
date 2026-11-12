//! Constant-margin customer lifetime value, with payments at period ends.
//!
//! CLV = margin * r / (1 + d - r), the sum of margin * (r / (1 + d))^t
//! for t = 1..infinity. Retention r, discount d and margin use the same period.
//! Assumes constant margin and retention, with permanent churn. Requires
//! 0 <= r <= 1, d > -1 and r < 1 + d. No payment occurs at time zero.
//!
//! Margin means contribution after variable and retention costs, excluding VAT.
//! Revenue is not margin. Supply one currency and unit. Acquisition cost is
//! excluded. The result is an estimate under these assumptions, not observed profit.
//! No currency conversion or rounding to accounting units occurs here.
//! RFM's "1 = best" convention does not apply to monetary values.
//!
//! Source: Sunil Gupta and Donald R. Lehmann (2003), "Customers as Assets",
//! Journal of Interactive Marketing 17(1), 9-24, equation 3,
//! <https://doi.org/10.1002/dir.10045>.

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

fn invalid(message: &'static str) -> crate::Error {
    crate::Error(message)
}
fn rates(r: f64, d: f64) -> Result<(), crate::Error> {
    if !r.is_finite() || !(0. ..=1.).contains(&r) {
        return Err(invalid("Retention must be between 0 and 1."));
    }
    if !d.is_finite() || d <= -1. {
        return Err(invalid("Discount rate must be greater than -100%."));
    }
    Ok(())
}
/// Expected future margin as a multiple of one period's margin.
///
/// Returns an error for invalid rates, a divergent sum or nonfinite output.
///
/// ```
/// let multiple = marketing::clv::infinite_multiple(0.81, 0.08)?;
/// assert!((multiple - 3.0).abs() < 1e-12);
/// # Ok::<(), marketing::Error>(())
/// ```
pub fn infinite_multiple(retention: f64, discount: f64) -> Result<f64, crate::Error> {
    rates(retention, discount)?;
    let denominator = 1. + discount - retention;
    if denominator <= 0. {
        return Err(invalid(
            "The infinite-horizon CLV does not converge for these inputs.",
        ));
    }
    let value = retention / denominator;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(
            "Insufficient precision for the infinite value multiple.",
        ))
    }
}

/// Expected discounted future margin, before acquisition cost.
///
/// Uses [`infinite_multiple`]. Margin may be negative but must be finite.
/// Returns an error for invalid rates, nonfinite margin or overflow.
///
/// ```
/// let value = marketing::clv::constant_margin_value(64.0, 0.81, 0.08)?;
/// assert!((value - 192.0).abs() < 1e-9);
/// # Ok::<(), marketing::Error>(())
/// ```
pub fn constant_margin_value(
    margin: f64,
    retention: f64,
    discount: f64,
) -> Result<f64, crate::Error> {
    if !margin.is_finite() {
        return Err(invalid("Margin must be finite."));
    }
    let value = margin * infinite_multiple(retention, discount)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(
            "Customer value exceeds the finite numerical range.",
        ))
    }
}
