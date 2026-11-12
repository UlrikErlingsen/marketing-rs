//! RFM customer segmentation. On each dimension, 1 is best.
//!
//! Sort recency ascending, frequency and monetary descending. Break ties by
//! original row order. For n usable rows and D distinct values, use
//! B = min(groups, n, D) bands. Zero-based position p belongs to the first band
//! b with p * B <= (n - 1) * b, or band B. This is first-rank, right-closed
//! linear quantile scoring. Constant columns get band 1.
//! Bands follow sorted position, so equal values can land in different bands.
//! Value-quantile binning that keeps equal values together, such as pandas
//! `qcut`, gives different bands on tied data.
//!
//! Independent mode ranks each column over all rows. Nested mode ranks F within
//! each R band and M within each R/F cell. In nested mode, a row missing R has
//! no F band, and a row missing R or F has no M band.
//! A nested cell can hold a single row, which then gets band 1, so nested bands
//! in small cells say little.
//! NaN means missing. Infinity is ranked. Signed zeros tie.
//!
//! RFM describes relative purchase patterns, not profit or lifetime value.
//! Use a common observation window and consistent monetary units across rows.
//!
//! Academic context: Jan Roelf Bult and Tom Wansbeek (1995), Optimal Selection
//! for Direct Mail, *Marketing Science* 14(4), 378–394.
//! <https://doi.org/10.1287/mksc.14.4.378>. That paper studies profit-based
//! selection. This module provides descriptive RFM bands, not its optimized
//! selection procedure. Exact quantile, tie and missing-value conventions are
//! documented above and are checked against an independent answer key on
//! synthetic inputs.

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

/// Original row order is retained. On every dimension, 1 means best.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RfmScore {
    /// Recency band. Lower elapsed time is better.
    pub r: Option<u8>,
    /// Frequency band. Higher frequency is better.
    pub f: Option<u8>,
    /// Monetary band. Higher value is better.
    pub m: Option<u8>,
}

impl RfmScore {
    /// The label `R-F-M`, or `None` when any of the three bands is missing.
    /// A display label is allocated only on request, outside the ranking loops.
    ///
    /// ```
    /// use marketing::rfm::RfmScore;
    /// let score = RfmScore { r: Some(1), f: Some(2), m: Some(3) };
    /// assert_eq!(score.segment().as_deref(), Some("1-2-3"));
    /// ```
    pub fn segment(self) -> Option<String> {
        Some(format!("{}-{}-{}", self.r?, self.f?, self.m?))
    }

    /// Missing dimensions or a sum outside the requested u8 output type yield None.
    /// The standard five-band RFM total always fits.
    ///
    /// ```
    /// use marketing::rfm::RfmScore;
    /// assert_eq!(RfmScore { r: Some(1), f: Some(2), m: Some(3) }.total(), Some(6));
    /// ```
    pub fn total(self) -> Option<u8> {
        self.r?.checked_add(self.f?)?.checked_add(self.m?)
    }
}

#[derive(Clone, Copy)]
enum Axis {
    Recency,
    Frequency,
    Monetary,
}

impl Axis {
    fn assign(self, score: &mut RfmScore, band: u8) {
        match self {
            Self::Recency => score.r = Some(band),
            Self::Frequency => score.f = Some(band),
            Self::Monetary => score.m = Some(band),
        }
    }
}

/// Sort valid row indices once. Explicit row-index tie breaking implements
/// rank(method="first") even though the sort itself is unstable/allocation-free.
fn rank_into(
    values: &[f64],
    rows: impl Iterator<Item = usize>,
    groups: u8,
    axis: Axis,
    output: &mut [RfmScore],
    scratch: &mut Vec<usize>,
) {
    scratch.clear();
    scratch.extend(rows.filter(|&row| !values[row].is_nan()));
    scratch.sort_unstable_by(|&a, &b| {
        let order = values[a]
            .partial_cmp(&values[b])
            .expect("NaN rows were excluded");
        let order = if matches!(axis, Axis::Recency) {
            order
        } else {
            order.reverse()
        };
        order.then_with(|| a.cmp(&b))
    });
    let n = scratch.len();
    if n == 0 {
        return;
    }
    let distinct = 1 + scratch
        .windows(2)
        .filter(|pair| values[pair[0]] != values[pair[1]])
        .count();
    let bins = usize::from(groups).min(n).min(distinct);
    // q_j = 1 + (n-1)*j/bins. An integer rank belongs to the first edge >= it.
    // Cross multiplication evaluates that linear quantile exactly, including an
    // integer on an edge in the lower, right-closed bin. Rank 1 is included.
    let mut band = 1usize;
    for (position, &row) in scratch.iter().enumerate() {
        while band < bins
            && (position as u128) * (bins as u128) > ((n - 1) as u128) * (band as u128)
        {
            band += 1;
        }
        axis.assign(&mut output[row], band as u8);
    }
}

/// Counting partition: each group retains original row order. No row-level
/// allocation and no scan of the entire dataset for each R or R/F group.
struct RowGroups {
    rows: Vec<usize>,
    offsets: Vec<usize>,
}

impl RowGroups {
    fn new(scores: &[RfmScore], slots: usize, key: impl Fn(RfmScore) -> usize) -> Self {
        let mut offsets = vec![0usize; slots + 1];
        for &score in scores {
            offsets[key(score) + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut next = offsets[..slots].to_vec();
        let mut rows = vec![0usize; scores.len()];
        for (row, &score) in scores.iter().enumerate() {
            let slot = key(score);
            rows[next[slot]] = row;
            next[slot] += 1;
        }
        Self { rows, offsets }
    }

    fn rows(&self, group: usize) -> &[usize] {
        &self.rows[self.offsets[group]..self.offsets[group + 1]]
    }
}

/// Score borrowed columns, preserving their row order. NaN represents missing.
/// Infinities are ranked as numeric values; +0 and -0 are tied.
/// `groups` supports 1..=255; prefer 5 (totals above 255 cannot fit `total()`).
///
/// Nested mode ranks F within each R band and M within each R/F cell.
/// A customer missing R or F has no M, total or segment.
///
/// # Errors
/// Rejects unequal column lengths, zero groups, or a dimension with no usable
/// values. Partial rows remain in place with missing scores.
///
/// ```
/// use marketing::rfm::scores;
/// let result = scores(&[1., 10., 30.], &[9., 5., 1.], &[100., 50., 10.], 3, false)?;
/// assert_eq!(result[0].segment().as_deref(), Some("1-1-1"));
/// # Ok::<(), marketing::Error>(())
/// ```
pub fn scores(
    recency: &[f64],
    frequency: &[f64],
    monetary: &[f64],
    groups: u8,
    nested: bool,
) -> Result<Vec<RfmScore>, crate::Error> {
    if recency.len() != frequency.len() || recency.len() != monetary.len() {
        return Err(crate::Error("RFM columns must have equal lengths"));
    }
    if groups == 0 {
        return Err(crate::Error("RFM groups must be at least one"));
    }
    if [recency, frequency, monetary]
        .iter()
        .any(|v| v.iter().all(|x| x.is_nan()))
    {
        return Err(crate::Error(
            "Each RFM dimension needs at least one usable value",
        ));
    }
    let mut output = vec![RfmScore::default(); recency.len()];
    let mut scratch = Vec::with_capacity(recency.len());
    rank_into(
        recency,
        0..recency.len(),
        groups,
        Axis::Recency,
        &mut output,
        &mut scratch,
    );
    if !nested {
        rank_into(
            frequency,
            0..frequency.len(),
            groups,
            Axis::Frequency,
            &mut output,
            &mut scratch,
        );
        rank_into(
            monetary,
            0..monetary.len(),
            groups,
            Axis::Monetary,
            &mut output,
            &mut scratch,
        );
        return Ok(output);
    }
    let width = usize::from(groups) + 1;
    {
        let grouped = RowGroups::new(&output, width, |score| usize::from(score.r.unwrap_or(0)));
        for r in 1..width {
            rank_into(
                frequency,
                grouped.rows(r).iter().copied(),
                groups,
                Axis::Frequency,
                &mut output,
                &mut scratch,
            );
        }
    }
    let grouped = RowGroups::new(&output, width * width, |score| {
        usize::from(score.r.unwrap_or(0)) * width + usize::from(score.f.unwrap_or(0))
    });
    for r in 1..width {
        for f in 1..width {
            rank_into(
                monetary,
                grouped.rows(r * width + f).iter().copied(),
                groups,
                Axis::Monetary,
                &mut output,
                &mut scratch,
            );
        }
    }
    Ok(output)
}
