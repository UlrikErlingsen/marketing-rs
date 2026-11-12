//! One call of each lifetime model with supplied parameters.
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

use marketing::{bgbb, bgnbd, sbg};

fn main() -> Result<(), marketing::Error> {
    let prediction = bgnbd::predict(
        bgnbd::Parameters {
            r: 0.8,
            alpha: 7.5,
            a: 1.4,
            b: 3.2,
        },
        bgnbd::History {
            x: 3,
            tx: 8.0,
            t: 12.0,
            weight: 1.0,
        },
        6.0,
    )?;
    println!(
        "Expected repeat purchases: {}",
        prediction.expected_purchases
    );
    let survival = sbg::survival(
        sbg::Parameters {
            alpha: 1.5,
            beta: 4.5,
        },
        2,
    )?;
    println!("Survival after two periods: {}", survival);
    let prediction = bgbb::predict(
        bgbb::Parameters {
            alpha: 1.8,
            beta: 2.4,
            gamma: 1.2,
            delta: 4.1,
        },
        bgbb::History {
            n: 1,
            tx: 0,
            x: 0,
            weight: 1.0,
        },
        5,
    )?;
    println!(
        "Expected purchased opportunities: {}",
        prediction.expected_future_purchases
    );
    Ok(())
}
