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

//! Small bounded Nelder-Mead solver in log-parameter space. Scratch is allocated once per start.
#[derive(Clone, Copy)]
struct Point<const N: usize> {
    x: [f64; N],
    cost: f64,
}
pub(crate) struct Solution<const N: usize> {
    pub x: [f64; N],
    pub cost: f64,
    pub converged: bool,
}
pub(crate) fn minimize<const N: usize>(
    starts: &[[f64; N]],
    bounds: (f64, f64),
    mut f: impl FnMut([f64; N]) -> f64,
) -> Solution<N> {
    let evaluate = |x: [f64; N], f: &mut dyn FnMut([f64; N]) -> f64| {
        let x = x.map(|v| v.clamp(bounds.0, bounds.1));
        let value = f(x);
        Point {
            x,
            cost: if value.is_finite() {
                value
            } else {
                f64::INFINITY
            },
        }
    };
    let mut best = Solution {
        x: starts[0],
        cost: f64::INFINITY,
        converged: false,
    };
    for &start in starts {
        let mut simplex = Vec::with_capacity(N + 1);
        simplex.push(evaluate(start, &mut f));
        for axis in 0..N {
            let mut x = start;
            x[axis] += 0.2;
            simplex.push(evaluate(x, &mut f));
        }
        let mut converged = false;
        for _ in 0..3000 {
            simplex.sort_unstable_by(|a, b| a.cost.total_cmp(&b.cost));
            if !simplex[0].cost.is_finite() {
                break;
            }
            let spread = simplex
                .iter()
                .flat_map(|p| p.x.iter().zip(simplex[0].x))
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            if spread < 1e-6
                && (simplex[N].cost - simplex[0].cost).abs() < 1e-10 * (1. + simplex[0].cost.abs())
            {
                converged = true;
                break;
            }
            let centroid: [f64; N] = std::array::from_fn(|axis| {
                simplex[..N].iter().map(|p| p.x[axis]).sum::<f64>() / N as f64
            });
            let reflection = evaluate(
                std::array::from_fn(|i| 2. * centroid[i] - simplex[N].x[i]),
                &mut f,
            );
            if reflection.cost < simplex[0].cost {
                let expansion = evaluate(
                    std::array::from_fn(|i| centroid[i] + 2. * (reflection.x[i] - centroid[i])),
                    &mut f,
                );
                simplex[N] = if expansion.cost < reflection.cost {
                    expansion
                } else {
                    reflection
                };
            } else if reflection.cost < simplex[N - 1].cost {
                simplex[N] = reflection;
            } else {
                let outside = reflection.cost < simplex[N].cost;
                let target = if outside { reflection } else { simplex[N] };
                let contraction = evaluate(
                    std::array::from_fn(|i| centroid[i] + 0.5 * (target.x[i] - centroid[i])),
                    &mut f,
                );
                if contraction.cost < target.cost {
                    simplex[N] = contraction;
                } else {
                    let first = simplex[0].x;
                    for point in simplex.iter_mut().skip(1) {
                        *point = evaluate(
                            std::array::from_fn(|i| first[i] + 0.5 * (point.x[i] - first[i])),
                            &mut f,
                        );
                    }
                }
            }
        }
        simplex.sort_unstable_by(|a, b| a.cost.total_cmp(&b.cost));
        if simplex[0].cost < best.cost {
            best = Solution {
                x: simplex[0].x,
                cost: simplex[0].cost,
                converged,
            };
        }
    }
    best
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_nonfinite_objective_cannot_converge() {
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let solution = super::minimize(&[[0., 0.]], (-12., 12.), |_| value);
            assert!(!solution.converged);
            assert!(!solution.cost.is_finite());
        }
    }
}
