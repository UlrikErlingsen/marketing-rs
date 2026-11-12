# Synthetic sBG references

This package contains four fixed parameter cases and two synthetic fits for the shifted-beta-geometric distribution. Each fit uses 2,000 independently generated customer durations and observes 20 renewal opportunities. Only aggregate counts enter the output.

Use Python 3.13.5 with the pinned dependencies.

```sh
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python generate_sbg.py output
```

The generator writes `sbg.json` and `validation.json` to the supplied directory. It checks versions and numerical agreement before writing. JSON retains binary64 precision and rejects nonfinite numbers. Repeated runs with the recorded software produce identical bytes on the recorded platform.

## Distribution convention

Peter S. Fader and Bruce G. S. Hardie define the sBG distribution in [How to project customer retention, 2007](https://doi.org/10.1002/dir.20074), Journal of Interactive Marketing 21, pages 76-90. Their equations 5 and 6 identify the churn mass and survivor function. Equation 2 defines retention; B3 gives the censored loglikelihood.

SciPy's documented beta-negative-binomial mass at `n=1` equals that churn mass after shifting the failure count `K` by one. Therefore `T=K+1`, `S(t)=P(T>t)=P(K>=t)`, and `S(t)=betanbinom.sf(t-1,1,alpha,beta)`. Churn at period `t` uses `betanbinom.pmf(t-1,1,alpha,beta)`. Retention is the ratio of consecutive survivor probabilities. The mean duration is `betanbinom.mean(1,alpha,beta)+1`. A null mean denotes an unbounded expectation when `alpha<=1`.

[`provenance.json`](../../../tests/fixtures/lifetime/provenance.json) records the original paper, official API documentation, exact upstream revisions, source-file hashes, and package wheel hashes. That file is a record kept by hand; the generator names it and does not write it. SciPy and NumPy use the BSD 3-Clause license. The package calls their installed APIs and contains no vendored implementation.

## Data and fitting

The generator initializes PCG64 separately for each recorded seed, draws churn probabilities with NumPy's beta sampler, and draws durations with its geometric sampler. A customer survives period `t` exactly when the generated duration exceeds `t`. The generator discards individual draws after aggregation.

The survival arrays cover periods 0 through 20. Retention and churn arrays cover periods 1 through 20. Every initial customer contributes either one observed churn or one right-censored duration. The likelihood includes all observed churn masses and the period-20 survivor probability. It omits the multinomial count coefficient.

SciPy BFGS minimizes the negative loglikelihood per initial customer in log-parameter space. The objective calls `betanbinom.logpmf` and `betanbinom.logsf`. The analytic gradient uses SciPy digamma values. Sixteen prescribed starts run without penalties or parameter bounds. Every start must converge. Selection uses the smallest gradient norm among solutions within `1e-8` of the best loglikelihood. Complete optimizer records remain in `sbg.json`.

## Numerical checks and tolerances

The generator checks fixed values against direct beta-function evaluations at 60 decimal digits. For each fit, a separate high-precision calculation differentiates the censored beta-function likelihood and finds its stationary point. Positive curvature and agreement across starts support identification of both parameters. These checks measure numerical error, not sampling uncertainty. They do not replace SciPy's fitted estimates.

[`validation.json`](../../../tests/fixtures/lifetime/validation.json) records discrepancies for curves, means, likelihoods, gradients, and fitted parameters. The gradient check covers every prescribed starting point. A fitted unbounded mean has no finite mean-error comparison.

The recommended absolute tolerances in `sbg.json` are `1e-9` for fixed values, `5e-5` for fitted parameters, `1e-7` for fitted loglikelihoods, `2e-5` for fitted curve probabilities, and `0.002` for a finite fitted mean. The largest discrepancy across fitted starts is below `2.1e-7` for parameters and `1e-12` for likelihoods in the recorded run. The wider fit tolerances allow different optimizer stopping rules. They are numerical comparison thresholds, not confidence limits.

The inherited SciPy survival implementation subtracts a finite mass sum from one. The checks qualify the supplied parameters and periods through 20. They make no claim about numerical accuracy arbitrarily far into the tail.
