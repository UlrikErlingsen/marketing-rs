# BG/NBD and BG/BB numerical references

`generate.py` creates `bgnbd.json` and `bgbb.json` from synthetic histories. Each file has 72 fixed cases and 1,000 individually weighted fitting histories. The BG/BB file also has 16 fixed purchase/dropout probability cases.

Use Python 3.13 and an isolated virtual environment with `requirements.txt` installed. Run:

```sh
python -m pip install -r requirements.txt
python generate.py output
```

The positional argument selects the output directory. The generator checks package versions and source hashes. JSON uses round-trip binary64 precision and rejects nonfinite numbers. Seeds, dependency versions, API identifiers and source hashes are in the metadata. Execution adds no dates or host identifiers. NumPy PCG64 seeds are 2026110901 for BG/NBD and 2026110902 for BG/BB.

## Model sources

- Peter S. Fader, Bruce G. S. Hardie and Ka Lok Lee. 2005. "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model. Marketing Science 24, 275-284. [DOI 10.1287/mksc.1040.0098](https://doi.org/10.1287/mksc.1040.0098). Equations 6, 7 and 10 define the integrated likelihood, fitting objective and conditional mean.
- Peter S. Fader, Bruce G. S. Hardie and Jen Shang. 2010. Customer-Base Analysis in a Discrete-Time Noncontractual Setting. Marketing Science 29, 1086-1108. [DOI 10.1287/mksc.1100.0580](https://doi.org/10.1287/mksc.1100.0580). Equations 4-6, 11, 13 and A10 define pattern likelihoods, next-opportunity activity and future purchases.

## Implementation provenance

The expected values call the unmodified open-source Lifetimes 0.11.3 package, released under MIT. Its release tag is `v0.11.3`, commit `bd3ae6e13bc82b46fca7ce93130b2cfce4e8d7cf`. The metadata identifies the exact APIs and source files. The implementation identifiers document the computation; the publications above identify the model authors.

Synthetic generation uses NumPy directly. BG/NBD uses continuous exponential event times and dropout after each repeat purchase. BG/BB uses a dropout draw before each discrete purchase opportunity. Histories include zero repeat purchases. Fitting uses three deterministic initial parameter vectors, SciPy BFGS and the package's unpenalized mean negative log likelihood. The generator exports the optimizer status, gradients and parameter profiles. It checks the gradient with a separate five-point finite difference.

## Conventions

BG/NBD starts active at time zero and excludes the initial purchase from `x`. Its likelihood is for an ordered event history. It includes neither a factorial nor a time-volume factor for a count-only probability. `probability_active` refers to the observation endpoint. `expected_purchases` averages over activity conditional on history. `expected_if_active` divides that mean by `probability_active`.

BG/BB uses one binary purchase pattern. It includes no combinatorial multiplicity. A summary row can represent several customers, but its weight multiplies the pattern log likelihood rather than adding a combinatorial term. `probability_alive` always refers to opportunity `n+1`, including the extra survival decision, regardless of the purchase horizon. The mean covers opportunities `n+1` through `n+horizon`.

Lifetimes initializes its BG/BB dropout sum with `1e-15` before beta normalization. The stored library outputs retain this numerical floor. A separate finite-state forward algorithm, integrated over beta distributions with Gauss-Jacobi quadrature, reports the difference for every fixed case. The maximum absolute difference among fixed fields is below `1.5e-10` for these inputs. The fitting diagnostics also measure the likelihood difference at the fitted parameters. Longer windows and more extreme parameters require separate numerical checks.

The fixed `p`/`q` cases use the finite-state forward algorithm directly. `p` is purchase probability and `q` is dropout probability. Each case stores one binary pattern. For a zero-likelihood pattern, `status` is `zero_likelihood`, the likelihood is zero, and the log likelihood and undefined conditional quantities are JSON null. Null represents an extended-real negative-infinite log likelihood only when this status is present.

## Comparison tolerances

For fixed values, use absolute tolerance `1e-9` and relative tolerance `1e-9`. For fitted parameters, use absolute tolerance `1e-5` and relative tolerance `1e-4`. Also require total log likelihood within absolute `1e-6` and a maximum absolute log-parameter gradient of the mean negative log likelihood below `1e-6`.

Across the three successful starts, the maximum relative parameter spread is below `5.7e-7` for BG/NBD and `6.3e-8` for BG/BB. Total log likelihood varies by less than `3e-12`. The fitting tolerances allow differences in optimizer stopping while requiring agreement on the objective and stationarity. These tolerances come from the reference computations.

The generated histories and numeric results contain no third-party datasets. The generator carries an MIT identifier. The other files here use the repository's MIT OR Apache-2.0 licences.
