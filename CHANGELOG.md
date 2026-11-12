# Changelog

## 0.2.2

No result changes. Every likelihood, fit, score and forecast equals 0.2.1 bit for bit, and no public item
is added, removed or renamed.

- The source, the tests and the answer keys are in a public repository,
  <https://github.com/UlrikErlingsen/marketing-rs>. `Cargo.toml` and the README link to it.
- RFM has a new answer key. A standalone generator applies the documented banding rules with exact
  integer arithmetic to new synthetic inputs, and the tests compare every row.
- Tidy the source comments, the README and the licence notes. Every file under `src/` and the example
  carry the MIT notice.
- The README says that the crate was written with AI assistance.

## 0.2.1

Fixes only. No public item is added, removed or renamed.

- Fix BG/NBD forecasts at large `r`. When `r` is much larger than `a + b + x - 1`, the
  forecast series cancelled and `predict` could return a wrong number without an
  error. For `r = 200`, `alpha = 200`, `a = 0.8`, `b = 2.5`, `x = 0`, `T = 52`, version
  0.2.0 returned 19.89 at horizon 104 and 2.0e14 at horizon 260. The model gives
  12.977 and 17.836. The series now carries a rounding bound. When the bound is
  within 1e-9, the forecast keeps every bit of 0.2.0. Otherwise the untransformed
  series of the cited paper answers, and returns an error unless its own rounding bound
  is within 1e-9 of the result. Some inputs that 0.2.0 refused for overflow now
  answer. Some extreme inputs that 0.2.0 answered now return an error, because
  neither series can bound their rounding within 100,000 terms.
- Fix BG/NBD forecasts with dropout shape `a` at or near one. Within
  `|a - 1| <= 1/64` the forecast uses a series with the factor `a - 1` removed.
  On 600 pseudo-random inputs at ordinary magnitudes, the largest relative error
  of 0.2.0 was 2e-4 at `|a - 1| = 1e-8` and larger than the forecast itself at
  `1e-12`. `a = 1` now answers with the continuous limit, where 0.2.0 returned an
  error.
- New errors inside that band. The new series returns an error when its rounding
  estimate exceeds 1e-11 of the expected purchases if active. That happens at
  large `r`, and at long horizons that bring `horizon / (alpha + T + horizon)`
  close to one, where 0.2.0 answered and was sometimes right. Example:
  `r = 0.1704`, `alpha = 5.92`, `a = 1.012`, `b = 7.727`, `x = 0`, `T = 2.383`
  answers at horizon 100 and returns an error from horizon 108. At horizon 200,
  0.2.0 returned 2.14676511, a relative error of 1.4e-11. In an independent
  sample of 34,400 inputs in the band with `r` up to 20, 309 changed from an
  answer to an error.
- BG/BB returns an error where 0.2.0 panicked: a `beta` below about 1.1e-16 times
  `n`, with a purchase at every opportunity up to the last purchase.
- Documentation: state the numerical limits of the BG/NBD forecast with measured
  figures; describe `Prediction::expected_purchases` as the product of the other
  two fields; note that RFM bands follow sorted position and so differ from
  value-quantile binning on tied data, and that a nested cell can hold one row;
  remove README text and a link about repository files that are not in the
  package; use straight quotes.
- Likelihoods, fits, RFM scores, constant-margin values, sBG and every BG/BB
  answer are unchanged. BG/NBD forecasts outside the two repaired regions keep
  every bit. The repaired forecasts are checked against independent high-precision answer keys.
- Add two BG/NBD answer keys from high-precision posterior integrals, each
  confirmed by an independent closed-form evaluation: 22 histories near `a = 1`
  and 87 histories at large `r`.

## 0.2.0

- Add a CLV answer key of 93 synthetic cases checked by an exact rational
  geometric-series closed form. Existing public CLV behavior is preserved.

- Add the pure BG/NBD, shifted beta-geometric and BG/BB lifetime-model cores,
  including likelihoods, bounded multistart fits and model-specific predictions.
  Fitted forecast methods refuse unsuccessful or bounded fit attempts. BG/NBD
  retains fit diagnostics. Parameter-uncertainty interfaces and discounted
  extensions are deferred.
- Refuse overflowing BG/NBD time combinations, overflowing parameter sums and
  nonfinite BG/NBD likelihood or sBG results. These range refusals are admission
  checks around the model equations; recorded finite answers stay unchanged.
- Return an error for fixed-p/q conditional activity when the history has zero
  probability, where that conditional probability is undefined.
- Refuse inconsistent zero-purchase BG/NBD and BG/BB histories, nonfinite fitted likelihoods
  and penalty-only optimizer convergence. Checked forecasts also inspect the
  fitted likelihood before returning values.
- Document primary academic sources, input requirements, assumptions, limits,
  examples and the provenance of synthetic reference inputs. Preserve existing
  RFM and constant-margin APIs.
- Add frozen lifetime-model answer keys. Expected numbers and comparison
  tolerances are declared with their independent generators.
- Include source and dependency notices. Keep repository-only fixtures,
  tests and tools out of the crate package.

## 0.1.0 (not published)

Independent and nested RFM scores, and constant-margin customer lifetime value with a margin multiple.
