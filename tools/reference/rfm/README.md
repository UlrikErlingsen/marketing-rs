# RFM answer key

`generate_rfm.py` writes the RFM answer key in `tests/fixtures/rfm/`. It uses the Python standard library only.

It implements the banding rules from the crate documentation: the section "RFM and constant-margin conventions" in `README.md` and the module documentation of `marketing::rfm`. Values are ordered as exact rationals, with infinities placed at the ends and both signed zeros equal to zero. Band positions use exact integer arithmetic. No floating-point comparison decides a band.

## Inputs

- `population.csv` holds 300 synthetic customers from `random.Random(20261112)`, with whole-day recency, purchase counts from 1 to 24, spend built from six price points (with a few refunds), many ties and a few NaN values in every column. Customer ids are `c0001` to `c0300`.
- 16 hand-built inputs, written in the script: a single row, fewer rows than groups, fewer distinct values than groups, constant columns, ties broken by row order, NaN in each column and in several, rows missing recency or frequency, monetary values only in rows without recency, positive and negative infinity, positive and negative zero, extreme magnitudes and subnormals, and 520 distinct values.
- Every input runs with groups 1, 2, 3, 5, 7 and 255, in independent and in nested mode: 204 cases and 11,076 expected rows.
- `errors.json` holds 12 inputs that the documentation says must return an error.

## Files

- `cases.json`: each case has its columns (or `"input": "population.csv"`), `groups`, `nested` and one expected row per input row, `[r, f, m, segment, total]`, with `null` for a missing value.
- `errors.json`: the error inputs and the rule each one breaks.
- `manifest.json`: the generator path and SHA-256, the Python version, the seed, the counts and the SHA-256 of every other file.

Input values are written as shortest round-trip decimal text, with `NaN`, `inf` and `-inf` spelled out, so each one parses to the exact binary64 value.

## Readings where the documentation leaves a point open

- In nested mode, a row without a recency band has no frequency band, because frequency is ranked within recency bands.
- `segment()` is `None` when any band is missing.
- "A dimension with no usable value" means a column in which every value is NaN. A nested monetary column whose only values sit in rows without recency is not an error; those rows get no monetary band.

## Regenerate

```sh
python3 tools/reference/rfm/generate_rfm.py
```

The script is deterministic. Running it again must reproduce every file byte for byte; check with `git status` after a run. A different Python version changes only the `python` field of `manifest.json`. `tests/rfm_golden.rs` checks the hashes in the manifest, then requires exact equality of every band, `segment()` and `total()`.
