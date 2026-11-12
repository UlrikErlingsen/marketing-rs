#!/usr/bin/env python3
"""Generate the independent RFM answer key.

This script implements the RFM banding rules exactly as the crate documents
them (the "RFM and constant-margin conventions" section of README.md and the
module documentation of `marketing::rfm`). It uses only the Python standard
library. Values are ordered as exact rationals, and band positions use exact
integer arithmetic, so no floating-point comparison decides a band.

The documented rules:

* Recency sorts ascending; frequency and monetary value sort descending.
  Ties keep input row order. Signed zeros tie. Infinity is ranked. NaN is
  missing and takes no part in ranking.
* For n usable rows and D distinct values, use B = min(groups, n, D) bands.
  Zero-based position p gets the first band b with p * B <= (n - 1) * b,
  or band B.
* Independent mode ranks each column over all rows. Nested mode ranks
  frequency within each recency band and monetary value within each
  recency/frequency cell. A row missing recency or frequency has no nested
  monetary score.
* segment() is "r-f-m" and total() is r + f + m when all three bands exist;
  total() is None when the sum does not fit in a u8.
* Errors: empty dimensions, unequal lengths, zero groups and a dimension
  with no non-NaN value.

Readings chosen where the text leaves a point open (see README.md in this
folder): in nested mode a row without a recency band has no frequency band;
segment() is None when any band is missing; "a dimension with no usable
value" means a column whose every entry is NaN.

Usage: python3 tools/reference/rfm/generate_rfm.py [--out DIR]
The default output folder is tests/fixtures/rfm under the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

SEED = 20261112
POPULATION_SIZE = 300
GROUP_COUNTS = (1, 2, 3, 5, 7, 255)
MODES = (False, True)
NAN = float("nan")
INF = float("inf")

GENERATOR = "tools/reference/rfm/generate_rfm.py"


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------


def order_key(value: float) -> tuple[int, Fraction]:
    """Exact total order on non-NaN floats; +0 and -0 map to the same key."""
    if value == INF:
        return (1, Fraction(0))
    if value == -INF:
        return (-1, Fraction(0))
    return (0, Fraction(value))


def band_rows(values: list[float], rows: list[int], groups: int, descending: bool) -> dict[int, int]:
    """Band the given rows of one column. Rows whose value is NaN are skipped."""
    usable = [row for row in rows if not math.isnan(values[row])]
    n = len(usable)
    if n == 0:
        return {}

    def key(row: int):
        cls, frac = order_key(values[row])
        if descending:
            cls, frac = -cls, -frac
        return (cls, frac, row)

    ordered = sorted(usable, key=key)
    distinct = len({order_key(values[row]) for row in usable})
    return {row: position_band(p, n, distinct, groups) for p, row in enumerate(ordered)}


def position_band(p: int, n: int, distinct: int, groups: int) -> int:
    """First band b with p * B <= (n - 1) * b, else B, where B = min(groups, n, D)."""
    bands = min(groups, n, distinct)
    for b in range(1, bands + 1):
        if p * bands <= (n - 1) * b:
            return b
    return bands


class RfmError(Exception):
    pass


def score(recency, frequency, monetary, groups, nested):
    """Return a list of (r, f, m) with None for a missing band, or raise RfmError."""
    if len(recency) != len(frequency) or len(recency) != len(monetary):
        raise RfmError("unequal lengths")
    if len(recency) == 0:
        raise RfmError("empty dimensions")
    if groups == 0:
        raise RfmError("zero groups")
    for column in (recency, frequency, monetary):
        if all(math.isnan(v) for v in column):
            raise RfmError("dimension with no usable value")

    all_rows = list(range(len(recency)))
    r_band = band_rows(recency, all_rows, groups, descending=False)

    if not nested:
        f_band = band_rows(frequency, all_rows, groups, descending=True)
        m_band = band_rows(monetary, all_rows, groups, descending=True)
    else:
        f_band = {}
        for b in sorted(set(r_band.values())):
            cell = [row for row in all_rows if r_band.get(row) == b]
            f_band.update(band_rows(frequency, cell, groups, descending=True))
        m_band = {}
        cells = sorted({(r_band[row], f_band[row]) for row in all_rows if row in r_band and row in f_band})
        for cell_key in cells:
            cell = [
                row
                for row in all_rows
                if row in r_band and row in f_band and (r_band[row], f_band[row]) == cell_key
            ]
            m_band.update(band_rows(monetary, cell, groups, descending=True))

    return [(r_band.get(row), f_band.get(row), m_band.get(row)) for row in all_rows]


def segment(r, f, m):
    if r is None or f is None or m is None:
        return None
    return f"{r}-{f}-{m}"


def total(r, f, m):
    if r is None or f is None or m is None:
        return None
    s = r + f + m
    return s if s <= 255 else None


# ---------------------------------------------------------------------------
# Value encoding
# ---------------------------------------------------------------------------


def encode(value: float) -> str:
    """Shortest round-trip decimal text; NaN, inf and -inf spelled out."""
    if math.isnan(value):
        return "NaN"
    if value == INF:
        return "inf"
    if value == -INF:
        return "-inf"
    text = repr(value)
    assert float(text) == value and math.copysign(1.0, float(text)) == math.copysign(1.0, value)
    return text


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def population(rng: random.Random):
    """Synthetic customers with plausible values and many ties."""
    ids, rec, freq, mon = [], [], [], []
    for i in range(POPULATION_SIZE):
        ids.append(f"c{i + 1:04d}")
        u = rng.random()
        # Days since last purchase: mostly recent, a long tail, whole days.
        days = int(365 * u * u)
        rec.append(float(days))
        # Purchase count: geometric-like, 1 to 24.
        count = 1
        while count < 24 and rng.random() < 0.55:
            count += 1
        freq.append(float(count))
        # Spend: price points times count, sometimes with cents, some refunds.
        price = (9.99, 19.0, 25.5, 49.0, 99.95, 5.0)[int(rng.random() * 6)]
        spend = round(price * count, 2)
        if rng.random() < 0.04:
            spend = -round(price, 2)
        mon.append(spend)
    # A few missing entries in each column.
    for column in (rec, freq, mon):
        for _ in range(6):
            column[int(rng.random() * POPULATION_SIZE)] = NAN
    return ids, rec, freq, mon


BIG = 1.7976931348623157e308
TINY = 5e-324
MIN_NORMAL = 2.2250738585072014e-308


def edge_cases():
    """Hand-built cases: (id, description, recency, frequency, monetary)."""
    z = -0.0
    cases = [
        ("single_row", "One customer.", [12.0], [3.0], [40.0]),
        (
            "fewer_rows_than_groups",
            "Three customers, all values distinct.",
            [30.0, 2.0, 9.0],
            [1.0, 6.0, 3.0],
            [15.0, 250.0, 70.0],
        ),
        (
            "fewer_distinct_than_groups",
            "Eight customers with two distinct values per column.",
            [10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 10.0, 40.0],
            [2.0, 2.0, 5.0, 5.0, 2.0, 5.0, 2.0, 5.0],
            [100.0, 100.0, 100.0, 20.0, 20.0, 20.0, 100.0, 20.0],
        ),
        (
            "constant_recency",
            "Recency is the same for every customer.",
            [7.0] * 6,
            [1.0, 4.0, 2.0, 8.0, 3.0, 5.0],
            [11.0, 90.0, 35.0, 60.0, 5.0, 12.5],
        ),
        (
            "all_constant",
            "Every column is constant.",
            [3.0] * 5,
            [2.0] * 5,
            [50.0] * 5,
        ),
        (
            "ties_by_row_order",
            "Long runs of tied values, broken by row order.",
            [3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0],
            [4.0, 4.0, 1.0, 4.0, 1.0, 1.0, 4.0, 4.0, 1.0, 1.0, 4.0, 4.0],
            [9.0, 9.0, 9.0, 9.0, 2.0, 2.0, 2.0, 9.0, 9.0, 2.0, 2.0, 9.0],
        ),
        (
            "nan_recency",
            "Some recency values missing.",
            [5.0, NAN, 1.0, 9.0, NAN, 3.0, 7.0],
            [2.0, 3.0, 1.0, 6.0, 2.0, 4.0, 2.0],
            [20.0, 30.0, 10.0, 60.0, 25.0, 40.0, 22.0],
        ),
        (
            "nan_frequency",
            "Some frequency values missing.",
            [5.0, 2.0, 1.0, 9.0, 4.0, 3.0, 7.0],
            [2.0, NAN, 1.0, 6.0, NAN, 4.0, 2.0],
            [20.0, 30.0, 10.0, 60.0, 25.0, 40.0, 22.0],
        ),
        (
            "nan_monetary",
            "Some monetary values missing.",
            [5.0, 2.0, 1.0, 9.0, 4.0, 3.0, 7.0],
            [2.0, 3.0, 1.0, 6.0, 2.0, 4.0, 2.0],
            [20.0, NAN, 10.0, NAN, 25.0, 40.0, NAN],
        ),
        (
            "nan_several",
            "Rows missing two or three values, and missing values in every column.",
            [NAN, 2.0, NAN, 9.0, 4.0, 3.0, 7.0, 6.0, 1.0],
            [NAN, NAN, 1.0, 6.0, 2.0, NAN, 2.0, 3.0, 5.0],
            [NAN, 30.0, NAN, 60.0, NAN, 40.0, 22.0, 18.0, 75.0],
        ),
        (
            "missing_recency_or_frequency_with_monetary",
            "Rows that have monetary value but lack recency or frequency.",
            [NAN, 4.0, 4.0, NAN, 8.0, 8.0, 2.0, 2.0],
            [3.0, NAN, 2.0, NAN, 5.0, NAN, 1.0, 1.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        ),
        (
            "monetary_only_without_recency",
            "The only non-missing monetary values sit in rows without recency.",
            [NAN, 1.0, 2.0, NAN, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [15.0, NAN, NAN, 25.0, NAN],
        ),
        (
            "infinities",
            "Positive and negative infinity in every column, some repeated.",
            [INF, 3.0, -INF, 0.0, INF, 12.0, -INF, 1.5],
            [-INF, 2.0, INF, 2.0, 7.0, INF, 1.0, -INF],
            [100.0, INF, -INF, 0.0, INF, -5.0, 40.0, -INF],
        ),
        (
            "signed_zeros",
            "Positive and negative zero tie and keep row order.",
            [0.0, z, 1.0, z, 0.0, -1.0, z],
            [z, 0.0, 0.0, 2.0, z, -2.0, 1.0],
            [z, 0.0, z, 0.0, 5.0, -5.0, z],
        ),
        (
            "extreme_magnitudes",
            "Largest and smallest finite magnitudes, subnormals and one-ulp gaps.",
            [BIG, -BIG, TINY, -TINY, MIN_NORMAL, 1e-300, 1e300, 1.0, 1.0000000000000002, 0.0],
            [TINY, 0.0, -TINY, BIG, 1e300, -BIG, 1.0000000000000002, 1.0, MIN_NORMAL, 1e-300],
            [1.0, 1.0000000000000002, 0.9999999999999999, BIG, BIG, -BIG, TINY, 2 * TINY, -TINY, 0.0],
        ),
    ]
    # Many distinct values so that 255 bands are all in use.
    n = 520
    cases.append(
        (
            "distinct_ramp",
            "520 customers with distinct values; reversed and shuffled orders.",
            [float(i) for i in range(n)],
            [float(n - i) for i in range(n)],
            [float((i * 37) % n) for i in range(n)],
        )
    )
    return cases


def error_cases():
    """Inputs that the documentation says must return an error."""
    return [
        ("empty_dimensions", "All three columns are empty.", [], [], [], 5, False),
        ("empty_dimensions_nested", "All three columns are empty.", [], [], [], 5, True),
        ("unequal_lengths_frequency_short", "Frequency is one row shorter.", [1.0, 2.0, 3.0], [1.0, 2.0], [1.0, 2.0, 3.0], 5, False),
        ("unequal_lengths_monetary_long", "Monetary is one row longer.", [1.0, 2.0], [1.0, 2.0], [1.0, 2.0, 3.0], 5, True),
        ("unequal_lengths_recency_empty", "Recency is empty, the others are not.", [], [1.0], [1.0], 5, False),
        ("zero_groups", "Zero bands requested.", [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], 0, False),
        ("zero_groups_nested", "Zero bands requested.", [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], 0, True),
        ("no_usable_recency", "Every recency value is NaN.", [NAN, NAN], [1.0, 2.0], [1.0, 2.0], 5, False),
        ("no_usable_frequency", "Every frequency value is NaN.", [1.0, 2.0], [NAN, NAN], [1.0, 2.0], 5, True),
        ("no_usable_monetary", "Every monetary value is NaN.", [1.0, 2.0], [1.0, 2.0], [NAN, NAN], 3, False),
        ("no_usable_monetary_nested", "Every monetary value is NaN.", [1.0, 2.0], [1.0, 2.0], [NAN, NAN], 3, True),
        ("single_row_all_nan", "One row, every value NaN.", [NAN], [NAN], [NAN], 1, True),
    ]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def expected_rows(rec, freq, mon, groups, nested):
    rows = []
    for r, f, m in score(rec, freq, mon, groups, nested):
        rows.append([r, f, m, segment(r, f, m), total(r, f, m)])
    return rows


def dumps_case(case: dict) -> str:
    lines = ["  {"]
    items = list(case.items())
    for i, (key, value) in enumerate(items):
        comma = "," if i + 1 < len(items) else ""
        if key == "expected":
            lines.append('    "expected": [')
            for j, row in enumerate(value):
                rc = "," if j + 1 < len(value) else ""
                lines.append("      " + json.dumps(row, separators=(", ", ": ")) + rc)
            lines.append("    ]" + comma)
        else:
            lines.append(f"    {json.dumps(key)}: {json.dumps(value, separators=(', ', ': '))}{comma}")
    lines.append("  }")
    return "\n".join(lines)


def dumps_list(cases: list[dict]) -> str:
    return "[\n" + ",\n".join(dumps_case(c) for c in cases) + "\n]\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--out", type=Path, default=root / "tests" / "fixtures" / "rfm")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    ids, rec, freq, mon = population(rng)

    csv_lines = ["customer_id,recency,frequency,monetary"]
    for row in zip(ids, rec, freq, mon):
        csv_lines.append(",".join([row[0]] + [encode(v) for v in row[1:]]))
    population_text = "\n".join(csv_lines) + "\n"

    cases = []
    for groups in GROUP_COUNTS:
        for nested in MODES:
            cases.append(
                {
                    "id": f"population_g{groups}_{'nested' if nested else 'independent'}",
                    "description": "Seeded synthetic population from population.csv.",
                    "input": "population.csv",
                    "groups": groups,
                    "nested": nested,
                    "expected": expected_rows(rec, freq, mon, groups, nested),
                }
            )
    for case_id, description, r, f, m in edge_cases():
        for groups in GROUP_COUNTS:
            for nested in MODES:
                cases.append(
                    {
                        "id": f"{case_id}_g{groups}_{'nested' if nested else 'independent'}",
                        "description": description,
                        "recency": [encode(v) for v in r],
                        "frequency": [encode(v) for v in f],
                        "monetary": [encode(v) for v in m],
                        "groups": groups,
                        "nested": nested,
                        "expected": expected_rows(r, f, m, groups, nested),
                    }
                )

    errors = []
    for case_id, description, r, f, m, groups, nested in error_cases():
        try:
            score(r, f, m, groups, nested)
        except RfmError as exc:
            reason = str(exc)
        else:
            raise AssertionError(f"{case_id} did not fail")
        errors.append(
            {
                "id": case_id,
                "description": description,
                "recency": [encode(v) for v in r],
                "frequency": [encode(v) for v in f],
                "monetary": [encode(v) for v in m],
                "groups": groups,
                "nested": nested,
                "reason": reason,
            }
        )

    files = {
        "population.csv": population_text,
        "cases.json": dumps_list(cases),
        "errors.json": dumps_list(errors),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")

    generator_bytes = Path(__file__).read_bytes()
    manifest = {
        "generator": GENERATOR,
        "generator_sha256": hashlib.sha256(generator_bytes).hexdigest(),
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
        "seed": SEED,
        "group_counts": list(GROUP_COUNTS),
        "population_rows": POPULATION_SIZE,
        "edge_inputs": len(edge_cases()),
        "cases": len(cases),
        "case_rows": sum(len(c["expected"]) for c in cases),
        "error_cases": len(errors),
        "sha256": {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in sorted(files.items())},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(cases)} cases, {manifest['case_rows']} rows, {len(errors)} error cases to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
