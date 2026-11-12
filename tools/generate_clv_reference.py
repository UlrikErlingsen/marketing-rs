#!/usr/bin/env python3
"""Synthetic constant-margin keys from an exact rational geometric series.

Gupta and Lehmann (2003), Customers as Assets, equation 3,
https://doi.org/10.1002/dir.10045 . For q = retention/(1+discount),
the margin multiple is sum(q**t, t=1..infinity) = q/(1-q).
All arithmetic until JSON conversion uses Python's Fraction, independently of
the Rust implementation. This creates new synthetic inputs, not borrowed data.
"""
from fractions import Fraction as F
import json
from pathlib import Path


def reference(margin, retention, discount):
    margin, retention, discount = map(F, (margin, retention, discount))
    q = retention / (1 + discount)
    assert 0 <= q < 1
    multiple = q / (1 - q)
    # Verify the closed form against a finite sum plus its exact remainder.
    partial = sum((q**t for t in range(1, 33)), F(0))
    assert multiple == partial + q**33 / (1 - q)
    return {"margin": float(margin), "retention": float(retention), "discount": float(discount),
            "expected_value": float(margin * multiple), "expected_multiple": float(multiple),
            "exact_value": str(margin * multiple), "exact_multiple": str(multiple)}


def main():
    rows = []
    for r in ["0", ".25", ".55", ".82", "1"]:
        for d in ["-.1", "0", ".07", ".13", ".25"]:
            if F(r) >= 1 + F(d):
                continue
            for margin in ["-12.5", "0", "17.25", "80"]:
                rows.append(reference(margin, r, d))
    rows.append(reference("64", ".81", ".08"))
    result = {"source": {"authors": ["Sunil Gupta", "Donald R. Lehmann"], "year": 2003,
                         "title": "Customers as Assets", "equation": 3,
                         "doi": "10.1002/dir.10045"},
              "data_origin": "Synthetic Cartesian grid; exact rational geometric-series closed form",
              "generator": "tools/generate_clv_reference.py", "cases": rows,
              "comparison": {"absolute": 1e-9, "relative": 1e-9}}
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/clv/constant_margin.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} new exact-rational synthetic cases")


if __name__ == "__main__":
    main()
