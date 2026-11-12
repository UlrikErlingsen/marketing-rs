"""Independent check of expected.json with eq. 10 of Fader, Hardie and Lee (2005). Reads only.

Usage: python3 -B validate.py

The generator integrates the posterior. This script evaluates the published closed form with mpmath's
hypergeometric function at 80 digits and requires the same binary64 answers.
"""
import hashlib
import json
from pathlib import Path

import mpmath as mp

mp.mp.dps = 80
root = Path(__file__).resolve().parent
data = json.loads((root / "expected.json").read_text())
checks = 0


def exact(value):
    numerator, denominator = float(value).as_integer_ratio()
    return mp.mpf(numerator) / denominator


def expect(actual, expected, label):
    global checks
    assert float(actual) == expected, (label, mp.nstr(actual, 25), expected)
    checks += 1


for row in data["cases"]:
    p, h = row["parameters"], row["history"]
    r, alpha, a, b = (exact(p[key]) for key in ["r", "alpha", "a", "b"])
    x, t, tx = h["x"], exact(h["t"]), exact(h["tx"])
    alive = mp.mpf(1) if x == 0 else 1 / (1 + a / (b + x - 1) * ((alpha + t) / (alpha + tx)) ** (r + x))
    c = a + b + x - 1
    for prediction in row["predictions"]:
        H = exact(prediction["horizon"])
        z = H / (alpha + t + H)
        series = mp.hyp2f1(r + x, b + x, c, z, maxterms=10**7)
        value = c / (a - 1) * (1 - ((alpha + t) / (alpha + t + H)) ** (r + x) * series)
        label = f"{row['id']} h={prediction['horizon']}"
        expect(alive, prediction["probability_active"], label)
        expect(value, prediction["expected_if_active"], label)
        expect(alive * value, prediction["expected_purchases"], label)

if (root / "manifest.json").exists():
    manifest = json.loads((root / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        raw = (root / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["sha256"] and len(raw) == record["bytes"], name
        checks += 1
print(json.dumps({"status": "passed", "checks": checks, "cases": len(data["cases"]), "precision_digits": 80,
                  "method": "Fader, Hardie and Lee (2005) eq. 10 with mpmath hyp2f1", "native_calls": 0}))
