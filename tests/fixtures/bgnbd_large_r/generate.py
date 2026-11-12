"""Answer key for BG/NBD forecasts at large r. Prints expected.json to stdout.

Usage: python3 -B generate.py DIGITS

Route: the posterior integral. Given an active customer, the purchase rate is Gamma(r+x, alpha+T) and the dropout
probability is Beta(a, b+x). Fader, Hardie and Lee (2005), eq. 5, gives the expected count for fixed rate and
dropout probability as (1-exp(-lambda*p*h))/p. Integrating the rate out leaves one integral over p. No series,
no hypergeometric function and no native code is used here. validate.py checks the same answers with eq. 10.
"""
import json
import sys

import mpmath as mp

DIGITS = int(sys.argv[1])
mp.mp.dps = DIGITS


def exact(value):
    numerator, denominator = float(value).as_integer_ratio()
    return mp.mpf(numerator) / denominator


def case(identifier, r, alpha, a, b, x, tx, t, horizons, note):
    return {"id": identifier, "note": note, "parameters": {"r": r, "alpha": alpha, "a": a, "b": b},
            "history": {"x": x, "tx": tx, "t": t, "weight": 1.0}, "horizons": horizons}


GRID_HORIZONS = [1.0, 13.0, 52.0, 260.0, 1040.0]
cases = [case("reported", 200.0, 200.0, 0.8, 2.5, 0, 0.0, 52.0, [104.0, 260.0], "the reported wrong forecasts")]
for r in [20.0, 50.0, 100.0, 200.0, 500.0, 2000.0]:
    for a, b in [(0.8, 2.5), (1.5, 0.6), (3.0, 40.0), (0.3, 0.2)]:
        for x, tx in [(0, 0.0), (3, 30.0), (40, 45.0)]:
            cases.append(case(f"grid_r{r:g}_a{a:g}_b{b:g}_x{x}", r, r, a, b, x, tx, 52.0, GRID_HORIZONS, "grid"))
for x, tx in [(0, 0.0), (400, 50.0)]:
    cases.append(case(f"rate_ten_x{x}", 200.0, 20.0, 0.8, 2.5, x, tx, 52.0, [1.0, 13.0, 52.0, 260.0],
                      "ten purchases per week"))
for a, b in [(0.8, 2.5), (2.0, 3.0)]:
    for x, tx in [(0, 0.0), (5, 40.0)]:
        cases.append(case(f"homogeneous_a{a:g}_x{x}", 50000.0, 500000.0, a, b, x, tx, 52.0,
                          [13.0, 52.0, 260.0, 1040.0], "nearly homogeneous purchase rates"))
cases.append(case("large_x", 500.0, 50.0, 1.5, 0.6, 1000, 51.0, 52.0, [1.0, 52.0, 260.0], "large r and large x"))
for x, tx in [(0, 0.0), (2, 30.43)]:
    cases.append(case(f"ordinary_small_r_x{x}", 0.243, 4.414, 0.793, 2.426, x, tx, 38.86, GRID_HORIZONS,
                      "ordinary fitted magnitudes"))
cases.append(case("ordinary_r5", 5.0, 10.0, 2.0, 3.0, 4, 20.0, 30.0, GRID_HORIZONS, "ordinary fitted magnitudes"))
cases.append(case("long_series", 0.243, 4.414, 2.3, 2.426, 0, 0.0, 38.86, [167000.0, 168000.0],
                  "either side of the 100000-term limit of the transformed series"))
for a in [1.0 - 1.0 / 128.0, 1.0 + 1.0 / 128.0]:
    cases.append(case(f"near_one_a{a:g}", 200.0, 200.0, a, 2.5, 0, 0.0, 52.0, [13.0, 104.0, 260.0],
                      "inside the near-one band at large r"))
cases.append(case("guard_threshold", 200.0, 200.0, 0.8, 2.5, 0, 0.0, 52.0,
                  [float(h) for h in range(24, 31)], "both sides of the rounding guard"))

for row in cases:
    p, h = row["parameters"], row["history"]
    r, alpha, a, b = (exact(p[key]) for key in ["r", "alpha", "a", "b"])
    x, t, tx = h["x"], exact(h["t"]), exact(h["tx"])
    shape, rate = r + x, alpha + t
    alive = mp.mpf(1) if x == 0 else 1 / (1 + a / (b + x - 1) * (rate / (alpha + tx)) ** shape)
    norm = mp.beta(a, b + x)
    predictions = []
    for horizon in row.pop("horizons"):
        H = exact(horizon)

        # Below one half the integrand is p^(a-1) * lower(p). Above it, with v = 1-p, it is v^(b+x-1) * upper(v).
        def lower(u):
            if u == 0:
                return shape * H / rate / norm
            return -mp.expm1(-shape * mp.log1p(u * H / rate)) / u * (1 - u) ** (b + x - 1) / norm

        def upper(v):
            u = 1 - v
            return -mp.expm1(-shape * mp.log1p(u * H / rate)) * u ** (a - 2) / norm

        # integral over (0, 1/2) of t^(e-1) * g(t) dt. The substitution w = t^e removes the endpoint power, so the
        # quadrature loses nothing next to a singular endpoint. The integrand turns near p = rate/(shape*H) and near
        # the beta mode; decade breakpoints resolve both.
        def powered(g, e):
            points = [0] + [mp.mpf(10) ** (-k * e) for k in range(9, 0, -1)] + [mp.mpf("0.5") ** e]
            return mp.quad(lambda w: g(w ** (1 / e)), points) / e

        value = powered(lower, a) + powered(upper, b + x)
        predictions.append({"horizon": horizon, "probability_active": float(alive),
                            "expected_if_active": float(value), "expected_purchases": float(alive * value)})
    row["predictions"] = predictions

out = {
    "schema": "bgnbd-large-r-synthetic-posterior-integral-v1",
    "input_semantics": "Each decimal JSON numeric input denotes its nearest binary64, integrated at its exact "
                       "rational value.",
    "primary_source": "https://doi.org/10.1287/mksc.1040.0098",
    "derivation": "Conditional on being active, lambda is Gamma(r+x, alpha+T) and p is Beta(a, b+x). E future | "
                  "active = integral over p of (1-(1+p*h/(alpha+T))^-(r+x))/p times the beta posterior. Multiply "
                  "by the posterior active probability.",
    "method": "mpmath tanh-sinh posterior integral; no Rust or native evaluation",
    "tolerances": {"scalar": {"absolute": 1e-12, "relative": 1e-9}},
    "cases": cases,
}
print(json.dumps(out, indent=2, allow_nan=False))
