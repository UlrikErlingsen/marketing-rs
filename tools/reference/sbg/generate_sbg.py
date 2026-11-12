#!/usr/bin/env python3
"""Generate deterministic synthetic shifted-beta-geometric references."""

import argparse
import json
import math
import platform
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import digamma
from scipy.stats import betanbinom


VERSIONS = {"numpy": "2.3.5", "scipy": "1.16.3", "mpmath": "1.3.0"}
HORIZON = 20
STARTS = [(a, b) for a in (0.3, 1.0, 3.0, 8.0) for b in (0.5, 2.0, 6.0, 12.0)]
FIXED = [
    ("finite_a", 1.5, 4.5),
    ("unbounded", 0.75, 2.0),
    ("boundary", 1.0, 3.6),
    ("finite_b", 3.2, 1.8),
]
DATASETS = [
    ("synthetic_finite", 2026110901, 2000, 1.5, 4.5),
    ("synthetic_unbounded", 2026110902, 2000, 0.75, 2.0),
]
TOLERANCES = {
    "fixed_absolute": 1e-9,
    "fit_parameters_absolute": 5e-5,
    "fit_loglikelihood_absolute": 1e-7,
    "fit_curves_absolute": 2e-5,
    "fit_mean_duration_absolute": 0.002,
}


def curves(alpha, beta):
    periods = np.arange(HORIZON + 1)
    survival = betanbinom.sf(periods - 1, 1, alpha, beta)
    retention = survival[1:] / survival[:-1]
    churn = betanbinom.pmf(periods[1:] - 1, 1, alpha, beta)
    raw_mean = float(betanbinom.mean(1, alpha, beta)) + 1.0
    if alpha <= 1.0:
        if raw_mean != math.inf:
            raise ValueError("Unexpected mean classification")
        mean = None
    else:
        if not math.isfinite(raw_mean):
            raise ValueError("Nonfinite mean for alpha greater than one")
        mean = raw_mean
    if not np.all(np.isfinite(survival)) or not np.all(np.isfinite(retention)):
        raise ValueError("Nonfinite curve")
    if survival[0] != 1.0 or not np.all(np.diff(survival) <= 0.0):
        raise ValueError("Invalid survival ordering")
    if not np.all((retention > 0.0) & (retention < 1.0)):
        raise ValueError("Invalid retention")
    np.testing.assert_allclose(survival[:-1] - survival[1:], churn, atol=1e-12, rtol=0)
    return {
        "survival": survival.tolist(),
        "retention": retention.tolist(),
        "churn_probability": churn.tolist(),
        "mean_duration": mean,
        "mean_status": "unbounded" if mean is None else "finite",
    }


def loglikelihood(survivors, alpha, beta):
    survivors = np.asarray(survivors, dtype=np.int64)
    losses = survivors[:-1] - survivors[1:]
    terms = betanbinom.logpmf(np.arange(HORIZON), 1, alpha, beta)
    tail = float(betanbinom.logsf(HORIZON - 1, 1, alpha, beta))
    return float(np.dot(losses, terms) + survivors[-1] * tail)


def mean_negative_loglikelihood_gradient(survivors, log_parameters):
    alpha, beta = np.exp(log_parameters)
    survivors = np.asarray(survivors, dtype=np.int64)
    losses = survivors[:-1] - survivors[1:]
    common = digamma(alpha + beta) - digamma(alpha + beta + 1 + np.arange(HORIZON))
    da = digamma(alpha + 1) - digamma(alpha) + common
    db = digamma(beta + np.arange(HORIZON)) - digamma(beta) + common
    tail_a = digamma(alpha + beta) - digamma(alpha + beta + HORIZON)
    tail_b = digamma(beta + HORIZON) - digamma(beta) + tail_a
    return -np.array([
        alpha * (np.dot(losses, da) + survivors[-1] * tail_a),
        beta * (np.dot(losses, db) + survivors[-1] * tail_b),
    ]) / survivors[0]


def fit(survivors):
    def objective(log_parameters):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            parameters = np.exp(log_parameters)
            if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0):
                return math.inf
            value = -loglikelihood(survivors, *parameters) / survivors[0]
        return value if math.isfinite(value) else math.inf

    records = []
    results = []
    for start in STARTS:
        result = minimize(
            objective,
            np.log(start),
            method="BFGS",
            jac=lambda x: mean_negative_loglikelihood_gradient(survivors, x),
            options={"gtol": 1e-8, "maxiter": 1000},
        )
        parameters = np.exp(result.x)
        if not math.isfinite(float(result.fun)):
            raise ValueError("Nonfinite optimizer result")
        records.append({
            "start_alpha": start[0],
            "start_beta": start[1],
            "alpha": float(parameters[0]),
            "beta": float(parameters[1]),
            "loglikelihood": loglikelihood(survivors, *parameters),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "gradient_evaluations": int(result.njev),
            "mean_negative_loglikelihood_gradient_max_absolute": float(np.max(np.abs(result.jac))),
        })
        results.append(result)
    if not all(record["success"] for record in records):
        raise ValueError("At least one optimizer start did not converge")
    best_ll = max(record["loglikelihood"] for record in records)
    candidates = [i for i, record in enumerate(records) if record["loglikelihood"] >= best_ll - 1e-8]
    best_index = min(candidates, key=lambda i: records[i]["mean_negative_loglikelihood_gradient_max_absolute"])
    best = records[best_index]
    return {
        "alpha": best["alpha"],
        "beta": best["beta"],
        "loglikelihood": best["loglikelihood"],
        "convergence": {
            "success": best["success"],
            "method": "scipy.optimize.minimize/BFGS",
            "parameterization": "natural logarithms of alpha and beta",
            "penalty": "none",
            "bounds": "none",
            "selected_start_index": best_index,
            "selection": "smallest gradient norm among starts within 1e-8 of the best loglikelihood",
            "starts": records,
        },
        "curves": curves(best["alpha"], best["beta"]),
    }


def mp_curves(alpha, beta):
    a, b = mp.mpf(alpha), mp.mpf(beta)
    base = mp.beta(a, b)
    survival = [mp.beta(a, b + t) / base for t in range(HORIZON + 1)]
    retention = [survival[t] / survival[t - 1] for t in range(1, HORIZON + 1)]
    churn = [mp.beta(a + 1, b + t - 1) / base for t in range(1, HORIZON + 1)]
    return {"survival": survival, "retention": retention, "churn_probability": churn}


def curve_errors(actual, alpha, beta):
    reference = mp_curves(alpha, beta)
    return {
        key: float(max(abs(mp.mpf(x) - y) for x, y in zip(actual[key], values, strict=True)))
        for key, values in reference.items()
    }


def audit_fit(survivors, expected):
    losses = [int(a - b) for a, b in zip(survivors[:-1], survivors[1:], strict=True)]

    def ll(x, y):
        a, b = mp.exp(x), mp.exp(y)
        base = mp.log(mp.beta(a, b))
        observed = mp.fsum(
            count * (mp.log(mp.beta(a + 1, b + k)) - base)
            for k, count in enumerate(losses)
        )
        return observed + int(survivors[-1]) * (mp.log(mp.beta(a, b + HORIZON)) - base)

    def score(x, y):
        return mp.diff(lambda z: ll(z, y), x), mp.diff(lambda z: ll(x, z), y)

    gradient_errors = []
    for start in STARTS:
        log_start = np.log(start)
        exact = [-v / survivors[0] for v in score(*(mp.mpf(v) for v in log_start))]
        actual = mean_negative_loglikelihood_gradient(survivors, log_start)
        gradient_errors.extend(float(abs(mp.mpf(u) - v)) for u, v in zip(actual, exact, strict=True))
    if max(gradient_errors) > 1e-11:
        raise ValueError("Gradient disagrees with high precision differentiation")
    fixed_likelihood_errors = {
        label: float(abs(mp.mpf(loglikelihood(survivors, a, b)) - ll(mp.log(mp.mpf(a)), mp.log(mp.mpf(b)))))
        for label, a, b in FIXED
    }
    if max(fixed_likelihood_errors.values()) > TOLERANCES["fixed_absolute"]:
        raise ValueError("Fixed likelihood disagrees with high precision check")

    x, y = mp.findroot(
        score,
        (mp.log(expected["alpha"]), mp.log(expected["beta"])),
        tol=mp.mpf("1e-45"),
        maxsteps=30,
    )
    a, b = mp.exp(x), mp.exp(y)
    value = ll(x, y)
    parameter_errors = {"alpha": float(abs(expected["alpha"] - a)), "beta": float(abs(expected["beta"] - b))}
    ll_error = float(abs(mp.mpf(expected["loglikelihood"]) - value))
    errors = curve_errors(expected["curves"], a, b)
    hxx = -mp.diff(lambda z: ll(z, y), x, 2)
    hyy = -mp.diff(lambda z: ll(x, z), y, 2)
    hxy = -mp.diff(lambda z: mp.diff(lambda w: ll(z, w), y), x)
    curvature = mp.eigsy(mp.matrix([[hxx, hxy], [hxy, hyy]]))[0]
    if not all(v > 0 for v in curvature):
        raise ValueError("Stationary point has nonpositive curvature")
    if max(parameter_errors.values()) > TOLERANCES["fit_parameters_absolute"] / 5:
        raise ValueError("Optimizer parameters disagree with high precision check")
    if ll_error > TOLERANCES["fit_loglikelihood_absolute"] / 5:
        raise ValueError("Optimizer likelihood disagrees with high precision check")
    if max(errors.values()) > TOLERANCES["fit_curves_absolute"] / 5:
        raise ValueError("Fitted curves disagree with high precision check")
    if a > 1:
        mean_error = float(abs(mp.mpf(expected["curves"]["mean_duration"]) - (1 + b / (a - 1))))
        if mean_error > TOLERANCES["fit_mean_duration_absolute"] / 5:
            raise ValueError("Fitted mean disagrees with high precision check")
    else:
        mean_error = None
        if expected["curves"]["mean_duration"] is not None:
            raise ValueError("Expected unbounded mean")
    starts = expected["convergence"]["starts"]
    max_start_error = max(float(max(abs(record["alpha"] - a), abs(record["beta"] - b))) for record in starts)
    max_start_ll_error = max(float(abs(record["loglikelihood"] - value)) for record in starts)
    if max_start_error > TOLERANCES["fit_parameters_absolute"]:
        raise ValueError("Optimizer starts disagree on parameters")
    if max_start_ll_error > TOLERANCES["fit_loglikelihood_absolute"]:
        raise ValueError("Optimizer starts disagree on likelihood")
    return {
        "precision_decimal_digits": mp.mp.dps,
        "stationary_alpha": float(a),
        "stationary_beta": float(b),
        "stationary_loglikelihood": float(value),
        "parameter_absolute_errors": parameter_errors,
        "loglikelihood_absolute_error": ll_error,
        "curve_absolute_errors": errors,
        "mean_duration_absolute_error": mean_error,
        "negative_loglikelihood_curvature_eigenvalues": [float(v) for v in curvature],
        "score_max_absolute": float(max(abs(v) for v in score(x, y))),
        "gradient_max_absolute_error_at_starts": max(gradient_errors),
        "fixed_loglikelihood_absolute_errors": fixed_likelihood_errors,
        "all_starts_max_parameter_absolute_error": max_start_error,
        "all_starts_max_loglikelihood_absolute_error": max_start_ll_error,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    actual_versions = {"numpy": np.__version__, "scipy": scipy.__version__, "mpmath": mp.__version__}
    if actual_versions != VERSIONS or platform.python_version() != "3.13.5":
        raise RuntimeError("Use Python 3.13.5 and the pinned package versions")
    mp.mp.dps = 60
    fixed_cases, fit_cases, fixed_audits, fit_audits = [], [], [], []
    for name, a, b in FIXED:
        result = curves(a, b)
        errors = curve_errors(result, a, b)
        if max(errors.values()) > 1e-12:
            raise ValueError("Fixed curve disagrees with high precision check")
        if a > 1:
            mean_error = float(abs(mp.mpf(result["mean_duration"]) - (1 + mp.mpf(b) / (mp.mpf(a) - 1))))
            if mean_error > 1e-12:
                raise ValueError("Fixed mean disagrees with high precision check")
        else:
            mean_error = None
        fixed_cases.append({"name": name, "alpha": a, "beta": b, "expected": result})
        fixed_audits.append({"name": name, "curve_absolute_errors": errors, "mean_duration_absolute_error": mean_error})
    for name, seed, customers, a, b in DATASETS:
        rng = np.random.Generator(np.random.PCG64(seed))
        probabilities = rng.beta(a, b, size=customers)
        if not np.all((probabilities > 0) & (probabilities < 1)):
            raise ValueError("Generated invalid churn probabilities")
        durations = rng.geometric(probabilities)
        if not np.all(durations >= 1) or np.any(durations == np.iinfo(np.int64).max):
            raise ValueError("Generated invalid durations")
        survivors = [int(np.count_nonzero(durations > period)) for period in range(HORIZON + 1)]
        expected = fit(survivors)
        audit = audit_fit(survivors, expected)
        fit_cases.append({
            "name": name,
            "generation": {"seed": seed, "customers": customers, "alpha": a, "beta": b},
            "survivors": survivors,
            "losses": [x - y for x, y in zip(survivors[:-1], survivors[1:], strict=True)],
            "censored": survivors[-1],
            "loglikelihood_at_generating_parameters": loglikelihood(survivors, a, b),
            "loglikelihood_at_fixed_parameters": [
                {"fixed_case": label, "loglikelihood": loglikelihood(survivors, fa, fb)}
                for label, fa, fb in FIXED
            ],
            "expected": expected,
        })
        fit_audits.append({"name": name, **audit})
    artifact = {
        "metadata": {
            "schema_version": 1,
            "provenance_file": "provenance.json",
            "validation_file": "validation.json",
            "method": "shifted-beta-geometric",
            "origin": "fresh synthetic beta-geometric sampling",
            "software": {"python": platform.python_version(), **actual_versions},
            "random_generator": "numpy.random.Generator(numpy.random.PCG64(seed))",
            "duration_definition": "T = K + 1; K counts completed renewals before first churn",
            "survivor_definition": "survivors[t] counts generated T > t, including all customers at t = 0",
            "distribution": "scipy.stats.betanbinom with n=1, a=alpha, b=beta, loc=0",
            "survival_definition": "S(t) = P(T > t) = P(K >= t) = betanbinom.sf(t-1, 1, alpha, beta)",
            "retention_definition": "r(t) = S(t)/S(t-1) for t >= 1",
            "mean_definition": "E[T] = betanbinom.mean(1, alpha, beta) + 1",
            "null_mean_definition": "null denotes an unbounded mean when alpha <= 1",
            "survival_periods": list(range(HORIZON + 1)),
            "retention_periods": list(range(1, HORIZON + 1)),
            "churn_periods": list(range(1, HORIZON + 1)),
            "loglikelihood_definition": "sum_t losses[t-1]*betanbinom.logpmf(t-1,1,alpha,beta) + survivors[20]*betanbinom.logsf(19,1,alpha,beta)",
            "loglikelihood_constant": "multinomial count coefficient omitted",
            "fitting": "unpenalized maximum likelihood in log-parameter space; 16 BFGS starts without parameter bounds",
            "objective_scale": "negative loglikelihood divided by the initial customer count",
            "gradient": "analytic beta-function score evaluated with scipy.special.digamma",
            "optimizer_options": {"gtol": 1e-8, "maxiter": 1000},
            "tolerances": TOLERANCES,
            "reference": {"authors": ["Peter S. Fader", "Bruce G. S. Hardie"], "year": 2007, "title": "How to project customer retention", "doi": "10.1002/dir.20074"},
        },
        "fixed_cases": fixed_cases,
        "fit_cases": fit_cases,
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    write_json(args.output_directory / "sbg.json", artifact)
    write_json(args.output_directory / "validation.json", {"fixed_cases": fixed_audits, "fit_cases": fit_audits})
    print("Generated sbg.json and validation.json")


if __name__ == "__main__":
    main()
