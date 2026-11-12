# SPDX-License-Identifier: MIT
"""Generate deterministic BG/NBD and BG/BB references using Lifetimes."""
import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from collections import Counter
import hashlib
from importlib import metadata
import inspect
import json
import math
from pathlib import Path
import platform

from autograd import value_and_grad
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import roots_jacobi
from lifetimes import BetaGeoFitter, BetaGeoBetaBinomFitter
import lifetimes.fitters as base_module
import lifetimes.fitters.beta_geo_fitter as nbd_module
import lifetimes.fitters.beta_geo_beta_binom_fitter as bb_module

VERSIONS = {
    "Lifetimes": "0.11.3", "numpy": "2.2.6", "scipy": "1.16.3",
    "pandas": "2.2.3", "autograd": "1.8.0", "dill": "0.4.0",
    "python-dateutil": "2.9.0.post0", "pytz": "2025.2",
    "tzdata": "2025.2", "six": "1.17.0",
}
PARAMETER_ORDER = {"BG/NBD": ["r", "alpha", "a", "b"],
                   "BG/BB": ["alpha", "beta", "gamma", "delta"]}
SOURCE_HASHES = {
    "lifetimes/fitters/__init__.py": "77a75fc3763c63943884f2d70c4386f28e3e23c80b1d34ec36469607f0816b8c",
    "lifetimes/fitters/beta_geo_fitter.py": "1304c4c12da26125069a564fc7747d4589f7563c1941cd70e105675368ef6069",
    "lifetimes/fitters/beta_geo_beta_binom_fitter.py": "a58207289f533c11459b0dc3320e3738d9aeac8083ef0ec7b5042c7572e01fcd",
}
TAG_COMMIT = "bd3ae6e13bc82b46fca7ce93130b2cfce4e8d7cf"
SOURCES = {
    "BG/NBD": {
        "authors": ["Peter S. Fader", "Bruce G. S. Hardie", "Ka Lok Lee"],
        "year": 2005,
        "title": '"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model',
        "doi": "10.1287/mksc.1040.0098",
        "url": "https://doi.org/10.1287/mksc.1040.0098",
        "equations": ["6", "7", "10"],
    },
    "BG/BB": {
        "authors": ["Peter S. Fader", "Bruce G. S. Hardie", "Jen Shang"],
        "year": 2010,
        "title": "Customer-Base Analysis in a Discrete-Time Noncontractual Setting",
        "doi": "10.1287/mksc.1100.0580",
        "url": "https://doi.org/10.1287/mksc.1100.0580",
        "equations": ["4", "5", "6", "11", "13", "A10"],
    },
}
NBD_PARAMETERS = [
    dict(r=0.8, alpha=7.5, a=1.4, b=3.2),
    dict(r=1.6, alpha=4.2, a=0.7, b=2.8),
    dict(r=2.2, alpha=10.0, a=2.6, b=1.5),
]
BB_PARAMETERS = [
    dict(alpha=1.8, beta=2.4, gamma=1.2, delta=4.1),
    dict(alpha=0.9, beta=3.1, gamma=0.7, delta=2.8),
    dict(alpha=2.2, beta=1.4, gamma=2.4, delta=2.2),
]


def check_versions():
    if platform.python_version_tuple()[:2] != ("3", "13"):
        raise RuntimeError("Python 3.13 is required")
    for name, version in VERSIONS.items():
        if metadata.version(name) != version:
            raise RuntimeError(f"Version mismatch for {name}")


def source_provenance():
    modules = {
        "lifetimes/fitters/__init__.py": base_module,
        "lifetimes/fitters/beta_geo_fitter.py": nbd_module,
        "lifetimes/fitters/beta_geo_beta_binom_fitter.py": bb_module,
    }
    for path, module in modules.items():
        digest = hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest()
        if digest != SOURCE_HASHES[path]:
            raise RuntimeError(f"Source hash mismatch for {path}")
    return {
        "package": "Lifetimes", "version": "0.11.3", "license": "MIT",
        "release": "https://pypi.org/project/Lifetimes/0.11.3/",
        "wheel_sha256": "261e3dc89977c2b60767ca6728aaa29d429de0c814635070a749c36555895da8",
        "git_tag": "v0.11.3", "git_commit": TAG_COMMIT,
        "source_files": [{
            "path": path,
            "url": f"https://github.com/CamDavidsonPilon/lifetimes/blob/{TAG_COMMIT}/{path}",
            "sha256": hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest(),
        } for path, module in modules.items()],
        "dependencies": VERSIONS, "python": "3.13",
    }


def scalar(value):
    result = float(np.asarray(value).reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError("Nonfinite reference value")
    return result


def array_history(history, continuous):
    return tuple(np.array([history[k]], dtype=float if k in ("tx", "t") and continuous else int)
                 for k in (("x", "tx", "t") if continuous else ("x", "tx", "n")))


def nbd_expected(parameters, history, horizon):
    model = BetaGeoFitter(penalizer_coef=0.0)
    model.params_ = pd.Series(parameters, dtype=float)
    x, tx, t = array_history(history, True)
    ll = -BetaGeoFitter._negative_log_likelihood(
        np.log([parameters[k] for k in PARAMETER_ORDER["BG/NBD"]]), x, tx, t, np.ones(1), 0.0)
    active = scalar(model.conditional_probability_alive(x, tx, t))
    mean = scalar(model.conditional_expected_number_of_purchases_up_to_time(horizon, x, tx, t))
    return dict(log_likelihood=scalar(ll), probability_active=active,
                expected_if_active=mean / active, expected_purchases=mean)


def bb_expected(parameters, history, horizon):
    model = BetaGeoBetaBinomFitter(penalizer_coef=0.0)
    model.params_ = pd.Series(parameters, dtype=float)
    x, tx, n = array_history(history, False)
    return dict(
        log_likelihood=scalar(model._loglikelihood([parameters[k] for k in PARAMETER_ORDER["BG/BB"]], x, tx, n)),
        probability_alive=scalar(model.conditional_probability_alive(1, x, tx, n)),
        expected_future_purchases=scalar(
            model.conditional_expected_number_of_purchases_up_to_time(horizon, x, tx, n)),
    )


def pattern_for(history):
    """Choose one binary pattern with the stated sufficient statistics."""
    x, tx, n = (history[k] for k in ("x", "tx", "n"))
    assert 0 <= x <= tx <= n and n >= 1
    assert (x == 0) == (tx == 0)
    pattern = [0] * n
    if x:
        for j in range(x - 1):
            pattern[j] = 1
        pattern[tx - 1] = 1
    return pattern


def forward_masses(p, q, pattern, horizon):
    """Track alive/dead joint masses, with dropout before each purchase chance."""
    alive, dead = np.ones_like(p + q), np.zeros_like(p + q)
    for bit in pattern:
        newly_dead = alive * q
        surviving = alive * (1.0 - q)
        if bit:
            alive, dead = surviving * p, np.zeros_like(dead)
        else:
            alive, dead = surviving * (1.0 - p), dead + newly_dead
    likelihood = alive + dead
    next_alive = alive * (1.0 - q)
    mean_mass = np.zeros_like(alive)
    future_alive = alive
    for _ in range(horizon):
        future_alive = future_alive * (1.0 - q)
        mean_mass = mean_mass + future_alive * p
    return likelihood, next_alive, mean_mass


def beta_nodes(alpha, beta, order):
    nodes, weights = roots_jacobi(order, beta - 1.0, alpha - 1.0)
    return (nodes + 1.0) / 2.0, weights / weights.sum()


def integrated_forward(parameters, history, horizon):
    """Integrate finite-state masses with Gauss-Jacobi beta quadrature."""
    order = max(16, history["n"] + horizon + 2)
    p, wp = beta_nodes(parameters["alpha"], parameters["beta"], order)
    q, wq = beta_nodes(parameters["gamma"], parameters["delta"], order)
    masses = forward_masses(p[:, None], q[None, :], pattern_for(history), horizon)
    likelihood, alive_mass, mean_mass = [scalar(np.einsum("i,ij,j->", wp, m, wq)) for m in masses]
    return dict(log_likelihood=math.log(likelihood), probability_alive=alive_mass / likelihood,
                expected_future_purchases=mean_mass / likelihood)


def fixed_cases(continuous):
    if continuous:
        histories = [(0, 0.0, 12.0), (1, 1.0, 12.0), (3, 8.0, 12.0),
                     (4, 12.0, 12.0), (8, 11.5, 12.0), (0, 0.0, 0.0)]
        sets, horizons, evaluate = NBD_PARAMETERS, [0.0, 0.25, 6.0, 20.0], nbd_expected
    else:
        histories = [(0, 0, 1), (0, 0, 6), (1, 1, 6), (2, 4, 6), (3, 6, 6), (6, 6, 6)]
        sets, horizons, evaluate = BB_PARAMETERS, [0, 1, 5, 12], bb_expected
    cases = []
    for pi, parameters in enumerate(sets):
        for hi, (x, tx, end) in enumerate(histories):
            for horizon in horizons:
                history = dict(x=x, tx=tx, **({"t": end} if continuous else {"n": end}), weight=1)
                case = dict(id=f"p{pi}_h{hi}_m{horizon:g}", parameters=parameters,
                            history=history, horizon=horizon,
                            expected=evaluate(parameters, history, horizon))
                if not continuous:
                    forward = integrated_forward(parameters, history, horizon)
                    differences = {key: case["expected"][key] - forward[key] for key in forward}
                    assert max(abs(v) for v in differences.values()) < 1e-9
                    case["independent_check"] = dict(method="finite-state beta quadrature",
                                                     expected=forward, library_minus_forward=differences)
                cases.append(case)
    return cases


def pq_cases():
    configurations = [
        (0.35, 0.2, dict(n=6, tx=4, x=2), 5),
        (0.7, 0.1, dict(n=6, tx=6, x=3), 4),
        (0.2, 0.4, dict(n=6, tx=0, x=0), 5),
        (0.6, 0.3, dict(n=1, tx=1, x=1), 0),
    ]
    for p in (0.0, 1.0):
        for q in (0.0, 1.0):
            for history in (dict(n=3, tx=0, x=0), dict(n=3, tx=3, x=3), dict(n=3, tx=2, x=1)):
                configurations.append((p, q, history, 4))
    cases = []
    for i, (p, q, history, horizon) in enumerate(configurations):
        pattern = pattern_for(history)
        masses = [scalar(m) for m in forward_masses(np.array(p), np.array(q), pattern, horizon)]
        likelihood, alive, mean = masses
        expected = dict(likelihood=likelihood,
                        log_likelihood=math.log(likelihood) if likelihood else None,
                        probability_alive=alive / likelihood if likelihood else None,
                        expected_future_purchases=mean / likelihood if likelihood else None)
        cases.append(dict(id=f"pq{i}", parameters=dict(p=p, q=q), history=dict(**history, weight=1),
                          pattern=pattern, horizon=horizon,
                          status="defined" if likelihood else "zero_likelihood", expected=expected))
    return cases


def simulate(continuous):
    seed = 2026110901 if continuous else 2026110902
    rng = np.random.Generator(np.random.PCG64(seed))
    parameters = NBD_PARAMETERS[0] if continuous else BB_PARAMETERS[0]
    rows = []
    if continuous:
        for _ in range(1000):
            t = float(rng.choice([8.0, 16.0, 24.0, 40.0, 60.0]))
            rate = rng.gamma(parameters["r"], 1.0 / parameters["alpha"])
            dropout = rng.beta(parameters["a"], parameters["b"])
            x, tx = 0, 0.0
            while True:
                event_time = tx + rng.exponential(1.0 / rate)
                if event_time > t:
                    break
                x, tx = x + 1, float(event_time)
                if rng.random() < dropout:
                    break
            rows.append(dict(x=x, tx=tx, t=t, weight=1))
    else:
        for _ in range(1000):
            n = int(rng.choice([4, 6, 8]))
            p = rng.beta(parameters["alpha"], parameters["beta"])
            q = rng.beta(parameters["gamma"], parameters["delta"])
            x, tx = 0, 0
            for opportunity in range(1, n + 1):
                if rng.random() < q:
                    break
                if rng.random() < p:
                    x, tx = x + 1, opportunity
            rows.append(dict(x=x, tx=tx, n=n, weight=1))
    return seed, parameters, rows


def vectors(rows, continuous):
    keys = ("x", "tx", "t", "weight") if continuous else ("x", "tx", "n", "weight")
    return tuple(np.array([row[k] for row in rows]) for k in keys)


def finite_gradient(objective, point):
    step = 1e-4
    result = []
    for i in range(len(point)):
        direction = np.zeros(len(point))
        direction[i] = step
        result.append(float((objective(point - 2 * direction) - 8 * objective(point - direction)
                             + 8 * objective(point + direction) - objective(point + 2 * direction)) / (12 * step)))
    return result


def fit_case(continuous):
    seed, true_parameters, rows = simulate(continuous)
    fitter = BetaGeoFitter if continuous else BetaGeoBetaBinomFitter
    x, tx, end, weights = vectors(rows, continuous)
    args = (x, tx, end, weights, 0.0)
    objective = lambda log_parameters: fitter._negative_log_likelihood(log_parameters, *args)
    starts = [np.log(list(true_parameters.values())),
              np.log(list(true_parameters.values())) + np.array([0.25, -0.2, 0.15, -0.15]),
              np.log(list(true_parameters.values())) + np.array([-0.2, 0.2, -0.15, 0.25])]
    results = [minimize(value_and_grad(objective), start, jac=True, method="BFGS",
                        options=dict(gtol=1e-8, maxiter=2000)) for start in starts]
    successful = [result for result in results if result.success]
    if not successful:
        raise RuntimeError("No optimizer start converged")
    best = min(successful, key=lambda result: result.fun)
    fitted = dict(zip(true_parameters, map(float, np.exp(best.x))))
    unweighted_ll = [float(-fitter._negative_log_likelihood(best.x, *vectors([row], continuous), 0.0))
                     for row in rows]
    ll = math.fsum(unweighted_ll)
    assert abs(ll + float(best.fun) * 1000) < 1e-9
    gradient = finite_gradient(objective, best.x)
    assert max(map(abs, gradient)) < 1e-6
    profiles = []
    for i, name in enumerate(fitted):
        direction = np.zeros(4)
        direction[i] = 0.01
        differences = [float((objective(best.x + sign * direction) - best.fun) * 1000) for sign in (-1, 1)]
        assert min(differences) > 0
        profiles.append(dict(parameter=name, log_parameter_offset=0.01,
                             total_negative_log_likelihood_increase=differences))
    optimizer_runs = []
    for start, result in zip(starts, results):
        optimizer_runs.append(dict(
            initial_parameters=dict(zip(fitted, map(float, np.exp(start)))),
            parameters=dict(zip(fitted, map(float, np.exp(result.x)))),
            success=bool(result.success), status=int(result.status), message=str(result.message),
            iterations=int(result.nit), function_evaluations=int(result.nfev),
            gradient_evaluations=int(result.njev),
            total_log_likelihood=float(-result.fun * 1000),
            log_parameter_gradient=list(map(float, result.jac)),
        ))
    prediction_horizons = [0, 2, 6]
    model = fitter(penalizer_coef=0.0)
    model.params_ = pd.Series(fitted)
    predictions = []
    for horizon in prediction_horizons:
        means = model.conditional_expected_number_of_purchases_up_to_time(horizon, x, tx, end)
        predictions.append(dict(horizon=horizon, total_expected_purchases=float(np.sum(means)),
                                mean_expected_purchases=float(np.mean(means))))
    likelihood_check = None
    if not continuous:
        gaps = []
        for row, library_ll in zip(rows, unweighted_ll):
            gaps.append(library_ll - integrated_forward(fitted, row, 0)["log_likelihood"])
        likelihood_check = dict(method="finite-state beta quadrature",
                                max_abs_row_log_likelihood_difference=max(map(abs, gaps)),
                                total_library_minus_forward_log_likelihood=math.fsum(gaps))
    return dict(
        generation=dict(seed=seed, bit_generator="numpy.random.PCG64", true_parameters=true_parameters,
                        algorithm="continuous exponential events with dropout after each repeat purchase" if continuous
                        else "Bernoulli purchases with dropout before each opportunity",
                        observation_lengths=[8, 16, 24, 40, 60] if continuous else [4, 6, 8]),
        history_count=len(rows), rows=rows,
        expected=dict(parameters=fitted, log_likelihood=ll, converged=bool(best.success)),
        diagnostics=dict(
            optimizer="scipy.optimize.minimize BFGS with autograd.value_and_grad",
            objective="unpenalized mean negative log likelihood in original time units",
            log_parameter_gradient=gradient, finite_difference_step=1e-4,
            finite_difference_rule="five-point central difference",
            optimizer_runs=optimizer_runs, parameter_profiles=profiles,
            max_successful_relative_parameter_spread=max(
                float(np.max(np.abs(np.exp(result.x - best.x) - 1))) for result in successful),
            successful_total_log_likelihood_spread=max(float(-r.fun * 1000) for r in successful)
                                                  - min(float(-r.fun * 1000) for r in successful),
            frequency_counts={str(k): v for k, v in sorted(Counter(row["x"] for row in rows).items())},
            unique_summary_count=len({tuple(row.values()) for row in rows}),
            predicted_purchases=predictions, likelihood_check=likelihood_check,
        ),
    )


def build(continuous):
    name = "BG/NBD" if continuous else "BG/BB"
    return dict(
        metadata=dict(
            schema_version=1, model=name, model_source=SOURCES[name],
            parameter_order=PARAMETER_ORDER[name],
            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            implementation=source_provenance(),
            api_paths=([
                "lifetimes.BetaGeoFitter._negative_log_likelihood",
                "lifetimes.BetaGeoFitter.conditional_probability_alive",
                "lifetimes.BetaGeoFitter.conditional_expected_number_of_purchases_up_to_time",
            ] if continuous else [
                "lifetimes.BetaGeoBetaBinomFitter._loglikelihood",
                "lifetimes.BetaGeoBetaBinomFitter._negative_log_likelihood",
                "lifetimes.BetaGeoBetaBinomFitter.conditional_probability_alive",
                "lifetimes.BetaGeoBetaBinomFitter.conditional_expected_number_of_purchases_up_to_time",
            ]),
            likelihood_convention=(
                "One ordered continuous event history, without time/factorial count normalization. "
                "Frequency excludes the initial purchase. All customers start active; x=0 implies probability_active=1. "
                "Fixed log_likelihood is per history; weight multiplies only the fit total."
                if continuous else
                "One binary pattern with given x and last purchase tx; no combinatorial multiplicity. "
                "Customers start active at time zero; dropout precedes each opportunity. "
                "Fixed log_likelihood is per history; weight multiplies only the fit total."
            ),
            probability_convention=("Active at the observation endpoint, after any purchase-triggered dropout."
                                    if continuous else
                                    "Alive at opportunity n+1, including survival before that opportunity; "
                                    "conditional_probability_alive is called with m_periods_in_future=1, independently of horizon."),
            mean_convention=("expected_purchases is conditional on history and averages over activity. "
                             "expected_if_active = expected_purchases / probability_active."
                             if continuous else "Purchases over opportunities n+1 through n+horizon."),
            numerical_note=("All expected values call the unmodified package."
                            if continuous else
                            "The package initializes the dropout sum with 1e-15 before beta normalization. "
                            "Library outputs retain that floor. Finite-state checks report its bias."),
            tolerances=dict(
                fixed=dict(atol=1e-9, rtol=1e-9),
                mle_parameters=dict(atol=1e-5, rtol=1e-4),
                mle_total_log_likelihood=dict(atol=1e-6, rtol=0.0),
                mle_log_parameter_mean_nll_gradient_max_abs=1e-6,
                justification="Fixed values use moderate finite inputs. MLE checks allow optimizer stopping differences; "
                              "the total objective and independently differenced gradient must also agree.",
            ),
        ),
        fixed_cases=fixed_cases(continuous), fit_case=fit_case(continuous),
        **({} if continuous else {"fixed_pq_cases": pq_cases(),
                                 "fixed_pq_convention": "Finite-state forward masses for one stored pattern. "
                                 "p is purchase probability; q is dropout probability. "
                                 "Zero likelihood has log_likelihood=null and undefined conditionals=null; "
                                 "status=zero_likelihood identifies this case."}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    check_versions()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for continuous, filename in ((True, "bgnbd.json"), (False, "bgbb.json")):
        data = build(continuous)
        content = (json.dumps(data, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (args.output_directory / filename).write_bytes(content)
        print(f"{hashlib.sha256(content).hexdigest()}  {filename}")


if __name__ == "__main__":
    main()
