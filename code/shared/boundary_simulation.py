"""B2 known-truth checks for center localization and working-interval coverage.

Use the common boundary estimator unchanged. These synthetic checks assess
numerical resolution and model assumptions; they do not supply chemical truth
for measured profiles. The retained RNG stream matches the historical study.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr
from scipy.stats import norm

if __package__:
    from .boundary_model import TransitionGrid, fit_step
else:
    from boundary_model import TransitionGrid, fit_step


DEPTH = np.r_[np.arange(10, 91, 10), 95].astype(float)
LOCALIZATION_SCENARIOS = (
    ("matched_iid", "sigmoid", "iid_gaussian"),
    ("matched_correlated", "sigmoid", "correlated_gaussian"),
    ("matched_t5", "sigmoid", "standardized_t5"),
    ("erf_shape", "erf", "iid_gaussian"),
    ("shallow_dip", "dip", "iid_gaussian"),
    ("sensor_bias", "sensor", "iid_gaussian"),
)


def mean_profiles(depth: np.ndarray, boundary: np.ndarray, width: float = 3.0,
                  shape: str = "sigmoid") -> np.ndarray:
    """Decreasing Eh, planted center b; width is logistic-equivalent scale."""
    b = np.asarray(boundary)[..., None]
    if shape == "erf":
        # The erf transition has the same planted 10%-90% width as the sigmoid.
        state = ndtr((b - depth) / (width * np.log(9) / norm.ppf(.9)))
    else:
        state = expit((b - depth) / width)
    mu = -350 + 800 * state
    if shape == "dip":
        mu = mu - 350 * np.exp(-.5 * ((depth - 30) / 7) ** 2)
    elif shape == "sensor":
        mu[..., np.argmin(abs(depth - 60))] += 200
    elif shape not in ("sigmoid", "erf"):
        raise ValueError(shape)
    return mu


def noise(rng: np.random.Generator, n: int, horizon: int, depth: np.ndarray,
          family: str, sigma: float = 25.) -> np.ndarray:
    """Known marginal SD; AR draws start at their stationary Gaussian law."""
    shape = (n, horizon, len(depth))
    if family == "iid_gaussian":
        return rng.normal(0, sigma, shape)
    if family == "standardized_t5":
        return rng.standard_t(5, shape) * (sigma * np.sqrt(3 / 5))
    if family != "correlated_gaussian":
        raise ValueError(family)
    covariance = .6 ** (abs(depth[:, None] - depth[None, :]) / 10)
    root = np.linalg.cholesky(covariance)
    innovation = (rng.normal(size=shape) @ root.T) * sigma
    out = np.empty_like(innovation)
    out[:, 0] = innovation[:, 0]
    for t in range(1, horizon):
        out[:, t] = .7 * out[:, t - 1] + np.sqrt(1 - .7 ** 2) * innovation[:, t]
    return out


def _localization(cfg: dict, out: Path, rng: np.random.Generator,
                  models: dict[str, TransitionGrid], depth: np.ndarray) -> dict:
    n = int(cfg.get("n_localization", 500))
    sigma = 25.
    true_b = 55 + 6 * np.sin(2 * np.pi * (np.arange(n) % 24) / 24)
    summaries, records, examples = [], [], []
    for scenario, shape, family in LOCALIZATION_SCENARIOS:
        truth = mean_profiles(depth, true_b, shape=shape)
        # One continuous sequence for AR; proportions are descriptive, not an
        # independence-based binomial interval across these correlated profiles.
        y = truth + noise(rng, 1, n, depth, family, sigma)[0]
        fits = {name: model.fit(y) for name, model in models.items()}
        fits["step"] = fit_step(y, depth)
        for name, fit in fits.items():
            b = np.asarray(fit["b"]).reshape(-1)
            amplitude = np.asarray(fit["amplitude"]).reshape(-1)
            width = np.asarray(fit.get("w", np.full(n, np.nan))).reshape(-1)
            usable = np.isfinite(b)
            lower, upper = np.full(n, np.nan), np.full(n, np.nan)
            truncated = np.zeros(n, dtype=bool)
            disconnected = np.zeros(n, dtype=bool)
            if name == "sigmoid":
                interval = models[name].profile_interval(y, sigma=sigma, level=.95)
                lower = np.asarray(interval["b_low"]).reshape(-1)
                upper = np.asarray(interval["b_high"]).reshape(-1)
                truncated = np.asarray(interval["grid_truncated"]).reshape(-1)
                disconnected = np.asarray(interval["support_components"]).reshape(-1) > 1
            interval_available = usable & np.isfinite(lower) & np.isfinite(upper)
            covered = interval_available & (lower <= true_b) & (true_b <= upper)
            error = b[usable] - true_b[usable]
            edge = np.asarray(fit.get("grid_edge", np.zeros(n, dtype=bool))).reshape(-1)
            summary = dict(
                scenario=scenario, shape=shape, noise=family, model=name, n=n,
                has_planted_boundary=True,
                boundary_available_fraction=float(usable.mean()),
                b_bias_cm=float(error.mean()) if len(error) else np.nan,
                b_rmse_cm=float(np.sqrt(np.mean(error ** 2))) if len(error) else np.nan,
                w_bias_cm=float(np.nanmean(width - 3)) if name != "step" else np.nan,
                w_rmse_cm=float(np.sqrt(np.nanmean((width - 3) ** 2))) if name != "step" else np.nan,
                working_interval_available_fraction=float(interval_available.mean()) if name == "sigmoid" else np.nan,
                working_interval_coverage_all=float(covered.mean()) if name == "sigmoid" else np.nan,
                working_interval_coverage_available=float(covered[interval_available].mean()) if interval_available.any() else np.nan,
                working_interval_median_width_cm=float(np.nanmedian(upper - lower)) if interval_available.any() else np.nan,
                working_interval_grid_truncated_fraction=float(truncated.mean()) if name == "sigmoid" else np.nan,
                working_interval_disconnected_fraction=float(disconnected.mean()) if name == "sigmoid" else np.nan,
                grid_edge_fraction=float(edge.mean()) if name != "step" else np.nan,
            )
            summaries.append(summary)
            for i in range(n):
                records.append(dict(scenario=scenario, model=name, replicate=i,
                    true_b_cm=float(true_b[i]),
                    estimated_b_cm=float(b[i]), estimated_w_cm=float(width[i]),
                    amplitude_mv=float(amplitude[i]), b_low_cm=float(lower[i]),
                    b_high_cm=float(upper[i]), grid_edge=bool(edge[i])))
            for i in range(min(3, n)):
                for d, z in enumerate(depth):
                    examples.append(dict(scenario=scenario, replicate=i, model=name,
                        depth_cm=float(z), observed_eh_mv=float(y[i, d]),
                        true_eh_mv=float(truth[i, d]), fitted_eh_mv=float(fit["prediction"][i, d]),
                        true_b_cm=float(true_b[i]),
                        estimated_b_cm=float(b[i])))
    pd.DataFrame(summaries).to_csv(out / "localization_summary.csv", index=False)
    pd.DataFrame(records).to_csv(out / "localization_profiles.csv", index=False)
    pd.DataFrame(examples).to_csv(out / "localization_examples.csv", index=False)
    return {"n_profiles_per_scenario": n, "n_scenarios": len(LOCALIZATION_SCENARIOS),
            "models": list(models) + ["step"], "summary": "localization_summary.csv"}


def grid_resolution_diagnostic(cfg: dict, out: str | Path) -> pd.DataFrame:
    """Post-hoc grid sensitivity for the B2 working interval.

    Replays the historical localization RNG stream, including skipped scenarios, so
    profiles are paired with the original experiment. Half-cell expansion is
    a numerical diagnostic, not a newly calibrated confidence procedure. Width
    candidates and the working Gaussian sigma remain unchanged throughout.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = int(cfg.get("n_localization", 500))
    seed = int(cfg.get("seed", 20260923))
    depth = np.asarray(cfg.get("depth", DEPTH), dtype=float)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[0])
    true_b = 55 + 6 * np.sin(2 * np.pi * (np.arange(n) % 24) / 24)
    selected = {"matched_iid", "shallow_dip", "sensor_bias"}
    rows = []
    for scenario, shape, family in LOCALIZATION_SCENARIOS:
        y = mean_profiles(depth, true_b, shape=shape) + noise(rng, 1, n, depth, family, 25.)[0]
        if scenario not in selected:
            continue
        steps = (1., .5, .25, .1) if scenario == "matched_iid" else (1., .1)
        for step in steps:
            b_grid = np.round(np.arange(15., 90. + step / 2, step), 10)
            interval = TransitionGrid(depth, b_grid=b_grid).profile_interval(y, sigma=25., level=.95)
            b = interval["b"]
            available = np.isfinite(b) & np.isfinite(interval["b_low"]) & np.isfinite(interval["b_high"])
            rmse = float(np.sqrt(np.mean((b[np.isfinite(b)] - true_b[np.isfinite(b)]) ** 2)))
            for endpoint_rule, expansion in (("grid_points_hull", 0.),
                                               ("half_cell_expansion_diagnostic", step / 2)):
                lower = np.maximum(interval["b_low"] - expansion, b_grid[0])
                upper = np.minimum(interval["b_high"] + expansion, b_grid[-1])
                width = upper - lower
                covered = available & (lower <= true_b) & (true_b <= upper)
                rows.append(dict(scenario=scenario, n=n, seed=seed,
                    analysis_status="post_hoc_numerical_sensitivity",
                    b_grid_step_cm=step, width_grid="1,2,3,5,8,12,18",
                    endpoint_rule=endpoint_rule, working_sigma_mv=25.,
                    b_rmse_cm=rmse,
                    working_interval_coverage_all=float(covered.mean()),
                    working_interval_coverage_available=float(covered[available].mean()),
                    working_interval_median_width_cm=float(np.nanmedian(width)),
                    working_interval_mean_width_cm=float(np.nanmean(width)),
                    working_interval_zero_width_fraction=float((available & np.isclose(width, 0.)).mean()),
                    working_interval_available_fraction=float(available.mean()),
                    working_interval_grid_truncated_fraction=float(interval["grid_truncated"].mean()),
                    working_interval_disconnected_fraction=float((interval["support_components"] > 1).mean()),
                    original_profiles_replayed=True))
    result = pd.DataFrame(rows)
    result.to_csv(out / "grid_resolution_diagnostic.csv", index=False)
    return result


def run(cfg: dict, out: str | Path) -> dict:
    """Run B2 localization and grid checks with the historical random draws."""
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if int(cfg.get("n_localization", 500)) < 2:
        raise ValueError("Localization requires at least two profiles per scenario")
    seed = int(cfg.get("seed", 20260923))
    # Child zero deliberately preserves the historical localization stream.
    stream = np.random.SeedSequence(seed).spawn(2)[0]
    depth = np.asarray(cfg.get("depth", DEPTH), dtype=float)
    models = {kind: TransitionGrid(depth, kind=kind) for kind in ("sigmoid", "sigmoid_trend")}
    localization = _localization(cfg, out, np.random.default_rng(stream), models, depth)
    grid = grid_resolution_diagnostic(cfg, out)
    summary = dict(
        scope="B2 numerical validation: localization and working-interval coverage",
        seed=seed, localization=localization,
        grid_diagnostic="grid_resolution_diagnostic.csv", grid_diagnostic_rows=len(grid),
        elapsed_seconds=time.perf_counter() - started,
        interpretation=[
            "Synthetic known-truth checks do not validate a chemical reaction interface.",
            "Six retained scenarios have a planted central transition; the historical random draws are preserved.",
            "Correlated localization profiles are serially dependent; coverage proportions have no binomial interval.",
            "95% profile-SSE working intervals assume supplied Gaussian sigma=25 mV and model shape.",
            "Intervals are hulls of accepted grid points; zero width is not zero estimation error.",
            "Half-cell endpoint expansion is a numerical diagnostic, not a calibrated confidence procedure.",
            "Erf has the same planted 10%-90% width; dip and static sensor-bias cases target the uncorrupted center.",
            "Model shape and width candidates are unchanged; finer location grids do not repair model misspecification.",
        ],
    )
    (out / "simulation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
