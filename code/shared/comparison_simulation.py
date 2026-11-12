"""The manuscript's four independent-profile simulation scenarios.

Every method receives the same observed array for each scenario. Smooth truth
uses the maximum negative derivative as its center; a step uses its jump
location. The erf and Gompertz scales match the logistic 10--90% span,
2*log(9)*width. Width is a logistic scale, not that full transition span.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr, ndtri


def scenario_catalog() -> list[dict[str, Any]]:
    """Return only the four fixed manuscript scenarios, in presentation order."""
    return [dict(id=identifier, family=family, amplitude=800., width=width,
                 sigma=25., center_low=30., center_high=80., level=50.,
                 noise="gaussian")
            for identifier, family, width in [
                ("logistic_w8_a800_s25", "logistic", 8.),
                ("step_a800_s25", "step", 0.),
                ("erf_w8_a800_s25", "erf", 8.),
                ("gompertz_w8_a800_s25", "gompertz", 8.)]]


def scenario_seed(master_seed: int, scenario_id: str) -> int:
    """Stable unsigned seed independent of catalogue order and Python hashing."""
    encoded = f"{int(master_seed)}:{scenario_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def _depth(depth):
    z = np.asarray(depth, dtype=float)
    if z.ndim != 1 or len(z) < 4 or not np.isfinite(z).all() or np.any(np.diff(z) <= 0):
        raise ValueError("depth must have >=4 finite, strictly increasing values")
    return z


def generate_scenario(spec: dict, n: int, seed: int, depth) -> dict[str, Any]:
    """Generate independent Gaussian-error profiles and continuous true centers.

    Center draws precede noise draws to reproduce the manuscript's original
    random streams exactly. No rounding aligns true centers to candidate grids.
    """
    z = _depth(depth)
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    family = spec["family"]
    amplitude, width, sigma = (float(spec[k]) for k in ("amplitude", "width", "sigma"))
    if not np.isfinite([amplitude, width, sigma]).all() or amplitude <= 0 or sigma <= 0:
        raise ValueError("amplitude and sigma must be finite and positive")
    if (family == "step" and width != 0) or (family != "step" and width <= 0):
        raise ValueError("step width must be zero; smooth width must be positive")
    if spec.get("noise", "gaussian") != "gaussian":
        raise ValueError("The manuscript uses independent Gaussian errors")
    low, high = float(spec["center_low"]), float(spec["center_high"])
    if not np.isfinite([low, high]).all() or low >= high or low < z[0] or high > z[-1]:
        raise ValueError("center range must be ordered and within observed depth range")
    center = rng.uniform(low, high, n)
    level = float(spec.get("level", 50.))
    if not np.isfinite(level):
        raise ValueError("level must be finite")
    zz, cc = z[None, :], center[:, None]
    shape_scale = width
    target_definition = "maximum_negative_derivative"
    truth_type = "single_smooth"
    if family == "logistic":
        fraction = expit((cc - zz) / width)
    elif family == "step":
        fraction = (zz < cc).astype(float)
        truth_type, target_definition = "single_jump", "jump_location"
    elif family == "erf":
        shape_scale = np.log(9.) * width / ndtri(.9)
        fraction = ndtr((cc - zz) / shape_scale)
    elif family == "gompertz":
        span = np.log(-np.log(.1)) - np.log(-np.log(.9))
        shape_scale = 2 * np.log(9.) * width / span
        # For q=exp(-exp((z-c)/s)), |q'|=u*exp(-u)/s is maximal at u=1.
        fraction = np.exp(-np.exp(np.clip((zz - cc) / shape_scale, -700, 700)))
    else:
        raise ValueError(f"unknown manuscript simulation family: {family}")
    mean = level - amplitude / 2 + amplitude * fraction
    noise = sigma * rng.normal(size=mean.shape)
    gap_index = np.searchsorted(z, center, side="right") - 1
    phase = (center - z[gap_index]) / (z[gap_index + 1] - z[gap_index])
    return dict(observed=mean + noise, mean=mean, true_center=center,
                true_sigma=sigma, true_gap_index=gap_index, true_gap_phase=phase,
                truth_type=truth_type, target_definition=target_definition,
                shape_scale=float(shape_scale),
                transition_span_10_90=2 * np.log(9.) * width if width > 0 else 0.,
                spec=dict(spec), seed=int(seed), depth=z.copy())


def _fit_array(fit, key, n, default=np.nan, dtype=float):
    value = np.asarray(fit.get(key, default), dtype=dtype)
    if value.ndim == 0:
        return np.full(n, value, dtype=dtype)
    if value.shape != (n,):
        raise ValueError(f"fit[{key!r}] must be scalar or shape ({n},), got {value.shape}")
    return value


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else np.nan


def profile_records(spec: dict, method: str, fit: dict, truth: dict, depth) -> pd.DataFrame:
    """Per-replicate localization records; unavailable intervals fail coverage.

    No diagnostic flags select or discard simulation replicates. A disconnected
    support is evaluated using its reported interval hull, as in the manuscript.
    """
    _depth(depth)
    true_center = np.asarray(truth["true_center"], dtype=float)
    n = len(true_center)
    if not np.isfinite(true_center).all():
        raise ValueError("All manuscript scenarios require finite true centers")
    center, lower, upper = (_fit_array(fit, key, n) for key in ("center", "lower", "upper"))
    estimate_valid = np.isfinite(center)
    interval_valid = np.isfinite(lower) & np.isfinite(upper) & (upper >= lower)
    error = np.where(estimate_valid, center - true_center, np.nan)
    return pd.DataFrame(dict(
        scenario=spec["id"], family=spec["family"], method=method,
        replicate=np.arange(n), seed=truth["seed"],
        truth_type=truth["truth_type"], target_definition=truth["target_definition"],
        true_center=true_center, center=center,
        center_map=_fit_array(fit, "center_map", n),
        center_median=_fit_array(fit, "center_median", n),
        lower=lower, upper=upper, error=error, absolute_error=np.abs(error),
        estimate_available=estimate_valid, interval_available=interval_valid,
        covered=interval_valid & (lower <= true_center) & (true_center <= upper),
        interval_length=np.where(interval_valid, upper - lower, np.nan),
        true_gap_phase=truth["true_gap_phase"],
        width_edge=_fit_array(fit, "width_edge", n, False, bool),
        center_edge=_fit_array(fit, "center_edge", n, False, bool),
        interval_components=_fit_array(fit, "interval_components", n),
        amplitude=_fit_array(fit, "amplitude", n), width=_fit_array(fit, "width", n),
        sse=_fit_array(fit, "sse", n), interval_kind=fit.get("interval_kind", "unspecified")))


def summarize_fit(spec: dict, method: str, fit: dict, truth: dict, depth,
                  records: pd.DataFrame | None = None) -> dict[str, Any]:
    """Summarize all replicates without diagnostic preselection.

    Coverage uses all n replicates; its Monte Carlo standard error is reported
    both as a proportion and in percentage points. RMSE and median length use
    available estimates/intervals, with counts explicitly reported. The current
    four scenarios have no missing fitted centers or intervals.
    """
    data = profile_records(spec, method, fit, truth, depth) if records is None else records
    n = len(data)
    if not n:
        raise ValueError("Cannot summarize an empty simulation")
    error = data.error.to_numpy()
    finite_error = np.isfinite(error)
    length = data.interval_length.to_numpy()
    finite_length = np.isfinite(length)
    coverage = float(data.covered.mean())
    mcse = float(np.sqrt(coverage * (1. - coverage) / n))
    return dict(
        scenario=spec["id"], family=spec["family"], method=method, n=n,
        seed=int(truth["seed"]), truth_type=truth["truth_type"],
        target_definition=truth["target_definition"],
        amplitude=spec["amplitude"], width=spec["width"], sigma=spec["sigma"],
        n_valid_estimate=int(finite_error.sum()), n_valid_interval=int(finite_length.sum()),
        estimate_availability=float(data.estimate_available.mean()),
        interval_availability=float(data.interval_available.mean()),
        bias=_finite_mean(error),
        rmse=float(np.sqrt(np.mean(error[finite_error] ** 2))) if finite_error.any() else np.nan,
        coverage_all=coverage, coverage_all_mcse=mcse,
        coverage_percent=100. * coverage, coverage_mcse_pp=100. * mcse,
        median_interval_length=float(np.median(length[finite_length])) if finite_length.any() else np.nan,
        interval_kind=str(data.interval_kind.iloc[0]))
