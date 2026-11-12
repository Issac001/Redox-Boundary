"""Recompute pooled descriptive fits and fixed-threshold diagnostics from Eh.

These descriptive outputs are independent of the per-profile position analyses.
They do not use saved fit parameters, historical predictions, or chemical data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .boundary_model import TransitionGrid


def crossings(eh, depth, threshold):
    """Interpolate strict sign changes between adjacent sensor observations.

    This is the Figure C.1 convention. An observation exactly equal to the
    threshold is not a strict crossing; the count of such observations is
    reported separately in threshold_profiles.csv rather than silently omitted
    from the audit. No present-manuscript observation equals its tested threshold.
    """
    values = np.asarray(eh, dtype=float) - threshold
    depth = np.asarray(depth, dtype=float)
    indices = np.flatnonzero(values[:-1] * values[1:] < 0)
    return (depth[indices] + (depth[indices + 1] - depth[indices])
            * (-values[indices]) / (values[indices + 1] - values[indices]))


def prepare_figure_data(observations, numeric_dir):
    """Save pooled parameters, 1001 curve points and threshold diagnostics.

    ``observations`` is the complete Eh DataFrame with Case, Time, profile_id,
    Depth, and Eh columns. Every profile must have identical depth coverage.
    Fitting the mean at each depth then exactly minimizes the full equal-weight
    observation SSE: the within-depth residual term is independent of the curve.
    Fit statistics are evaluated against all observations, not the depth means.
    """
    required = {"Case", "Time", "profile_id", "Depth", "Eh"}
    if not required.issubset(observations.columns):
        raise ValueError(f"Missing Eh columns: {sorted(required - set(observations.columns))}")
    obs = observations.copy()
    obs["Case"] = obs.Case.astype(str)
    if obs.empty or obs[list(required)].isna().any().any():
        raise ValueError("Pooled fitting requires nonmissing complete observations")
    if not np.isfinite(obs[["Time", "Depth", "Eh"]].to_numpy(float)).all():
        raise ValueError("Pooled fitting requires finite numerical observations")
    if obs.duplicated(["profile_id", "Depth"]).any():
        raise ValueError("Duplicate profile/depth observations")
    grid = obs.pivot(index="profile_id", columns="Depth", values="Eh").sort_index(axis=1)
    if grid.isna().any().any():
        raise ValueError("Pooled fitting requires identical depths in all profiles")
    depth = grid.columns.to_numpy(float)
    model = TransitionGrid(depth, b_grid=np.arange(15.0, 91.0),
                           w_grid=np.array([1., 2., 3., 5., 8., 12., 18.]))
    fitted = model.fit(grid.mean(axis=0).to_numpy(float))
    if not fitted["identifiable"][0]:
        raise ValueError("The pooled profile does not identify a sigmoid center")
    parameters = {"a": float(fitted["intercept"][0]),
                  "A": float(fitted["amplitude"][0]),
                  "c": float(fitted["b"][0]), "w": float(fitted["w"][0])}
    predicted = model.predict(fitted, obs.Depth.to_numpy(float))[0]
    residual = obs.Eh.to_numpy(float) - predicted
    total = np.sum((obs.Eh - obs.Eh.mean()) ** 2)
    record = {
        "parameters": parameters,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1 - np.sum(residual ** 2) / total),
        "n_profiles": len(grid), "n_observations": len(obs),
        "estimator": "Constrained sigmoid on depth means; equal observation weights",
        "position_grid_cm": model.b_grid.tolist(),
        "width_grid_cm": model.w_grid.tolist(),
        "refitted_from_observations": True,
        "uncertainty": "Descriptive pooled fit; no independent-time confidence interval",
    }
    query = np.linspace(depth.min(), depth.max(), 1001)
    predictions = pd.DataFrame({"depth_cm": query,
                                "Eh_mV": model.predict(fitted, query)[0]})
    counts = []
    for (case, time, pid), part in obs.groupby(["Case", "Time", "profile_id"], sort=True):
        part = part.sort_values("Depth")
        for threshold in (0., 200., 400.):
            positions = crossings(part.Eh, part.Depth, threshold)
            counts.append({"Case": case, "Time": time, "profile_id": pid,
                           "threshold_mV": threshold, "n_crossings": len(positions),
                           "exact_threshold_observations": int(part.Eh.eq(threshold).sum()),
                           "crossing_depths_cm": json.dumps(positions.tolist())})
    counts = pd.DataFrame(counts)
    summary = []
    for (case, threshold), part in counts.groupby(["Case", "threshold_mV"], sort=True):
        summary.append({"Case": case, "threshold_mV": threshold,
                        "n_profiles": len(part),
                        "no_crossing_percent": float(part.n_crossings.eq(0).mean() * 100),
                        "one_crossing_percent": float(part.n_crossings.eq(1).mean() * 100),
                        "multiple_crossing_percent": float(part.n_crossings.gt(1).mean() * 100),
                        "exact_threshold_observations": int(part.exact_threshold_observations.sum())})
    out = Path(numeric_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pooled_fit_parameters.json").write_text(json.dumps(record, indent=2))
    predictions.to_csv(out / "pooled_sigmoid_predictions.csv", index=False)
    counts.to_csv(out / "threshold_profiles.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "threshold_summary.csv", index=False)
    return record
