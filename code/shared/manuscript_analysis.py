"""Current-manuscript chemistry, diagnostics and result tables.

These functions consume newly fitted results. No archived estimates, external
source directory, temporal interpolation or observation-support preselection is
used. The three center diagnostics describe the sigmoid fit; all profiles and
all simulation replicates remain in method comparisons.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .boundary_model import TransitionGrid
from .comparison_simulation import scenario_catalog

METHODS = ("sigmoid", "step", "bayes_step")
ANALYTES = ("NO3", "NH4", "Fe", "Mn")
CHEMICAL_COLUMNS = [
    "case", "time_day", "analyte", "n_depths", "b", "w", "amplitude_log1p",
    "sse", "grid_edge", "b_grid_edge", "width_grid_edge", "identifiable",
    "observed_depth_min_cm", "observed_depth_max_cm", "observed_depths_cm",
    "center_within_observed_range", "n_shallower_observations",
    "n_deeper_observations", "nearest_shallower_depth_cm", "nearest_deeper_depth_cm",
    "extrapolated", "role",
]


def _require_columns(frame, columns, label):
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _records(frame):
    """Convert nonfinite values to JSON null without changing saved CSVs."""
    return json.loads(frame.to_json(orient="records", double_precision=15))


def fit_chemical_centers(rows: pd.DataFrame) -> pd.DataFrame:
    """Fit each chemical profile independently on its observed depths <=90 cm.

    Input columns are Case, Time, Depth, NO3, NH4, Fe, Mn. Missing analyte values
    are excluded separately; zero is an observation. At least five distinct
    depths are required. The chemical grid is 15..90 cm by 1 cm with scale
    candidates 1, 2, 3, 5, 8, 12, 18 cm, independent of the Eh comparison grid.
    Concentrations must be nonnegative; infinity is invalid, not missing.
    """
    _require_columns(rows, ["Case", "Time", "Depth", *ANALYTES], "chemistry")
    data = rows.copy()
    if data[["Case", "Time", "Depth"]].isna().any().any():
        raise ValueError("Chemical case, time and depth cannot be missing")
    data["Case"] = data.Case.astype(str)
    for column in ["Time", "Depth", *ANALYTES]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data[["Time", "Depth"]]).all().all():
        raise ValueError("Chemical times and depths must be finite")
    if data.duplicated(["Case", "Time", "Depth"]).any():
        raise ValueError("Duplicate chemical case/time/depth records")
    data = data.loc[data.Depth <= 90].copy()
    values = data[list(ANALYTES)].to_numpy(float)
    if np.isinf(values).any() or np.any(values < 0):
        raise ValueError("Chemical concentrations must be nonnegative or missing")
    centers = []
    for (case, time), part in data.groupby(["Case", "Time"], sort=True):
        for analyte in ANALYTES:
            obs = part.loc[part[analyte].notna()].sort_values("Depth")
            if len(obs) < 5:
                continue
            z = obs.Depth.to_numpy(float)
            direction = 1.0 if analyte == "NO3" else -1.0
            y = direction * np.log1p(obs[analyte].to_numpy(float))
            fitted = TransitionGrid(z, b_grid=np.arange(15., 91.),
                                    w_grid=np.array([1., 2., 3., 5., 8., 12., 18.])).fit(y)
            center = float(fitted["b"][0])
            shallow, deep = z[z < center], z[z > center]
            finite = bool(np.isfinite(center))
            within = bool(finite and z[0] <= center <= z[-1])
            centers.append(dict(
                case=case, time_day=float(time), analyte=analyte, n_depths=len(obs),
                b=center, w=float(fitted["w"][0]),
                amplitude_log1p=float(fitted["amplitude"][0]), sse=float(fitted["sse"][0]),
                grid_edge=bool(fitted["grid_edge"][0]),
                b_grid_edge=bool(fitted["b_grid_edge"][0]),
                width_grid_edge=bool(fitted["width_grid_edge"][0]),
                identifiable=bool(fitted["identifiable"][0]),
                observed_depth_min_cm=float(z[0]), observed_depth_max_cm=float(z[-1]),
                observed_depths_cm=",".join(f"{d:g}" for d in z),
                center_within_observed_range=within,
                n_shallower_observations=len(shallow), n_deeper_observations=len(deep),
                nearest_shallower_depth_cm=float(shallow[-1]) if len(shallow) else np.nan,
                nearest_deeper_depth_cm=float(deep[0]) if len(deep) else np.nan,
                extrapolated=finite and not within, role="descriptive_chemical_only"))
    return pd.DataFrame(centers, columns=CHEMICAL_COLUMNS)


def diagnose_centers(centers: pd.DataFrame, bounds=(20., 90.)) -> pd.DataFrame:
    """Apply the current three-component sigmoid diagnostic without filtering."""
    _require_columns(centers, ["method", "center", "amplitude", "lower", "upper", "width"], "centers")
    s = centers.loc[centers.method.eq("sigmoid")].copy()
    low = s.support_lower if "support_lower" in s else bounds[0]
    high = s.support_upper if "support_upper" in s else bounds[1]
    s["interval_length_cm"] = s.upper - s.lower
    s["amplitude_fail_current"] = ~(s.amplitude >= 100.)
    s["interval_fail_current"] = ~(s.interval_length_cm <= 20.)
    s["position_fail_current"] = ~(np.isfinite(s.center) & (s.center > low) & (s.center < high))
    s["diagnostic_pass_current"] = ~(s.amplitude_fail_current | s.interval_fail_current | s.position_fail_current)
    s["interval_boundary_current"] = (s.lower <= low) | (s.upper >= high)
    s["width_lower_current"] = s.width == 1.
    s["width_upper_current"] = s.width == 18.
    s["width_edge"] = s.width_lower_current | s.width_upper_current
    return s


def summarize_lodo(predictions: pd.DataFrame):
    """Equal depth weights within profile, then equal profile weights.

    ALL pools profiles from all cases; it does not give a short case the same
    total weight as a long case. Pointwise errors are never diagnostic-filtered.
    """
    p = predictions.copy()
    _require_columns(p, ["Case", "profile_id", "method", "depth_cm", "observed", "predicted", "region"], "LODO predictions")
    if set(p.method) != set(METHODS):
        raise ValueError("LODO predictions must contain exactly the three manuscript methods")
    if p.duplicated(["Case", "profile_id", "method", "depth_cm"]).any():
        raise ValueError("Duplicate LODO predictions")
    if not np.isfinite(p[["depth_cm", "observed", "predicted"]]).all().all():
        raise ValueError("Nonfinite LODO observations or predictions")
    p["squared_error"] = (p.predicted - p.observed) ** 2
    p["absolute_error"] = abs(p.predicted - p.observed)
    regions, depths = [], []
    for case, group in list(p.groupby("Case", sort=True)) + [("ALL", p)]:
        for method, h in group.groupby("method", sort=True):
            for region in ["all", "interior", "endpoints"]:
                r = h if region == "all" else h.loc[h.region.eq(region)]
                per_profile = r.groupby(["Case", "profile_id"])[["squared_error", "absolute_error"]].mean()
                regions.append(dict(Case=case, method=method, region=region,
                                    n_profiles=len(per_profile), n_predictions=len(r),
                                    rmse_mv=float(np.sqrt(per_profile.squared_error.mean())),
                                    mae_mv=float(per_profile.absolute_error.mean())))
            for z, r in h.groupby("depth_cm", sort=True):
                depths.append(dict(Case=case, method=method, depth_cm=z,
                                   n_profiles=len(r[["Case", "profile_id"]].drop_duplicates()),
                                   rmse_mv=float(np.sqrt(r.squared_error.mean()))))
    depth = pd.DataFrame(depths)
    pivot = depth.pivot(index=["Case", "depth_cm"], columns="method", values="rmse_mv").reset_index()
    errors = pivot[list(METHODS)]
    tied = errors.sub(errors.min(axis=1), axis=0).abs() <= 1e-10
    pivot["tie_count"] = tied.sum(axis=1)
    pivot["winner"] = errors.idxmin(axis=1).where(pivot.tie_count.eq(1), "tie")
    wins = []
    label = f"{len(pivot[pivot.Case != 'ALL'])}_case_depth_cells"
    for case, g in list(pivot.groupby("Case", sort=True)) + [(label, pivot[pivot.Case != "ALL"])]:
        for method in METHODS:
            wins.append(dict(Case=case, method=method, wins=int(g.winner.eq(method).sum()),
                             n_cells=len(g), n_tied_cells=int(g.tie_count.gt(1).sum()),
                             depths_cm=";".join(f"{v:g}" for v in g.loc[g.winner.eq(method), "depth_cm"])))
    return pd.DataFrame(regions), depth, pivot, pd.DataFrame(wins)


def match_chemical_centers(chemical_centers: pd.DataFrame, sigmoid: pd.DataFrame):
    """Pair only the same case and recorded time (absolute tolerance 1e-9 d)."""
    chem = chemical_centers.rename(columns={"case": "Case", "time_day": "chemical_time_day", "b": "chemical_center_cm"}).copy()
    chem["Case"] = chem.Case.astype(str)
    rows = []
    for record in chem.to_dict("records"):
        candidates = sigmoid.loc[sigmoid.Case.eq(record["Case"])]
        delta = (candidates.Time - record["chemical_time_day"]).abs()
        matches = candidates.loc[np.isclose(candidates.Time, record["chemical_time_day"], rtol=0, atol=1e-9)]
        record.update(nearest_eh_time_day=float(candidates.loc[delta.idxmin(), "Time"]) if len(delta) else np.nan,
                      nearest_time_difference_day=float(delta.min()) if len(delta) else np.nan,
                      exact_time_match_count=len(matches))
        if len(matches) > 1:
            raise ValueError("Multiple Eh profiles match one chemical case/time")
        if len(matches) == 1:
            eh = matches.iloc[0]
            record.update(eh_center_cm=eh.center, eh_profile_id=eh.profile_id,
                          eh_diagnostic_pass_current=bool(eh.diagnostic_pass_current),
                          chemical_minus_eh_cm=record["chemical_center_cm"] - eh.center)
        else:
            record.update(eh_center_cm=np.nan, eh_profile_id=None,
                          eh_diagnostic_pass_current=None, chemical_minus_eh_cm=np.nan)
        record["chemical_two_sided"] = record["n_shallower_observations"] > 0 and record["n_deeper_observations"] > 0
        rows.append(record)
    extra = ["nearest_eh_time_day", "nearest_time_difference_day", "exact_time_match_count",
             "eh_center_cm", "eh_profile_id", "eh_diagnostic_pass_current", "chemical_minus_eh_cm", "chemical_two_sided"]
    matched = pd.DataFrame(rows, columns=[*chem.columns, *extra])
    summaries = []
    for (case, analyte), g in matched.groupby(["Case", "analyte"], sort=True):
        for subset in ["all", "two_sided_only"]:
            h = g if subset == "all" else g.loc[g.chemical_two_sided]
            d = h.chemical_minus_eh_cm
            summaries.append(dict(Case=case, analyte=analyte, subset=subset, n_chemical=len(h),
                                  n_matched=int(d.notna().sum()), difference_min=d.min(),
                                  difference_median=d.median(), difference_max=d.max(),
                                  n_shallower=int((d < 0).sum()), n_equal=int((d == 0).sum()),
                                  n_deeper=int((d > 0).sum())))
    columns = ["Case", "analyte", "subset", "n_chemical", "n_matched", "difference_min", "difference_median",
               "difference_max", "n_shallower", "n_equal", "n_deeper"]
    return matched, pd.DataFrame(summaries, columns=columns)


def summarize_simulations(records: pd.DataFrame):
    """Recompute the four reported metrics directly from all replicate fits."""
    _require_columns(records, ["scenario", "method", "replicate", "true_center", "center", "lower", "upper"], "simulation records")
    catalog = {s["id"]: s for s in scenario_catalog()}
    if set(records.scenario) != set(catalog) or set(records.method) != set(METHODS):
        raise ValueError("Simulation records must contain exactly the four scenarios and three methods")
    if records.duplicated(["scenario", "method", "replicate"]).any():
        raise ValueError("Duplicate simulation scenario/method/replicate")
    output = []
    for scenario, spec in catalog.items():
        expected = None
        for method in METHODS:
            g = records.loc[records.scenario.eq(scenario) & records.method.eq(method)].sort_values("replicate")
            current = g[["replicate", "true_center"]].to_numpy()
            if not len(g) or (expected is not None and not np.array_equal(current, expected)):
                raise ValueError("Methods must use identical simulation replicate IDs and true positions")
            expected = current
            if not np.isfinite(g[["true_center", "center", "lower", "upper"]]).all().all() or (g.upper < g.lower).any():
                raise ValueError("A current-scenario fit or interval is unavailable; do not silently discard the replicate")
            error = g.center - g.true_center
            coverage = float(((g.lower <= g.true_center) & (g.true_center <= g.upper)).mean())
            output.append(dict(scenario=scenario, method=method, n=len(g),
                               rmse=float(np.sqrt(np.mean(error ** 2))), coverage_all=coverage,
                               coverage_percent=100 * coverage,
                               coverage_mcse_pp=100 * np.sqrt(coverage * (1 - coverage) / len(g)),
                               median_interval_length=float((g.upper - g.lower).median()),
                               bias=float(error.mean()), width=spec["width"], sigma=spec["sigma"],
                               seed=int(g.seed.iloc[0]) if "seed" in g else None,
                               interval_kind=str(g.interval_kind.iloc[0]) if "interval_kind" in g else None))
    return pd.DataFrame(output)


def _diagnostic_summary(s):
    rows = []
    for case, g in list(s.groupby("Case", sort=True)) + [("ALL", s)]:
        r = dict(Case=case, n=len(g), finite=int(np.isfinite(g.center).sum()),
                 center_min=g.center.min(), center_median=g.center.median(), center_max=g.center.max(),
                 pass_current=int(g.diagnostic_pass_current.sum()), fail_current=int((~g.diagnostic_pass_current).sum()))
        for key in ["amplitude_fail_current", "interval_fail_current", "position_fail_current", "interval_boundary_current"]:
            r[key] = int(g[key].sum())
        r.update(amplitude_min=g.amplitude.min(), width_1=int(g.width_lower_current.sum()),
                 width_18=int(g.width_upper_current.sum()),
                 width_1_pass=int((g.width_lower_current & g.diagnostic_pass_current).sum()),
                 width_18_pass=int((g.width_upper_current & g.diagnostic_pass_current).sum()),
                 width_edge=int(g.width_edge.sum()), width_edge_pass=int((g.width_edge & g.diagnostic_pass_current).sum()),
                 width_not_edge_pass=int((~g.width_edge & g.diagnostic_pass_current).sum()),
                 width_median=g.width.median(), interval_min=g.interval_length_cm.min(),
                 interval_median=g.interval_length_cm.median(), interval_max=g.interval_length_cm.max(),
                 interval_zero=int(g.interval_length_cm.eq(0).sum()),
                 interval_disconnected=int(g.interval_components.gt(1).sum()))
        rows.append(r)
    return pd.DataFrame(rows)


def mechanism_statistics(predictions, simulation, omission=None):
    """Secondary explanations in sections 3.2 and 3.3, from the same fits."""
    rows = []
    for method in METHODS:
        g = predictions.loc[predictions.method.eq(method)]
        total_sse = np.sum((g.predicted - g.observed) ** 2)
        for depth, h in g.groupby("depth_cm", sort=True):
            errors = h.predicted - h.observed
            rows.append(dict(method=method, depth_cm=depth, n=len(h),
                             rmse_mv=float(np.sqrt(np.mean(errors ** 2))),
                             share_total_sse=float(np.sum(errors ** 2) / total_sse) if total_sse else None,
                             n_overpredicted=int((errors > 0).sum()), n_underpredicted=int((errors < 0).sum())))
    result = {"depth_error_decomposition": rows, "simulation_shape_explanations": []}
    for scenario in ["step_a800_s25", "gompertz_w8_a800_s25"]:
        g = simulation.loc[simulation.scenario.eq(scenario) & simulation.method.eq("sigmoid")]
        error = g.center - g.true_center
        record = dict(scenario=scenario, n=len(g), bias_cm=float(error.mean()),
                      error_sd_cm=float(np.std(error, ddof=0)), error_sd_ddof=0,
                      rmse_cm=float(np.sqrt(np.mean(error ** 2))),
                      coverage=float(((g.lower <= g.true_center) & (g.true_center <= g.upper)).mean()),
                      interval_entirely_shallower=float((g.upper < g.true_center).mean()),
                      interval_entirely_deeper=float((g.lower > g.true_center).mean()),
                      mean_interval_length_cm=float((g.upper - g.lower).mean()))
        if "width" in g:
            record["width1_count"] = int(g.width.eq(1).sum())
        result["simulation_shape_explanations"].append(record)
    if omission is not None:
        result["mean_center_shift_cm"] = _records(omission.groupby(["Case", "method", "depth_cm"]).agg(
            n=("center_shift_cm", "count"), mean_shift_cm=("center_shift_cm", "mean")).reset_index())
        result["pooled_center_shift_cm"] = _records(omission.groupby(["method", "depth_cm"]).agg(
            n=("center_shift_cm", "count"), mean_shift_cm=("center_shift_cm", "mean")).reset_index())
    return result


def summarize_results(real_dir, simulation_dir, chemical_centers, out_dir):
    """Write current numeric tables from newly generated fit and prediction CSVs.

    real_dir contains real_centers.csv, real_lodo_predictions.csv and optionally
    real_omission_stability.csv. simulation_dir contains simulation_profiles.csv.gz
    (or simulation_profiles.csv). chemical_centers is the DataFrame returned by
    fit_chemical_centers. Returns JSON-safe summary tables.
    """
    real_dir, simulation_dir, out_dir = map(Path, (real_dir, simulation_dir, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    centers = pd.read_csv(real_dir / "real_centers.csv", dtype={"Case": str})
    if set(centers.method) != set(METHODS):
        raise ValueError("Center results must contain exactly the three manuscript methods")
    s = diagnose_centers(centers)
    predictions = pd.read_csv(real_dir / "real_lodo_predictions.csv", dtype={"Case": str})
    sim_path = simulation_dir / "simulation_profiles.csv.gz"
    if not sim_path.is_file():
        sim_path = simulation_dir / "simulation_profiles.csv"
    simulation = pd.read_csv(sim_path)
    tables = {"sigmoid_profiles_current": s, "center_diagnostics": _diagnostic_summary(s)}
    tables["all_method_centers"] = centers.groupby(["Case", "method"]).agg(
        n=("center", "size"), finite=("center", "count"), center_min=("center", "min"),
        center_median=("center", "median"), center_max=("center", "max")).reset_index()
    examples = []
    for case, g in s.groupby("Case", sort=True):
        row = g.sort_values("Time").iloc[len(g) // 2]
        chosen = centers.loc[centers.Case.eq(case) & centers.profile_id.eq(row.profile_id)].copy()
        chosen["time_order_one_based"] = len(g) // 2 + 1
        chosen["sigmoid_diagnostic_pass_current"] = bool(row.diagnostic_pass_current)
        examples.append(chosen)
    tables["representative_profiles"] = pd.concat(examples, ignore_index=True)
    lodo_names = ["lodo_regions", "lodo_by_depth", "lodo_depth_winners", "lodo_win_counts"]
    tables.update(zip(lodo_names, summarize_lodo(predictions)))
    tables["simulation_four_scenarios"] = summarize_simulations(simulation)
    tables["chemical_transition_centers"] = chemical_centers
    tables["chemical_eh_exact_matches"], tables["chemical_eh_differences"] = match_chemical_centers(chemical_centers, s)
    for name, frame in tables.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    summaries = {name: _records(frame) for name, frame in tables.items()}
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    omission_path = real_dir / "real_omission_stability.csv"
    omission = pd.read_csv(omission_path, dtype={"Case": str}) if omission_path.is_file() else None
    mechanisms = mechanism_statistics(predictions, simulation, omission)
    (out_dir / "mechanism_statistics.json").write_text(json.dumps(mechanisms, indent=2, allow_nan=False), encoding="utf-8")
    return summaries
