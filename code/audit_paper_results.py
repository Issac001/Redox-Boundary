"""Rebuild the manuscript's numerical summaries from one results directory.

This audit reuses the shared loader and estimator. The Case 1 leave-depth-out
replay attributes an existing prediction error; it is not a new experiment.
The output contains no source-data paths, and nonfinite values become JSON null.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from itertools import combinations
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "shared"))
from boundary_model import TransitionGrid  # noqa: E402
from core import load_profiles  # noqa: E402
from run_boundary_study import CENTER_RULE_VERSION, center_diagnostics  # noqa: E402


def _json_safe(value):
    """Convert numpy scalars and missing/nonfinite values without stringifying."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _boolean(values):
    """Read CSV booleans without treating the string 'False' as true."""
    if values.dtype == bool:
        return values
    parsed = values.astype(str).str.lower().map({"true": True, "false": False})
    if parsed.isna().any():
        raise ValueError(f"Missing or invalid boolean values in {values.name}")
    return parsed.astype(bool)


def _chemical_metrics(metrics, keys):
    table = metrics.pivot(index=keys, columns="model", values="rmse_log1p")
    summary = table[["depth_water", "plus_Eh", "boundary_sigmoid"]].copy()
    for name in ("plus_Eh", "boundary_sigmoid"):
        summary[name + "_relative_change_percent"] = 100 * (
            summary[name] / summary.depth_water - 1)
    return summary.reset_index().to_dict("records")


def _case_one_lodo(data, saved):
    """Replay the unchanged estimator to split Case 1 error by held-out depth."""
    ix = data.meta.index[data.meta.Case.eq("1")]
    if len(ix) == 0:
        return None
    y, z = data.eh[ix], data.depth
    squared, details = [], []
    for j, depth in enumerate(z):
        keep = np.arange(len(z)) != j
        model = TransitionGrid(z[keep])
        fit = model.fit(y[:, keep])
        error = (model.predict(fit, [depth])[:, 0] - y[:, j]) ** 2
        squared.append(error)
        details.append({"heldout_depth_cm": float(depth),
                        "rmse_mV": float(np.sqrt(error.mean())),
                        "sum_squared_error": float(error.sum())})
    total = float(np.sum(squared))
    for row in details:
        row["fraction_of_total_squared_error"] = (
            row["sum_squared_error"] / total if total else None)
    recalculated = float(np.sqrt(np.mean(squared)))
    match = saved.loc[saved.case.eq("1") & saved.model.eq("sigmoid"), "rmse_mV"]
    if len(match) != 1:
        raise ValueError("Expected exactly one saved Case 1 sigmoid LODO result")
    expected = float(match.iloc[0])
    if not np.isclose(recalculated, expected, rtol=1e-12):
        raise ValueError("Case 1 LODO replay does not match the supplied results")
    return {"procedure": "Replay existing estimator with unchanged grids",
            "saved_rmse_mV": expected, "recalculated_rmse_mV": recalculated,
            "depth_attribution": details}


def _shape_rules(fits, *, current_center=False):
    """Audit either legacy joint rules or current center-only edge criteria."""
    edge_name = "position_grid_edge" if current_center else "grid_edge"
    failures = pd.DataFrame({
        "amplitude_below_100": ~fits.amplitude.ge(100),
        "aicc_not_better": ~fits.delta_aicc_vs_no_transition.lt(0),
        "interval_wider_than_20": ~fits.b_interval_width25.le(20),
        edge_name: _boolean(fits.b_grid_edge if current_center else fits.grid_edge),
    })
    failed_count = failures.sum(axis=1)
    all_pass = failed_count.eq(0)
    marker = "supported_center" if current_center else "supported_shape"
    if marker not in fits:
        raise ValueError(f"Missing {marker}; first run run_one_transition.py --reclassify-from")
    if not np.array_equal(all_pass.to_numpy(), _boolean(fits[marker]).to_numpy()):
        raise ValueError(f"Recomputed rules differ from saved {marker} markers")
    patterns = failures.apply(
        lambda row: ",".join(row.index[row]) or "all_pass", axis=1)
    summary = {
        "n": len(fits), "n_all_pass": int(all_pass.sum()),
        "n_any_failed": int((~all_pass).sum()),
        "individual_failure_counts": failures.sum().astype(int).to_dict(),
        "fail_only_counts": {name: int((failures[name] & failed_count.eq(1)).sum())
                             for name in failures},
        "joint_patterns": patterns.value_counts().sort_index().to_dict(),
        "pairwise_intersection_counts": {
            f"{left},{right}": int((failures[left] & failures[right]).sum())
            for left, right in combinations(failures.columns, 2)},
        "n_position_grid_edge": int(_boolean(fits.b_grid_edge).sum()),
        "n_width_grid_edge": int(_boolean(fits.width_grid_edge).sum()),
        "n_width_at_lower_grid_bound": int(fits.w.eq(1).sum()),
    }
    if current_center:
        flags = center_diagnostics(fits)
        summary.update({
            "rule_version": CENTER_RULE_VERSION,
            "n_legacy_joint_pass": int(flags.supported_shape.sum()),
            "n_new_pass_from_width_separation": int((all_pass & ~flags.supported_shape).sum()),
            "n_pass_with_width_grid_edge": int((all_pass & _boolean(fits.width_grid_edge)).sum()),
            "interpretation": "Operational diagnostic counts, not calibrated localization accuracy",
        })
    return summary


def audit(data, config, results, out):
    """Write and return manuscript statistics for loaded Profiles and saved CSVs.

    ``results`` is a directory; ``out`` is the output JSON filename. Numerical
    simulation summaries are included when present, so observed-only runs work.
    This function never changes the observations, configuration, or input CSVs.
    """
    results, out = Path(results), Path(out)
    hashes = {}

    def read(name):
        path = results / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path, dtype={"case": str, "Case": str})

    metrics = read("B3_chemistry_metrics.csv")
    metrics["include_95cm"] = _boolean(metrics.include_95cm)
    primary = metrics[metrics.role.eq("primary")]
    summary = _chemical_metrics(primary[~primary.include_95cm], ["case", "fold", "analyte"])
    fe_sensitivity = _chemical_metrics(
        primary[primary.case.eq("3.3") & primary.analyte.eq("Fe")],
        ["case", "fold", "analyte", "include_95cm"])

    by_time = read("B3_chemistry_by_time.csv")
    by_time = by_time[by_time.case.eq("3.3") & by_time.role.eq("primary")
                      & ~_boolean(by_time.include_95cm)]
    temporal = by_time.pivot(index=["case", "fold", "analyte", "time_day"],
                             columns="model", values="mse_log1p")
    temporal = temporal[["depth_water", "boundary_sigmoid"]].copy()
    temporal["relative_rmse_change_percent"] = 100 * (
        np.sqrt(temporal.boundary_sigmoid / temporal.depth_water) - 1)
    time_rows = temporal.reset_index()
    counts = time_rows.groupby("analyte").relative_rmse_change_percent.agg(
        n_test_times="size", n_improved=lambda values: int((values < 0).sum()))

    centers = read("B3_chemical_transition_centers.csv")
    centers = centers[centers.case.eq("3.3")].copy()
    centers["grid_edge"] = _boolean(centers.grid_edge)
    descriptive = centers.groupby("analyte").agg(
        n_times=("b", "size"), n_finite_centers=("b", "count"),
        center_min=("b", "min"), center_median=("b", "median"), center_max=("b", "max"),
        n_grid_edge=("grid_edge", "sum"),
        n_depths_min=("n_depths", "min"), n_depths_max=("n_depths", "max"))
    shared = read("B3_shared_transition_diagnostic.csv")
    shared = shared[shared.case.eq("3.3")]
    fits = read("B2_boundary_estimates.csv")
    fits = fits[fits.model.eq("sigmoid")]

    coverage = []
    main_rows = data.rows[data.rows.Depth.le(90)]
    saved_coverage = read("B1_chemistry_audit.csv")
    saved_coverage = saved_coverage[saved_coverage.depth_cm.le(90)]
    for case, part in main_rows.groupby("Case"):
        for analyte in ("NO3", "NH4", "Fe", "Mn"):
            at_time = part.groupby("Time")[analyte].count()
            nonempty = at_time[at_time > 0]
            n_rows = int(part[analyte].notna().sum())
            existing = saved_coverage[saved_coverage.case.eq(case)
                                      & saved_coverage.analyte.eq(analyte)]
            if n_rows != int(existing.n.sum()):
                raise ValueError(f"Chemical coverage differs for Case {case}, {analyte}")
            coverage.append({
                "case": case, "analyte": analyte, "n_nonempty_times": len(nonempty),
                "n_nonmissing_rows": n_rows, "n_zero_rows": int(part[analyte].eq(0).sum()),
                "min_depths_per_nonempty_time": int(nonempty.min()) if len(nonempty) else None,
                "max_depths_per_nonempty_time": int(nonempty.max()) if len(nonempty) else None,
            })

    sensitivity = read("B2_interval_sensitivity.csv")
    interval_summary = []
    for (case, sigma), part in sensitivity.groupby(["case", "sigma_mV"]):
        interval_summary.append({
            "case": case, "working_sigma_mV": float(sigma), "n_profiles": len(part),
            "n_available": int(part.working_width_cm.notna().sum()),
            "median_width_cm": float(part.working_width_cm.median()),
            "n_zero_width": int(part.working_width_cm.eq(0).sum()),
            "n_disconnected": int(part.support_components.gt(1).sum()),
            "n_grid_truncated": int(_boolean(part.grid_truncated).sum()),
        })
    equality = {}
    for threshold in config["thresholds_mV"]:
        equal = data.eh == threshold
        equality[threshold] = {"exact_nodes": int(equal.sum()),
                               "adjacent_flat_pairs": int((equal[:, :-1] & equal[:, 1:]).sum())}
    lodo = read("B2_leave_depth_out.csv")
    report = {
        "scope": "Manuscript numerical summaries and replay of existing Case 1 depth validation",
        "runtime": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "matplotlib")}},
        "raw_loader_audit": data.audit,
        "case_time_ranges": data.meta.groupby("Case").Time.agg(["min", "max"]).reset_index().to_dict("records"),
        "threshold_exact_equalities": equality,
        "threshold_crossing_counts": read("B1_threshold_audit.csv").to_dict("records"),
        "case_summary": read("case_summary.csv").to_dict("records"),
        "case33_shape_rules": _shape_rules(fits[fits.Case.eq("3.3")]),
        "center_rule_version": CENTER_RULE_VERSION,
        "center_rules_by_case": {case: _shape_rules(part, current_center=True)
                                 for case, part in fits.groupby("Case")},
        "n_width_at_lower_grid_bound_by_case": fits.groupby("Case").w.apply(lambda values: int(values.eq(1).sum())).to_dict(),
        "median_fitted_profile_rmse_mV_by_case": fits.groupby("Case").fit_rmse_mV.median().to_dict(),
        "chemical_coverage_main_depths": coverage,
        "working_interval_sigma_sensitivity": interval_summary,
        "leave_depth_out_metrics": lodo.to_dict("records"),
        "chemical_primary_metrics": summary,
        "fe_depth_sensitivity": fe_sensitivity,
        "chemical_primary_folds": read("B3_fold_audit.csv").query("role == 'primary'").to_dict("records"),
        "case33_chemical_by_test_time": time_rows.to_dict("records"),
        "case33_chemical_improved_time_counts": counts.reset_index().to_dict("records"),
        "case33_descriptive_chemical_centers": descriptive.reset_index().to_dict("records"),
        "case33_chemical_center_totals": {"n_fits": len(centers),
                                          "n_grid_edge": int(centers.grid_edge.sum())},
        "case33_shared_centers": {"n_times": len(shared), "center_min_cm": shared.common_b.min(),
                                  "center_max_cm": shared.common_b.max()},
        "case1_lodo_replay": _case_one_lodo(data, lodo),
    }
    for filename, key in (("localization_summary.csv", "localization_summary"),
                          ("grid_resolution_diagnostic.csv", "grid_resolution_diagnostic")):
        report[key] = read(filename).to_dict("records") if (results / filename).exists() else None
    for name, digest in hashes.items():
        if hashlib.sha256((results / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Result changed during audit: {name}")
    report["input_csv_sha256"] = hashes
    report = _json_safe(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    return report


def check_reference(report, reference_path, keys=None):
    """Compare report fields with a reference JSON's ``metrics`` subset.

    Dictionaries permit additional report keys; lists preserve length/order.
    Numbers use rtol=atol=1e-8, while booleans, strings and null match strictly.
    ``keys`` can select a subset, for example the two simulation-only outputs.
    """
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    expected = reference.get("metrics")
    if not isinstance(expected, dict):
        raise ValueError("Reference must contain a metrics object")
    selected = list(expected) if keys is None else list(keys)
    checked_leaves = 0

    def compare(actual, target, path):
        nonlocal checked_leaves
        if isinstance(target, dict):
            if not isinstance(actual, dict):
                raise ValueError(f"Reference mismatch at {path}: expected an object")
            for key, value in target.items():
                if key not in actual:
                    raise ValueError(f"Reference mismatch at {path}.{key}: missing field")
                compare(actual[key], value, f"{path}.{key}")
            return
        if isinstance(target, list):
            if not isinstance(actual, list) or len(actual) != len(target):
                raise ValueError(f"Reference mismatch at {path}: expected a list of length {len(target)}")
            for index, (value, expected_value) in enumerate(zip(actual, target)):
                compare(value, expected_value, f"{path}[{index}]")
            return
        if isinstance(target, Real) and not isinstance(target, bool):
            matches = (isinstance(actual, Real) and not isinstance(actual, bool)
                       and bool(np.isclose(actual, target, rtol=1e-8, atol=1e-8)))
        else:
            matches = type(actual) is type(target) and actual == target
        if not matches:
            raise ValueError(f"Reference mismatch at {path}: expected {target!r}, got {actual!r}")
        checked_leaves += 1

    for key in selected:
        if key not in expected:
            raise ValueError(f"Reference metrics.{key}: no expected field")
        if key not in report:
            raise ValueError(f"Reference mismatch at metrics.{key}: missing report field")
        compare(report[key], expected[key], f"metrics.{key}")
    return {"matched": True, "checked_keys": selected, "checked_leaves": checked_leaves}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        help="Default: config data path, relative to the repository root")
    parser.add_argument("--config", type=Path, default=HERE / "config_one_transition.json")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, help="Default: <results>/paper_audit.json")
    parser.add_argument("--check-reference", type=Path,
                        help="Compare the audit with the metrics in this reference JSON")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_path = args.data if args.data is not None else ROOT / config["data"]
    data = load_profiles(data_path, config["cases"])
    output = args.output if args.output is not None else args.results / "paper_audit.json"
    report = audit(data, config, args.results, output)
    print(f"Manuscript numerical audit saved to {output}")
    if args.check_reference is not None:
        checked = check_reference(report, args.check_reference)
        print(f"Reference verified: {checked['checked_leaves']} values in {len(checked['checked_keys'])} fields")


if __name__ == "__main__":
    main()
