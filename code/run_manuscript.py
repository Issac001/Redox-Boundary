"""Recompute the current manuscript, from observations to tables and seven figures.

Default execution is the complete 223-profile, four-scenario experiment.
No archived results, manuscript documents, or machine-specific paths are inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import scipy

from shared.comparison_io import prepare_output_directory
from shared.comparison_models import DEFAULT_CONFIG as MODEL_CONFIG, fit_models
from shared.comparison_simulation import (
    scenario_catalog, generate_scenario, scenario_seed, profile_records, summarize_fit,
)
from shared.manuscript_analysis import fit_chemical_centers, summarize_results
from shared.manuscript_figures_data import prepare_figure_data

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "code/config_manuscript.json"
DEFAULT_DATA = ROOT / "data/manuscript"
METHODS = ("sigmoid", "step", "bayes_step")
SCALARS = ("center", "lower", "upper", "sse", "amplitude", "width",
           "diagnostic_pass", "width_edge", "center_edge", "interval_components",
           "center_map", "center_median", "gap_index", "map_gap_index")


def validate_config(cfg):
    """Keep the manuscript's scientific definitions and figure labels aligned.

    Seed and replicate count may change for independent numerical checks. Other
    scientific settings require a deliberate coordinated revision of the paper.
    """
    if (cfg["model"] != MODEL_CONFIG or cfg["cases"] != ["3.1", "3.2", "3.3"]
            or cfg["expected_counts"] != {"3.1": 77, "3.2": 78, "3.3": 68}
            or cfg["depth"] != [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]):
        raise ValueError("This entry reproduces the manuscript's fixed model, cases and depths; only seed and replicate count may vary")
    for key in ("seed", "n_replicates"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < (1 if key == "n_replicates" else 0):
            raise ValueError(f"{key} must be a valid integer")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(frame, path):
    compression = {"method": "gzip", "mtime": 0} if str(path).endswith(".gz") else None
    frame.to_csv(path, index=False, compression=compression)


def load_observations(path, cfg):
    """Reject incomplete/ambiguous profiles before any analysis is run."""
    data = pd.read_csv(path, dtype={"Case": str, "profile_id": str})
    required = ["Case", "Time", "Depth", "Eh", "WL", "Cycle", "profile_id"]
    if not set(required).issubset(data.columns):
        raise ValueError(f"Eh data require columns: {required}")
    if data[required].isna().any().any():
        raise ValueError("Missing required Eh observation or metadata")
    if set(data.Case) != set(cfg["cases"]):
        raise ValueError("Eh cases do not match the manuscript configuration")
    if data.duplicated(["profile_id", "Depth"]).any():
        raise ValueError("Duplicate profile/depth observations")
    data = data.sort_values(["Case", "Time", "Depth"])
    z = np.asarray(cfg["depth"], dtype=float)
    if not np.isfinite(data[["Time", "Depth", "Eh", "WL", "Cycle"]].to_numpy(float)).all():
        raise ValueError("Non-finite Eh observations or metadata")
    metadata, profiles = [], []
    for pid, part in data.groupby("profile_id", sort=False):
        if not np.array_equal(part.Depth.to_numpy(), z):
            raise ValueError(f"Incomplete or unexpected depths: {pid}")
        if part[["Case", "Time", "WL", "Cycle"]].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Inconsistent within-profile metadata: {pid}")
        profiles.append(part.Eh.to_numpy(float))
        metadata.append(part.iloc[0][["Case", "Time", "profile_id", "WL", "Cycle"]].to_dict())
    meta = pd.DataFrame(metadata)
    if meta.duplicated(["Case", "Time"]).any():
        raise ValueError("Multiple Eh profiles share a case/time")
    if meta.groupby("Case").size().to_dict() != cfg["expected_counts"]:
        raise ValueError("Profile counts do not match the manuscript configuration")
    return data, meta, np.asarray(profiles), z


def scalar_frame(fit, n):
    return pd.DataFrame({key: np.asarray(fit[key]) for key in SCALARS
                         if key in fit and np.asarray(fit[key]).shape == (n,)})


def real_experiment(meta, y, z, model_cfg, out):
    """Common-support fits and all ten held-depth refits for every profile."""
    fits = fit_models(y, z, model_cfg)
    if tuple(fits) != METHODS:
        raise RuntimeError("The manuscript requires exactly three methods")
    records, posterior = [], []
    for method, fit in fits.items():
        frame = pd.concat([meta.reset_index(drop=True), scalar_frame(fit, len(y))], axis=1)
        frame["method"], frame["interval_kind"] = method, fit["interval_kind"]
        records.append(frame)
        if method == "bayes_step":
            for k, (low, high) in enumerate(zip(fit["gap_lower"], fit["gap_upper"])):
                p = meta[["Case", "Time", "profile_id"]].copy()
                p["gap_lower"], p["gap_upper"], p["probability"] = low, high, fit["gap_probability"][:, k]
                posterior.append(p)
    save_csv(pd.concat(records, ignore_index=True), out / "real_centers.csv")
    save_csv(pd.concat(posterior, ignore_index=True), out / "real_gap_posterior.csv")
    predictions, omissions = [], []
    for j, depth in enumerate(z):
        keep = np.arange(len(z)) != j
        fold = fit_models(y[:, keep], z[keep], model_cfg, query_depth=[depth])
        for method, fit in fold.items():
            p = meta[["Case", "Time", "profile_id"]].copy()
            p["method"], p["depth_cm"] = method, depth
            p["observed"], p["predicted"] = y[:, j], fit["prediction"][:, 0]
            p["region"] = "endpoints" if j in (0, len(z) - 1) else "interior"
            predictions.append(p)
            c = p[["Case", "profile_id", "method", "depth_cm"]].copy()
            c["center_omitted"] = fit["center"]
            c["center_full"] = fits[method]["center"]
            c["center_shift_cm"] = c.center_omitted - c.center_full
            omissions.append(c)
    save_csv(pd.concat(predictions, ignore_index=True), out / "real_lodo_predictions.csv")
    save_csv(pd.concat(omissions, ignore_index=True), out / "real_omission_stability.csv")
    curves, points = [], []
    for case in meta.Case.unique():
        indices = np.flatnonzero(meta.Case.to_numpy() == case)
        i = indices[len(indices) // 2]
        query = np.linspace(z.min(), z.max(), 341)
        example = fit_models(y[i:i + 1], z, model_cfg, query_depth=query)
        for method, fit in example.items():
            curves.append(pd.DataFrame(dict(Case=case, profile_id=meta.iloc[i].profile_id,
                                            depth_cm=query, method=method, prediction=fit["prediction"][0])))
        points.append(pd.DataFrame(dict(Case=case, profile_id=meta.iloc[i].profile_id,
                                        depth_cm=z, eh_mv=y[i])))
    save_csv(pd.concat(curves, ignore_index=True), out / "real_example_curves.csv")
    save_csv(pd.concat(points, ignore_index=True), out / "real_example_observations.csv")


def simulation_experiment(cfg, z, out):
    """Four independent truths, the same realization for all three methods."""
    n = cfg["n_replicates"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n_replicates must be a positive integer")
    catalog = scenario_catalog()
    summaries, details, generated = [], [], {}
    for number, spec in enumerate(catalog, 1):
        seed = scenario_seed(cfg["seed"], spec["id"])
        truth = generate_scenario(spec, n, seed, z)
        model_cfg = dict(cfg["model"], sigma=float(spec["sigma"]))
        fits = fit_models(truth["observed"], z, model_cfg)
        for method, fit in fits.items():
            records = profile_records(spec, method, fit, truth, z)
            summary = summarize_fit(spec, method, fit, truth, z, records=records)
            summary.update(seed=seed, working_sigma=model_cfg["sigma"])
            summaries.append(summary)
            records["seed"] = seed
            details.append(records)
        for key in ("observed", "mean", "true_center"):
            generated[spec["id"] + "__" + key] = truth[key]
        print(f"Simulation {number}/{len(catalog)}: {spec['id']} ({n} replicates)", flush=True)
    save_csv(pd.DataFrame(summaries), out / "simulation_summary.csv")
    save_csv(pd.concat(details, ignore_index=True), out / "simulation_profiles.csv.gz")
    np.savez_compressed(out / "simulation_inputs.npz", depth=z, **generated)
    (out / "scenario_catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


def verify_reference(numeric, expected_path):
    """Compare computed tables against the frozen manuscript, never fit to it."""
    expected = json.loads(expected_path.read_text())
    reports = []
    keys = {
        "center_diagnostics": ["Case"],
        "lodo_regions": ["Case", "method", "region"],
        "lodo_by_depth": ["Case", "method", "depth_cm"],
        "simulation_four_scenarios": ["scenario", "method"],
        "chemical_eh_differences": ["Case", "analyte", "subset"],
    }
    for stem, records in expected.items():
        reference = pd.DataFrame(records)
        computed = pd.read_csv(numeric / (stem + ".csv"), dtype={"Case": str})
        columns = list(reference.columns)
        reference = reference.sort_values(keys[stem]).reset_index(drop=True)
        computed = computed[columns].sort_values(keys[stem]).reset_index(drop=True)
        pd.testing.assert_frame_equal(computed, reference, check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-8)
        reports.append({"table": stem, "rows": len(reference), "passed": True})
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "results_manuscript")
    parser.add_argument("--no-figures", action="store_true", help="Compute every analysis and table; omit image rendering")
    parser.add_argument("--font-family", default="Times New Roman", help="Explicit figure-font override if Times New Roman is unavailable")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    provenance_path = DEFAULT_DATA / "provenance.json"
    expected_path = DEFAULT_DATA / "expected_results.json"
    provenance = json.loads(provenance_path.read_text())
    if sha256(expected_path) != provenance["expected_results"]["sha256"]:
        raise ValueError("Frozen manuscript reference checksum mismatch")
    # Refuse an occupied destination before doing calculations or loading observations.
    out = prepare_output_directory(args.output)
    start = time.perf_counter()
    raw, numeric = out / "raw", out / "numeric"
    raw.mkdir(); numeric.mkdir()
    inputs = {name: args.data_dir / name for name in ("eh_observations.csv", "chemical_observations.csv")}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing manuscript input: {path}")
    # Check fonts before spending time on analysis; figures never silently substitute fonts.
    if not args.no_figures:
        from plot_manuscript_figures import render_all, style
        style(args.font_family)
    observations, meta, y, z = load_observations(inputs["eh_observations.csv"], cfg)
    print(f"Eh: {len(meta)} complete profiles; three models; ten leave-depth-out refits", flush=True)
    real_experiment(meta, y, z, cfg["model"], raw)
    simulation_experiment(cfg, z, raw)
    chemical_rows = pd.read_csv(inputs["chemical_observations.csv"], dtype={"Case": str})
    chemical_centers = fit_chemical_centers(chemical_rows)
    save_csv(chemical_centers, raw / "chemical_centers.csv")
    summarize_results(raw, raw, chemical_centers, numeric)
    prepare_figure_data(observations, numeric)
    canonical = sha256(args.config) == provenance["configuration_sha256"]
    canonical = canonical and all(sha256(path) == provenance["input_files"][name]["sha256"]
                                  for name, path in inputs.items())
    checks = verify_reference(numeric, expected_path) if canonical else []
    if not args.no_figures:
        render_all(out, inputs["eh_observations.csv"], font_family=args.font_family)
    manifest = {
        "manuscript_revision": cfg["manuscript_revision"], "configuration": cfg,
        "complete": True, "canonical_manuscript_inputs_and_settings": canonical,
        "methods": list(METHODS), "n_real_profiles": len(meta),
        "n_chemical_centers": len(chemical_centers), "n_scenarios": len(scenario_catalog()),
        "n_replicates": cfg["n_replicates"], "figures_rendered": not args.no_figures,
        "font_family": None if args.no_figures else args.font_family,
        "reference_checks": checks, "elapsed_seconds": time.perf_counter() - start,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__},
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "configuration_sha256": sha256(args.config),
        "reference_sha256": sha256(expected_path), "provenance_sha256": sha256(provenance_path),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "code").rglob("*.py"))},
        "result_sha256": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "reference_tables_verified": len(checks),
                      "elapsed_seconds": manifest["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return manifest


if __name__ == "__main__":
    main()
