"""Reproduce the manuscript's Eh estimates, chemistry and localization checks."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SHARED = HERE / "shared"
sys.path.insert(0, str(SHARED))

from core import load_profiles  # noqa: E402
from boundary_simulation import run as validate_localization  # noqa: E402
from run_boundary_study import (  # noqa: E402
    CENTER_RULE_VERSION, center_diagnostics, chemistry_validation, diagnose,
    estimate_boundaries, save, write_json,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_cases(fits, cases):
    """Report center criteria and width flags separately, retaining old counts."""
    summary = []
    for case in cases:
        rows = fits[fits.Case.eq(case) & fits.model.eq("sigmoid")]
        flags = center_diagnostics(rows)
        finite = rows.b.dropna()
        width_edge = rows.width_grid_edge.astype(str).str.lower().eq("true")
        summary.append({
            "case": case, "n_profiles": len(rows), "n_finite_centers": len(finite),
            "center_min_cm": finite.min(), "center_median_cm": finite.median(),
            "center_max_cm": finite.max(),
            "n_shape_supported": int(flags.supported_shape.sum()),  # legacy rule
            "n_center_supported": int(flags.supported_center.sum()),
            "n_width_grid_edge": int(width_edge.sum()),
            "n_center_supported_width_grid_edge": int((flags.supported_center & width_edge).sum()),
            "median_width_scale_cm": rows.w.median(),
            "n_zero_width_working_interval": int(rows.b_interval_width25.eq(0).sum()),
        })
    return summary


def reclassify_saved_results(source, out, cfg, config_path):
    """Copy saved results to a new directory and revise only diagnostic columns."""
    source, out = source.resolve(), out.resolve()
    if source == out or (out.exists() and any(out.iterdir())):
        raise ValueError("Diagnostic reclassification requires a separate, empty output directory")
    old_manifest_path = source / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    if old_manifest["configuration"] != cfg:
        raise ValueError("Source results use another configuration; supply their --config")
    fits = pd.read_csv(source / "B2_boundary_estimates.csv", dtype={"Case": str},
                       float_precision="round_trip")
    original_fits = fits.copy()
    rows = fits.model.eq("sigmoid")
    flags = center_diagnostics(fits.loc[rows])
    legacy = fits.loc[rows, "supported_shape"].astype(str).str.lower().eq("true")
    if not legacy.equals(flags.supported_shape):
        raise ValueError("Source legacy shape markers disagree with the unchanged rule")
    for name, values in flags.items():
        fits.loc[rows, name] = values
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name: sha256(p) for p in sorted(source.glob("*.csv"))}
    for name in source_hashes:
        shutil.copy2(source / name, out / name)
    shutil.copy2(old_manifest_path, out / "source_manifest.json")
    save(out, "B2_boundary_estimates", fits)
    revised = pd.read_csv(out / "B2_boundary_estimates.csv", dtype={"Case": str},
                          float_precision="round_trip")
    pd.testing.assert_frame_equal(original_fits, revised[original_fits.columns], check_exact=True)
    save(out, "case_summary", summarize_cases(fits, cfg["cases"]))
    # The source audit is not copied: it describes the previous CSV hashes.
    write_json(out / "manifest.json", {
        "scope": "Saved-result diagnostic reclassification; no fitting, chemistry or simulation rerun",
        "mode": "diagnostic_revision_only", "center_rule_version": CENTER_RULE_VERSION,
        "configuration": cfg, "configuration_sha256": sha256(config_path),
        "input_workbook_sha256": old_manifest.get("input_workbook_sha256"),
        "n_profiles": int(rows.sum()),
        "source_manifest_sha256": sha256(old_manifest_path),
        "source_result_sha256": source_hashes,
        "source_sha256": {str(p.relative_to(PACKAGE)): sha256(p)
                          for p in sorted(HERE.rglob("*.py"))},
        "result_sha256": {p.name: sha256(p) for p in sorted(out.glob("*.csv"))},
        "unchanged_numeric_estimates": True,
        "reference_verification": None,
    })


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_one_transition.json")
    parser.add_argument("--output", type=Path, default=PACKAGE / "results")
    parser.add_argument("--data", type=Path, help="Path to the collaborator's original workbook")
    parser.add_argument("--check-reference", type=Path,
                        help="Verify reproduced statistics against a frozen paper_metrics.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--simulation-only", action="store_true",
                       help="Reproduce numerical experiments without the private workbook")
    modes.add_argument("--skip-numerical-validation", action="store_true",
                       help="Run the observed-data analyses and paper audit only")
    modes.add_argument("--reclassify-from", type=Path, metavar="RESULTS",
                       help="Revise saved diagnostic flags into a new --output directory without refitting")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if args.reclassify_from:
        if args.check_reference:
            parser.error("Run audit_paper_results.py --check-reference after reclassification")
        reclassify_saved_results(args.reclassify_from, args.output, cfg, config_path)
        print(f"Center/width diagnostics revised without refitting: {args.output.resolve()}")
        return
    data_path = args.data.resolve() if args.data else (PACKAGE / cfg["data"]).resolve()
    if not args.simulation_only and not data_path.is_file():
        parser.error("Workbook missing: use --data PATH or see data/README.md. "
                     "--simulation-only does not require the workbook.")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.simulation_only and (out / "B2_boundary_estimates.csv").exists():
        parser.error("Use a separate output directory for simulation-only results.")
    if args.skip_numerical_validation and (out / "localization_summary.csv").exists():
        parser.error("Use a separate output directory when skipping numerical experiments.")

    numerical = None
    data = None
    if not args.simulation_only:
        data = load_profiles(data_path, cfg["cases"])
        diagnose(data, cfg, out)
        fitted, fits = estimate_boundaries(data, cfg, out)
        print(f"Eh estimation and depth validation: {len(data.meta)} profiles", flush=True)
        # Restrict the caller to the manuscript's chemical cases and features.
        # Keep full metadata: its row indices address the original Eh fit arrays.
        chemical_cases = list(dict.fromkeys(fold["case"] for fold in cfg["folds"]))
        chemical_data = replace(data, rows=data.rows[data.rows.Case.isin(chemical_cases)])
        chemistry_validation(
            chemical_data, {"sigmoid": fitted["sigmoid"]},
            {**cfg, "cases": chemical_cases}, out,
        )
        save(out, "case_summary", summarize_cases(fits, cfg["cases"]))
        print("Chemical prediction and chemical-position analyses complete", flush=True)

    if not args.skip_numerical_validation:
        numerical = validate_localization(cfg.get("numerical_validation", {}), out)
        print("Six localization scenarios and paired grid checks complete", flush=True)
    report = None
    if data is not None:
        from audit_paper_results import audit
        report = audit(data, cfg, out, out / "paper_audit.json")
        print("Supplementary manuscript statistics and depth-error attribution complete", flush=True)
    verification = None
    if args.check_reference:
        from audit_paper_results import check_reference
        reference = json.loads(args.check_reference.read_text(encoding="utf-8"))
        if data is not None and sha256(data_path) != reference["input_workbook_sha256"]:
            raise ValueError("The workbook hash differs from the manuscript reference.")
        if report is None:
            report = {name: json.loads(pd.read_csv(out / f"{name}.csv").to_json(
                orient="records", double_precision=15))
                for name in ("localization_summary", "grid_resolution_diagnostic")}
        keys = list(reference["metrics"])
        if args.simulation_only:
            keys = ["localization_summary", "grid_resolution_diagnostic"]
        elif args.skip_numerical_validation:
            keys = [key for key in keys if key not in
                    ("localization_summary", "grid_resolution_diagnostic")]
        verification = check_reference(report, args.check_reference, keys)
        print(f"Paper reference verified: {verification['checked_leaves']} values", flush=True)

    sources = sorted(HERE.rglob("*.py"))
    manifest = {
        "scope": "Current manuscript only: Eh, chemical prediction/positions, localization/coverage",
        "center_rule_version": CENTER_RULE_VERSION,
        "mode": "simulation_only" if args.simulation_only else "observed_only" if args.skip_numerical_validation else "full",
        "runtime": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "matplotlib"]}},
        "configuration": cfg, "configuration_sha256": sha256(config_path),
        "input_workbook_sha256": sha256(data_path) if data is not None else None,
        "n_profiles": len(data.meta) if data is not None else None,
        "source_sha256": {str(p.relative_to(PACKAGE)): sha256(p) for p in sources},
        "result_sha256": {p.name: sha256(p) for p in sorted(out.glob("*.csv"))},
        "numerical_validation_performed": numerical is not None,
        "reference_verification": verification,
    }
    write_json(out / "manifest.json", manifest)
    print(f"Reproduction complete: {out}")


if __name__ == "__main__":
    main()
