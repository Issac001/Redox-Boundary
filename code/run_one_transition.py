"""Reproduce the manuscript's Eh estimates, chemistry and localization checks."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import platform
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
    chemistry_validation, diagnose, estimate_boundaries, save, write_json,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
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
        summary = []
        for case in cfg["cases"]:
            rows = fits[(fits["Case"] == case) & (fits["model"] == "sigmoid")]
            finite = rows["b"].dropna()
            summary.append({
                "case": case, "n_profiles": len(rows),
                "n_finite_centers": len(finite),
                "center_min_cm": finite.min(),
                "center_median_cm": finite.median(),
                "center_max_cm": finite.max(),
                "n_shape_supported": int(rows["supported_shape"].eq(True).sum()),
                "median_width_scale_cm": rows["w"].median(),
                "n_zero_width_working_interval": int(rows["b_interval_width25"].eq(0).sum()),
            })
        save(out, "case_summary", summary)
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
