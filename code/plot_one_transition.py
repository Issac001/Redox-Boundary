"""Plot one Eh transition, B2 precision diagnostics and separate B3 chemical checks."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "redox_boundary_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "redox_boundary_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(HERE / "shared"))
from core import load_profiles  # noqa: E402
from plot_boundary_study import (  # noqa: E402
    COLORS, CHEMICAL_STYLES, boolean, center_flags, fixed_threshold_diagnostic,
    grid_resolution_diagnostic, localization_validation,
    observed_boundaries, save_figure, setup,
)


def main(argv=None, *, font_language="auto", font=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        help="Input workbook; default: config data path, relative to repository root")
    parser.add_argument("--config", type=Path, default=HERE / "config_one_transition.json")
    parser.add_argument("--results", type=Path, default=PACKAGE / "results")
    parser.add_argument("--out", type=Path, default=PACKAGE / "figures")
    parser.add_argument("--skip-numerical-validation", action="store_true",
                        help="Plot observations and B3 only, even if simulation CSVs exist")
    args = parser.parse_args(argv)
    setup(language=font_language, font=font)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    data_path = args.data if args.data is not None else PACKAGE / Path(cfg.get("data", "data/Whole3.1.xlsx"))
    data = load_profiles(data_path, cfg["cases"])
    fits = pd.read_csv(args.results / "B2_boundary_estimates.csv", dtype={"Case": str})
    case = cfg["primary_case"]
    one = fits[(fits.Case == case) & (fits.model == "sigmoid")].sort_values("Time")
    representative = one[one.profile_id.eq("3.3_10558")]
    sample = representative.iloc[0] if len(representative) else one.iloc[len(one) // 2]
    index = data.meta.index[data.meta.profile_id.eq(sample.profile_id)][0]
    depth, eh = data.depth, data.eh[index]
    b, width, amplitude = float(sample.b), float(sample.w), float(sample.amplitude)
    level = float(eh.mean() - amplitude * expit((b - depth) / width).mean())
    grid = np.linspace(depth.min(), depth.max(), 401)
    curve = level + amplitude * expit((b - grid) / width)
    reconstructed_sse = float(np.sum((eh - level - amplitude * expit((b - depth) / width)) ** 2))
    if not np.isclose(reconstructed_sse, sample.sse, rtol=1e-8, atol=1e-6):
        raise ValueError("Displayed curve does not reproduce the saved fit")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.8), layout="constrained")
    ax = axes[0]
    ax.axhspan(sample.b_low_working25, sample.b_high_working25,
               color=COLORS["interval"], alpha=.55)
    ax.plot(curve, grid, color=COLORS["center"], linewidth=1.3, label="One-transition fit")
    ax.scatter(eh, depth, color=COLORS["observed"], s=16, zorder=3, label="Observed Eh")
    ax.axhline(b, color=COLORS["center"], linestyle="--", linewidth=.8)
    ax.set(xlabel="Eh (mV)", ylabel="Depth below column top (cm)", ylim=(98, 7),
           title=f"Observed profile at day {sample.Time:.2f}: center = {b:.0f} cm")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    supported, width_edge = center_flags(one)
    ax.fill_between(one.Time.to_numpy(), one.b_low_working25.to_numpy(),
                    one.b_high_working25.to_numpy(), color=COLORS["interval"], alpha=.55)
    ax.plot(one.Time, one.b, color=COLORS["center"], linewidth=.85)
    ax.scatter(one.loc[~supported, "Time"], one.loc[~supported, "b"],
               color=COLORS["warning"], marker="x", linewidths=.75, s=17,
               label="Fails center rule", zorder=5)
    ax.scatter(one.loc[supported, "Time"], one.loc[supported, "b"],
               color=COLORS["center"], s=14, label="Passes center rule", zorder=3)
    ax.scatter(one.loc[width_edge, "Time"], one.loc[width_edge, "b"],
               facecolors="white", edgecolors=COLORS["center"], marker="s", linewidths=.65,
               s=17, label="Width at search limit", zorder=4)
    ax.axhline(one.b.median(), color=COLORS["step"], linestyle=":", linewidth=.8,
               label=f"Median = {one.b.median():.0f} cm")
    ax.set(xlabel=f"Time within Case {case} (days)", ylabel="Center below column top (cm)",
           title=f"Independent fits: {int(supported.sum())}/{len(one)} pass center rule")
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=.12, linewidth=.5)
    ax.legend(frameon=False, loc="lower right")

    fig._paper_info = {"case": case, "day": float(sample.Time), "center": b,
                       "supported": int(supported.sum()), "n": len(one),
                       "median": float(one.b.median())}
    args.out.mkdir(parents=True, exist_ok=True)
    paths = save_figure(fig, args.out, "one_Eh_transition")

    metrics = pd.read_csv(args.results / "B3_chemistry_metrics.csv", dtype={"case": str})
    folds = [fold["fold"] for fold in cfg["folds"]
             if fold["case"] == case and fold["role"] == "primary"]
    metrics = metrics[(metrics.case == case) & ~boolean(metrics.include_95cm)
                      & metrics.fold.isin(folds)]
    fig, axes = plt.subplots(1, len(folds), figsize=(3.4 * len(folds), 3.0),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    axes = axes.ravel()
    markers = ["NO3", "NH4", "Fe", "Mn"]
    for ax, fold in zip(axes, folds):
        part = metrics[metrics.fold.eq(fold)]
        table = part.pivot(index="analyte", columns="model", values="rmse_log1p")
        y = np.arange(len(markers))
        base = table.loc[markers, "depth_water"]
        for offset, model, label, color in [
            (-.12, "plus_Eh", "Add current Eh", COLORS["baseline"]),
            (.12, "boundary_sigmoid", "Add transition features", COLORS["center"]),
        ]:
            percent = 100 * (table.loc[markers, model] / base - 1)
            ax.scatter(percent, y + offset, s=23, color=color,
                       marker="s" if model == "plus_Eh" else "o", label=label, zorder=3)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(yticks=y, yticklabels=["NO$_3$", "NH$_4$", "Fe", "Mn"], xlabel="Change in log1p-RMSE vs depth + water (%)",
               title=f"Case {case} / {fold}: {int(part.n_times.iloc[0])} chemical times")
        ax.grid(axis="x", alpha=.12, linewidth=.5)
    axes[0].invert_yaxis()  # Invert the shared axis once, rather than once per panel.
    axes[-1].legend(frameon=False, loc="lower right")
    paths.extend(save_figure(fig, args.out, "chemical_prediction"))

    centers = pd.read_csv(args.results / "B3_chemical_transition_centers.csv", dtype={"case": str})
    centers = centers[centers.case.eq(case)]
    shared = pd.read_csv(args.results / "B3_shared_transition_diagnostic.csv", dtype={"case": str})
    shared = shared[shared.case.eq(case)].sort_values("time_day")
    fig, ax = plt.subplots(figsize=(6.8, 3.7), layout="constrained")
    ax.plot(one.Time, one.b, color=COLORS["center"], linewidth=1.15, label="Eh center (B2)")
    for marker, color, shape in CHEMICAL_STYLES:
        part = centers[centers.analyte.eq(marker)]
        edge = boolean(part.grid_edge)
        ax.scatter(part.time_day, part.b, s=24, color=color, marker=shape,
                   label=f"{marker} descriptive center", zorder=3)
        ax.scatter(part.loc[edge, "time_day"], part.loc[edge, "b"], s=30,
                   marker="x", color=COLORS["observed"], linewidths=.6, zorder=4)
    ax.scatter([], [], marker="x", color="#222222", label="Position/width at search-grid edge")
    ax.plot(shared.time_day, shared.common_b, color=COLORS["shared"], linestyle="--",
            linewidth=1.0, label="Shared-fit compromise")
    ax.set(xlabel=f"Time within Case {case} (days)", ylabel="Center below column top (cm)",
           title="Chemical transition locations: descriptive comparison")
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=.12, linewidth=.5)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center")
    paths.extend(save_figure(fig, args.out, "chemical_centers_descriptive"))
    paths.extend(observed_boundaries(data, args.out, fits))
    paths.extend(fixed_threshold_diagnostic(data, cfg, args.results, args.out))
    if not args.skip_numerical_validation:
        paths.extend(localization_validation(args.results, args.out))
        paths.extend(grid_resolution_diagnostic(args.results, args.out))
    print(f"Saved {len(paths)} figure assets in {args.out}")


if __name__ == "__main__":
    main()
