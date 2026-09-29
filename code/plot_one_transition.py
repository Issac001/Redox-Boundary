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
    boolean, grid_resolution_diagnostic, localization_validation,
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

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), layout="constrained")
    ax = axes[0]
    ax.plot(grid, curve, color="#b20b2d", linewidth=2.5, label="One-transition fit")
    ax.scatter(depth, eh, color="#1b3553", s=38, zorder=3, label="Observed Eh")
    ax.axvline(b, color="#b20b2d", linestyle="--", alpha=.65)
    ax.set(xlabel="Depth below column top (cm)", ylabel="Eh (mV)",
           title=f"Observed profile at day {sample.Time:.2f}: center = {b:.0f} cm")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=.18)

    ax = axes[1]
    supported = boolean(one.supported_shape)
    ax.plot(one.Time, one.b, color="#8e9aa8", linewidth=1, alpha=.85)
    ax.scatter(one.loc[~supported, "Time"], one.loc[~supported, "b"],
               facecolors="white", edgecolors="#7d8996", s=32, label="Fails shape rule", zorder=3)
    ax.scatter(one.loc[supported, "Time"], one.loc[supported, "b"],
               color="#b20b2d", s=38, label="Passes shape rule", zorder=4)
    ax.axhline(one.b.median(), color="#b20b2d", linestyle=":", alpha=.7,
               label=f"Median = {one.b.median():.0f} cm")
    ax.set(xlabel=f"Time within Case {case} (days)", ylabel="Center below column top (cm)",
           title=f"Independent fits: {int(supported.sum())}/{len(one)} pass shape rule")
    ax.invert_yaxis()
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle(f"One statistical Eh transition in Case {case}", fontsize=15, fontweight="bold")
    args.out.mkdir(parents=True, exist_ok=True)
    paths = save_figure(fig, args.out, "one_Eh_transition")

    metrics = pd.read_csv(args.results / "B3_chemistry_metrics.csv", dtype={"case": str})
    folds = [fold["fold"] for fold in cfg["folds"]
             if fold["case"] == case and fold["role"] == "primary"]
    metrics = metrics[(metrics.case == case) & ~boolean(metrics.include_95cm)
                      & metrics.fold.isin(folds)]
    fig, axes = plt.subplots(1, len(folds), figsize=(5.7 * len(folds), 4.2),
                             sharey=True, squeeze=False, layout="constrained")
    axes = axes.ravel()
    markers = ["NO3", "NH4", "Fe", "Mn"]
    for ax, fold in zip(axes, folds):
        part = metrics[metrics.fold.eq(fold)]
        table = part.pivot(index="analyte", columns="model", values="rmse_log1p")
        y = np.arange(len(markers))
        base = table.loc[markers, "depth_water"]
        for offset, model, label, color in [
            (-.12, "plus_Eh", "Add current Eh", "#345e93"),
            (.12, "boundary_sigmoid", "Add transition features", "#b20b2d"),
        ]:
            percent = 100 * (table.loc[markers, model] / base - 1)
            ax.scatter(percent, y + offset, s=53, color=color, label=label, zorder=3)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(yticks=y, yticklabels=markers, xlabel="Change in log1p-RMSE vs depth + water (%)",
               title=f"Case {case} / {fold}: {int(part.n_times.iloc[0])} chemical times")
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()  # Invert the shared axis once, rather than once per panel.
    axes[-1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Forward-time chemical prediction check", fontsize=15, fontweight="bold")
    paths.extend(save_figure(fig, args.out, "chemical_prediction"))

    centers = pd.read_csv(args.results / "B3_chemical_transition_centers.csv", dtype={"case": str})
    centers = centers[centers.case.eq(case)]
    shared = pd.read_csv(args.results / "B3_shared_transition_diagnostic.csv", dtype={"case": str})
    shared = shared[shared.case.eq(case)].sort_values("time_day")
    fig, ax = plt.subplots(figsize=(9.8, 4.5), layout="constrained")
    ax.plot(one.Time, one.b, color="#657789", linewidth=1.3, label="Eh center (B2)")
    styles = [("NO3", "#345e93", "^"), ("NH4", "#008478", "s"),
              ("Fe", "#8b5999", "D"), ("Mn", "#bc7428", "o")]
    for marker, color, shape in styles:
        part = centers[centers.analyte.eq(marker)]
        edge = boolean(part.grid_edge)
        ax.scatter(part.time_day, part.b, s=43, color=color, marker=shape,
                   label=f"{marker} descriptive center", zorder=3)
        ax.scatter(part.loc[edge, "time_day"], part.loc[edge, "b"], s=70,
                   marker="x", color="#222222", linewidths=1, zorder=4)
    ax.scatter([], [], marker="x", color="#222222", label="Position/width at search-grid edge")
    ax.plot(shared.time_day, shared.common_b, color="#b20b2d", linestyle="--",
            linewidth=1.5, label="Shared-fit compromise")
    ax.set(xlabel=f"Time within Case {case} (days)", ylabel="Center below column top (cm)",
           title="Chemical transition locations: descriptive comparison")
    ax.invert_yaxis()
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center")
    paths.extend(save_figure(fig, args.out, "chemical_centers_descriptive"))
    paths.extend(observed_boundaries(data, args.out, fits))
    if not args.skip_numerical_validation:
        paths.extend(localization_validation(args.results, args.out))
        paths.extend(grid_resolution_diagnostic(args.results, args.out))
    print(f"Saved {len(paths)} figure assets in {args.out}")


if __name__ == "__main__":
    main()
