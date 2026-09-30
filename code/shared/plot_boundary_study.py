"""Reusable B2 figures; read saved fits and numerical diagnostics, never refit."""
from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CJK_FONTS = (
    "Noto Sans CJK SC", "Source Han Sans SC", "Source Han Sans CN", "SimHei",
    "Microsoft YaHei", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Heiti TC", "Songti SC",
)

# Semantic colors are shared by all plots and language exports.
COLORS = {"center": "#17658A", "observed": "#343A40", "step": "#707070",
          "trend": "#D28A22", "warning": "#B65A25", "shared": "#A75835",
          "baseline": "#707070", "interval": "#BDD7E5"}
CHEMICAL_STYLES = [("NO3", "#0072B2", "^"), ("NH4", "#009E73", "s"),
                   ("Fe", "#AA6C99", "D"), ("Mn", "#D28A22", "o")]


def setup(*, language="auto", font=None) -> str:
    """Use installed fonts only; require CJK support for Chinese paper exports."""
    available = {item.name for item in font_manager.fontManager.ttflist}
    if font and font not in available:
        raise ValueError(f"Font {font!r} is not installed; available font names are platform dependent.")
    selected = font or ("DejaVu Sans" if language == "en" else
                        next((name for name in CJK_FONTS if name in available), None))
    if selected is None and language == "zh":
        raise RuntimeError(
            "No supported Chinese font is installed. Install Noto Sans CJK SC / "
            "Source Han Sans SC, specify an installed CJK font with --font, "
            "or export with --language en. Fonts are never downloaded automatically."
        )
    if selected is None:
        warnings.warn(
            "No Chinese font was detected for diagnostic plot labels. Install a CJK font "
            "or use export_paper_figures.py --language en for fully English figures.",
            RuntimeWarning, stacklevel=2,
        )
    selected = selected or "DejaVu Sans"
    families = list(dict.fromkeys([selected, "DejaVu Sans"]))
    plt.rcParams.update({"font.family": families, "font.size": 8.5,
                         "axes.titlesize": 9, "axes.labelsize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .65, "lines.linewidth": 1.2,
                         "lines.markersize": 3.5, "xtick.major.width": .6,
                         "ytick.major.width": .6, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.unicode_minus": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    return selected


def boolean(values):
    return values.astype(str).str.lower().eq("true").to_numpy()


def center_flags(part):
    """Read the revised decision; never silently reinterpret a legacy cache."""
    required = {"supported_center", "width_grid_edge"}
    if not required.issubset(part.columns):
        raise ValueError("Saved fits lack revised center diagnostics. Run "
                         "run_one_transition.py --reclassify-from OLD_RESULTS --output NEW_RESULTS first.")
    return boolean(part.supported_center), boolean(part.width_grid_edge)


def save_figure(fig, out: Path, name: str, *, dpi=220, tight=True) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg"):
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight" if tight else None)
        paths.append(path)
    plt.close(fig)
    return paths


def edges(values):
    values = np.asarray(values)
    return np.r_[values[0], (values[1:] + values[:-1]) / 2, values[-1]]


def observed_boundaries(data, out: Path, estimates):
    """Preserve the original observed Eh maps using data loaded by the caller."""
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.5), sharey=True)
    fig.subplots_adjust(left=.075, right=.88, bottom=.12, top=.83, hspace=.30)
    for ax, case in zip(axes, ["3.2", "3.3"]):
        ix = data.meta.index[data.meta.Case.eq(case)]
        t = data.meta.loc[ix, "Time"].to_numpy()
        im = ax.pcolormesh(edges(t), edges(data.depth), data.eh[ix].T,
                           cmap="RdBu_r", vmin=-500, vmax=600, shading="flat", rasterized=True)
        for model, style in [("sigmoid", "-"), ("step", "--")]:
            part = estimates[estimates.Case.eq(case) & estimates.model.eq(model)].sort_values("Time")
            x, b = part.Time.to_numpy(), part.b.to_numpy()
            if model == "sigmoid":
                ax.fill_between(x, part.b_low_working25.to_numpy(), part.b_high_working25.to_numpy(),
                                facecolor="white", alpha=.42, zorder=3)
                ax.plot(x, b, color="white", lw=2.5, zorder=4)
                ax.plot(x, b, color=COLORS["center"], lw=1.15, zorder=5)
                supported, width_edge = center_flags(part)
                ax.scatter(x[width_edge], b[width_edge], marker="s", facecolors="white",
                           edgecolors=COLORS["center"], linewidths=.55, s=13, zorder=6)
                ax.scatter(x[~supported], b[~supported], marker="x", c=COLORS["warning"],
                           linewidths=.7, s=13, zorder=7)
            else:
                ax.plot(x, b, style, color="white", lw=2.1, zorder=4)
                ax.plot(x, b, style, color=COLORS["step"], lw=.85, zorder=5)
        ax.set(title=f"Case {case}：实测 Eh 与直接拟合的转变中心", ylabel="柱顶以下深度（cm）",
               xlabel="工况内时间（天）", ylim=(95, 10), xlim=(t[0], t[-1]), yticks=[10,30,50,70,95])
    cax = fig.add_axes([.902,.22,.018,.51])
    fig.colorbar(im, cax=cax, label="实测 Eh（mV）")
    handles = [Line2D([],[],color=COLORS["center"],lw=1.2,label="Sigmoid center"),
               Patch(facecolor=COLORS["interval"],alpha=.7,label="Working interval (SD 25 mV)"),
               Line2D([],[],color=COLORS["step"],ls="--",lw=1,label="Step-model split midpoint"),
               Line2D([],[],color=COLORS["warning"],marker="x",ls="",label="Fails center rule"),
               Line2D([],[],color=COLORS["center"],marker="s",markerfacecolor="white",ls="",label="Width at search limit")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.49,.99), ncol=2, frameon=False)
    return save_figure(fig, out, "01_direct_boundary_maps", dpi=200, tight=False)


def fixed_threshold_diagnostic(data, cfg, results: Path, out: Path):
    """Visualize existing B1 crossing counts; examples are selected deterministically."""
    from core import crossings

    thresholds = cfg["thresholds_mV"]
    counts = np.array([[len(crossings(row, data.depth, value)[0]) for value in thresholds]
                       for row in data.eh])
    audit = pd.read_csv(results / "B1_threshold_audit.csv", dtype={"case": str})
    # Ensure the plotted summary is the saved experiment, not a different counting rule.
    for row in audit.itertuples(index=False):
        c = counts[data.meta.Case.eq(row.case), thresholds.index(row.threshold_mV)]
        actual = (len(c), int((c == 1).sum()), int((c > 1).sum()), int((c == 0).sum()))
        if actual != (row.n, row.single, row.multiple, row.none):
            raise ValueError(f"B1 crossing counts differ from the saved audit: {row.case}, {row.threshold_mV}")
    specs = [("1", 0, "none"), ("3.3", 200, "single"), ("3.3", 400, "multiple")]
    fig = plt.figure(figsize=(6.8, 5.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, .8], left=.09, right=.98,
                          bottom=.15, top=.9, hspace=.52, wspace=.32)
    examples = []
    colors = [COLORS["step"], COLORS["center"], COLORS["trend"]]
    for column, (case, threshold, kind) in enumerate(specs):
        c = counts[:, thresholds.index(threshold)]
        condition = c == 0 if kind == "none" else c == 1 if kind == "single" else c > 1
        eligible = data.meta[data.meta.Case.eq(case) & condition].sort_values("Time")
        index = eligible.index[len(eligible) // 2]
        profile = data.meta.loc[index]
        ax = fig.add_subplot(gs[0, column])
        ax.plot(data.eh[index], data.depth, "o-", color=COLORS["observed"], markersize=3, lw=.8)
        for value, color in zip(thresholds, colors):
            ax.axvline(value, color=color, ls="--", lw=.75, alpha=.85)
            locations = crossings(data.eh[index], data.depth, value)[0]
            valid = locations[np.isfinite(locations)]
            ax.scatter(np.full(len(valid), value), valid, facecolors="white", edgecolors=color,
                       marker="s", s=15, linewidths=.8, zorder=4)
        ax.set(xlabel="Eh (mV)", ylim=(98, 7), yticks=[10, 30, 50, 70, 90],
               title=f"({chr(97 + column)}) Case {case}; day {profile.Time:.2f}")
        ax.set_ylabel("Depth below column top (cm)" if column == 0 else "")
        examples.append({"case": case, "time": float(profile.Time), "profile_id": profile.profile_id,
                         "threshold_mV": threshold, "kind": kind})
    ax = fig.add_subplot(gs[1, :])
    positions, labels, rows = [], [], []
    for group, case in enumerate(cfg["cases"]):
        for j, threshold in enumerate(thresholds):
            positions.append(group * 4 + j)
            labels.append(str(threshold))
            rows.append(audit[audit.case.eq(case) & audit.threshold_mV.eq(threshold)].iloc[0])
    summary = pd.DataFrame(rows)
    bottom = np.zeros(len(summary))
    for kind, color in [("none", "#C5C5C5"), ("single", COLORS["center"]), ("multiple", COLORS["trend"])]:
        percent = summary[kind].to_numpy() / summary.n.to_numpy() * 100
        ax.bar(positions, percent, bottom=bottom, width=.8, color=color, label=kind)
        bottom += percent
    ax.set(xticks=positions, xticklabels=labels, ylim=(0, 100), yticks=[0, 50, 100],
           ylabel="Profiles (%)", title="(d) Profiles by crossing category")
    for group, case in enumerate(cfg["cases"]):
        ax.text(group * 4 + 1, -.25, f"Case {case}", transform=ax.get_xaxis_transform(), ha="center")
    ax.set_xlabel("Eh threshold (mV)", labelpad=28)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.36))
    fig.legend(handles=[Line2D([], [], color=color, ls="--", label=f"{value} mV")
                        for value, color in zip(thresholds, colors)],
               frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, .995))
    fig._paper_info = {"examples": examples}
    return save_figure(fig, out, "fixed_threshold_diagnostic")


def localization_validation(results: Path, out: Path) -> list[Path]:
    """B2 precision check on profiles with a known planted center."""
    path = results / "localization_summary.csv"
    if not path.exists():
        print(f"Skip B2 localization plot: missing {path}")
        return []
    table = pd.read_csv(path)
    scenarios = ["matched_iid", "matched_correlated", "matched_t5", "erf_shape", "shallow_dip", "sensor_bias"]
    labels = ["同形\n独立噪声", "同形\n相关噪声", "同形\n重尾噪声", "erf\n形态失配", "浅层\n局部凹陷", "单电极\n偏差"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.8))
    fig.subplots_adjust(left=.07, right=.98, top=.81, bottom=.25, wspace=.23)
    for j, (model, label, color) in enumerate([
        ("sigmoid", "sigmoid", COLORS["center"]),
        ("sigmoid_trend", "sigmoid＋趋势", COLORS["trend"]),
        ("step", "单变点", COLORS["step"]),
    ]):
        part = table[table.model.eq(model)].set_index("scenario").reindex(scenarios)
        axes[0].scatter(np.arange(6) + (j - 1) * .23, part.b_rmse_cm,
                        s=23, marker=["o", "^", "s"][j], color=color, label=label, zorder=3)
    axes[0].set(title="A · 已知模拟中心的定位误差", ylabel="位置 RMSE（cm）")
    axes[0].legend(frameon=False, fontsize=9, ncol=3)
    part = table[table.model.eq("sigmoid")].set_index("scenario").reindex(scenarios)
    coverage = part.working_interval_coverage_all * 100
    axes[1].scatter(np.arange(6), coverage, color=COLORS["center"], s=25, zorder=3)
    for x, value in enumerate(coverage):
        axes[1].annotate(f"{value:.1f}", (x, value), xytext=(0, 5),
                         textcoords="offset points", ha="center", fontsize=7.5)
    axes[1].axhline(95, ls="--", color=COLORS["warning"], linewidth=.85, label="工作名义水平 95%")
    axes[1].set(title="B · 25 mV 工作区间的模拟覆盖率", ylabel="覆盖已知中心的比例（%）", ylim=(0, 108))
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set(xticks=np.arange(6), xticklabels=labels)
        ax.grid(axis="y", alpha=.12, linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylim(bottom=0)
    return save_figure(fig, out, "B2_localization_validation")


def grid_resolution_diagnostic(results: Path, out: Path) -> list[Path]:
    """Expose grid-induced zero-width intervals; expanded endpoints are diagnostic."""
    path = results / "grid_resolution_diagnostic.csv"
    if not path.exists():
        print(f"Skip B2 grid diagnostic plot: missing {path}")
        return []
    table = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    fig.subplots_adjust(left=.065, right=.98, top=.81, bottom=.26, wspace=.28)
    rules = [("grid_points_hull", "网格点包络", "-", "o"),
             ("half_cell_expansion_diagnostic", "端点外扩半格（诊断）", "--", "s")]
    for j, (rule, label, style, marker) in enumerate(rules):
        part = table[table.scenario.eq("matched_iid") & table.endpoint_rule.eq(rule)].sort_values("b_grid_step_cm")
        axes[0].plot(part.b_grid_step_cm, part.working_interval_coverage_all * 100,
                     linestyle=style, marker=marker, color=[COLORS["center"], COLORS["trend"]][j], label=label)
    axes[0].axhline(95, ls=":", color=COLORS["warning"], label="工作名义水平 95%")
    axes[0].set(title="A · 同形、独立噪声：覆盖率", ylabel="覆盖率（%）", ylim=(0, 105))
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    for scenario, label, color in [("matched_iid", "同形、独立噪声", COLORS["center"]),
                                    ("shallow_dip", "浅层局部凹陷", COLORS["trend"]),
                                    ("sensor_bias", "单电极偏差", "#AA6C99")]:
        raw = table[table.scenario.eq(scenario) & table.endpoint_rule.eq("grid_points_hull")].sort_values("b_grid_step_cm")
        axes[1].plot(raw.b_grid_step_cm, raw.working_interval_zero_width_fraction * 100,
                     marker="o", color=color, label=label)
        if scenario != "matched_iid":
            for rule, _, style, marker in rules:
                part = table[table.scenario.eq(scenario) & table.endpoint_rule.eq(rule)].sort_values("b_grid_step_cm")
                axes[2].plot(part.b_grid_step_cm, part.working_interval_coverage_all * 100,
                             linestyle=style, marker=marker, color=color,
                             label=f"{label}：{'包络' if rule == 'grid_points_hull' else '外扩诊断'}")
    axes[1].set(title="B · 网格点包络：零宽区间", ylabel="零宽区间比例（%）", ylim=(-1, 30))
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].axhline(95, ls=":", color=COLORS["warning"])
    axes[2].set(title="C · 失配：细网格仍不能消除偏差", ylabel="覆盖率（%）", ylim=(0, 105))
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set(xticks=[.1, .25, .5, 1], xticklabels=["0.1", "0.25", "0.5", "1"],
               xlabel="位置搜索步长（cm）", xlim=(1.12, .09))
        ax.minorticks_off()
        ax.grid(alpha=.12, linewidth=.5)
    return save_figure(fig, out, "B2_grid_resolution_diagnostic")
