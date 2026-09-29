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
    "Microsoft YaHei", "WenQuanYi Zen Hei", "PingFang SC", "Songti SC", "Arial Unicode MS",
)


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
    plt.rcParams.update({"font.family": families, "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "svg.fonttype": "none"})
    return selected


def boolean(values):
    return values.astype(str).str.lower().eq("true").to_numpy()


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
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.7), sharey=True)
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
                ax.plot(x, b, color="white", lw=4, zorder=4)
                ax.plot(x, b, color="#182b3a", lw=2, zorder=5)
                weak = ~boolean(part.supported_shape)
                ax.scatter(x[weak], b[weak], marker="x", c="#d08900", s=25, zorder=6)
            else:
                ax.plot(x, b, style, color="#fff29a", lw=1.8, zorder=5)
        ax.set(title=f"Case {case}：实测 Eh 与直接拟合的转变中心", ylabel="柱顶以下深度（cm）",
               xlabel="工况内时间（天）", ylim=(95, 10), xlim=(t[0], t[-1]), yticks=[10,30,50,70,95])
    cax = fig.add_axes([.902,.22,.018,.51])
    fig.colorbar(im, cax=cax, label="实测 Eh（mV）")
    handles = [Line2D([],[],color="#182b3a",lw=2,label="sigmoid 中心 b(t)"),
               Patch(facecolor="#becbd4",alpha=.7,label="25 mV 假设下的位置工作区间"),
               Line2D([],[],color="#b29b0b",ls="--",lw=2,label="单变点中心"),
               Line2D([],[],color="#d08900",marker="x",ls="",label="未满足预定形态支持判据")]
    fig.suptitle("直接估计得到的是 Eh 转变中心，化学意义需另行检验", y=.98, fontsize=18)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.49,.94), ncol=2, frameon=False)
    fig.text(.075,.035,"深度沿用原表柱坐标；相对土面深度＝图示深度−9 cm。底图仅展示观测网格。\n浅色带依赖独立、已知噪声标准差 25 mV 的工作假设，并非已验证的真实反应边界置信带。",fontsize=10,color="#455565")
    return save_figure(fig, out, "01_direct_boundary_maps", dpi=200, tight=False)


def localization_validation(results: Path, out: Path) -> list[Path]:
    """B2 precision check on profiles with a known planted center."""
    path = results / "localization_summary.csv"
    if not path.exists():
        print(f"Skip B2 localization plot: missing {path}")
        return []
    table = pd.read_csv(path)
    scenarios = ["matched_iid", "matched_correlated", "matched_t5", "erf_shape", "shallow_dip", "sensor_bias"]
    labels = ["同形\n独立噪声", "同形\n相关噪声", "同形\n重尾噪声", "erf\n形态失配", "浅层\n局部凹陷", "单电极\n偏差"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3))
    fig.subplots_adjust(left=.07, right=.98, top=.81, bottom=.25, wspace=.23)
    for j, (model, label, color) in enumerate([
        ("sigmoid", "sigmoid", "#216b91"),
        ("sigmoid_trend", "sigmoid＋趋势", "#e59c32"),
        ("step", "单变点", "#9767a7"),
    ]):
        part = table[table.model.eq(model)].set_index("scenario").reindex(scenarios)
        axes[0].bar(np.arange(6) + (j - 1) * .23, part.b_rmse_cm,
                    width=.21, color=color, label=label)
    axes[0].set(title="A · 已知模拟中心的定位误差", ylabel="位置 RMSE（cm）")
    axes[0].legend(frameon=False, fontsize=9, ncol=3)
    part = table[table.model.eq("sigmoid")].set_index("scenario").reindex(scenarios)
    coverage = part.working_interval_coverage_all * 100
    bars = axes[1].bar(np.arange(6), coverage, color="#216b91", width=.65)
    axes[1].bar_label(bars, labels=[f"{x:.1f}%" for x in coverage], padding=3, fontsize=9)
    axes[1].axhline(95, ls="--", color="#b4463f", label="工作名义水平 95%")
    axes[1].set(title="B · 25 mV 工作区间的模拟覆盖率", ylabel="覆盖已知中心的比例（%）", ylim=(0, 108))
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set(xticks=np.arange(6), xticklabels=labels)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
    fig.suptitle("B2 数值验证：定位误差与工作区间覆盖", y=.97, fontsize=18)
    counts = ", ".join(str(int(n)) for n in sorted(part.n.unique()))
    fig.text(.07, .095, f"每情景 {counts} 个模拟剖面；位置搜索步长为 1 cm。区间为满足工作准则的网格点包络。\n"
             "覆盖率只适用于所设模拟情景；并非实测反应边界验证。相关噪声、局部失配与电极偏差均用于评价定位误差。",
             fontsize=10, color="#455565")
    return save_figure(fig, out, "B2_localization_validation")


def grid_resolution_diagnostic(results: Path, out: Path) -> list[Path]:
    """Expose grid-induced zero-width intervals; expanded endpoints are diagnostic."""
    path = results / "grid_resolution_diagnostic.csv"
    if not path.exists():
        print(f"Skip B2 grid diagnostic plot: missing {path}")
        return []
    table = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    fig.subplots_adjust(left=.065, right=.98, top=.81, bottom=.26, wspace=.28)
    rules = [("grid_points_hull", "网格点包络", "-", "o"),
             ("half_cell_expansion_diagnostic", "端点外扩半格（诊断）", "--", "s")]
    for rule, label, style, marker in rules:
        part = table[table.scenario.eq("matched_iid") & table.endpoint_rule.eq(rule)].sort_values("b_grid_step_cm")
        axes[0].plot(part.b_grid_step_cm, part.working_interval_coverage_all * 100,
                     linestyle=style, marker=marker, label=label)
    axes[0].axhline(95, ls=":", color="#b4463f", label="工作名义水平 95%")
    axes[0].set(title="A · 同形、独立噪声：覆盖率", ylabel="覆盖率（%）", ylim=(0, 105))
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    for scenario, label, color in [("matched_iid", "同形、独立噪声", "#216b91"),
                                    ("shallow_dip", "浅层局部凹陷", "#e59c32"),
                                    ("sensor_bias", "单电极偏差", "#9767a7")]:
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
    axes[2].axhline(95, ls=":", color="#b4463f")
    axes[2].set(title="C · 失配：细网格仍不能消除偏差", ylabel="覆盖率（%）", ylim=(0, 105))
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set(xticks=[.1, .25, .5, 1], xticklabels=["0.1", "0.25", "0.5", "1"],
               xlabel="位置搜索步长（cm）", xlim=(1.12, .09))
        ax.minorticks_off()
        ax.grid(alpha=.16)
    fig.suptitle("B2 网格诊断：零宽区间不等于高精度", y=.97, fontsize=18)
    fig.text(.065, .08, "同一批模拟剖面的事后数值敏感性分析；宽度网格和 25 mV 工作假设保持不变。\n"
             "端点外扩半格仅用于诊断网格离散化影响，未经覆盖校准，不能作为修正后的置信区间。模拟结果不构成实测化学边界验证。",
             fontsize=10, color="#455565")
    return save_figure(fig, out, "B2_grid_resolution_diagnostic")
