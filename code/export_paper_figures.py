"""Export the seven manuscript figures in Chinese and/or English.

Reuse the production plotters, changing only presentation at save time. A
numerical-artist signature checks coordinates and limits before/after layout.
No historical folders, author-specific paths, or font downloads are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "redox_boundary_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "redox_boundary_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(HERE / "shared"))
import plot_boundary_study as shared  # noqa: E402
import plot_one_transition as entry  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_signature(fig):
    """Fingerprint plotted coordinates, color values, bars, and axis limits."""
    digest = hashlib.sha256()
    for ax in fig.axes:
        arrays = [ax.get_xlim(), ax.get_ylim()]
        for line in ax.lines:
            arrays.extend([line.get_xdata(), line.get_ydata()])
        for collection in ax.collections:
            arrays.extend([collection.get_offsets(), collection.get_array()])
            arrays.extend(path.vertices for path in collection.get_paths())
        for patch in ax.patches:
            arrays.append(patch.get_path().vertices)
            if hasattr(patch, "get_xy"):
                arrays.append(patch.get_xy())
            if hasattr(patch, "get_width"):
                arrays.extend([[patch.get_width(), patch.get_height()]])
        for values in arrays:
            if values is None:
                continue
            array = np.ma.asarray(values, dtype=float)
            digest.update(str(array.shape).encode())
            digest.update(np.ma.filled(array, np.nan).tobytes())
    return digest.hexdigest()


def relabel_legend(ax, labels, *, loc="upper center", anchor=(.5, -.26), ncol=1):
    legend = ax.get_legend()
    handles = (getattr(legend, "legend_handles", None) or legend.legendHandles) if legend else ax.get_legend_handles_labels()[0]
    if legend:
        legend.remove()
    return ax.legend(handles, labels, frameon=False, fontsize=8, loc=loc,
                     bbox_to_anchor=anchor, ncol=ncol, handlelength=1.8,
                     columnspacing=1.0, labelspacing=.35, borderaxespad=0.)


def paper_layout(fig, name, language):
    choose = lambda zh, en: zh if language == "zh" else en
    fig.set_layout_engine(None)
    for text in list(fig.texts):
        text.remove()  # Long global titles/notes are already in the manuscript captions.
    axes = fig.axes
    for ax in axes:
        ax.title.set_fontsize(9.5)
        ax.xaxis.label.set_fontsize(8.5)
        ax.yaxis.label.set_fontsize(8.5)
        ax.tick_params(labelsize=8.5)
    if name == "one_Eh_transition":
        info = fig._paper_info
        day, center, median_cm = info["day"], info["center"], info["median"]
        fig.set_size_inches(6.8, 4.0)
        for ax, left in zip(axes, [.10, .59]):
            ax.set_position([left, .36, .36, .53])
        axes[0].set(title=choose(f"(a) 第 {day:.2f} 天；中心 {center:.0f} cm", f"(a) Day {day:.2f}; center {center:.0f} cm"),
                    xlabel="Eh (mV)", ylabel=choose("柱顶以下深度（cm）", "Depth below column top (cm)"))
        axes[1].set(title=choose("(b) 逐时刻中心估计", "(b) Pointwise center estimates"),
                    xlabel=choose("工况内时间（天）", "Time within case (days)"),
                    ylabel=choose("柱顶以下中心深度（cm）", "Center depth (cm)"))
        relabel_legend(axes[0], [choose("单转变拟合", "One-transition fit"), choose("实测 Eh", "Observed Eh")], anchor=(.5, -.27))
        relabel_legend(axes[1], [choose("中心诊断未通过", "Fails center diagnostic"), choose("中心诊断通过", "Passes center diagnostic"),
                              choose("宽度触搜索边界", "Width at search limit"),
                              choose(f"中位数 {median_cm:.0f} cm", f"Median: {median_cm:.0f} cm")], anchor=(.5, -.27))
    elif name == "chemical_prediction":
        fig.set_size_inches(6.8, 3.25)
        for ax, left, letter in zip(axes, [.10, .60], ["a", "b"]):
            panel = re.fullmatch(r"Case (.+) / (.+): (\d+) chemical times", ax.get_title())
            if not panel:
                raise ValueError("The chemical prediction plot no longer exposes its fold labels")
            case, fold, count = panel.groups()
            ax.set_position([left, .24, .35, .55])
            ax.set_title(f"({letter}) Case {case} / {fold}" + choose(f"（{count} 时点）", f" ({count} times)"))
            ax.set_xlabel(choose("log1p-RMSE 相对变化\n（相对基准，%）", "Change in log1p-RMSE\n(% vs baseline)"))
        relabel_legend(axes[1], [choose("加入同期 Eh", "Add current Eh"), choose("加入转变特征", "Add transition features")],
                       anchor=(-.22, 1.32), ncol=2)
    elif name == "chemical_centers_descriptive":
        fig.set_size_inches(6.8, 3.7)
        ax = axes[0]
        ax.set_position([.10, .32, .86, .59])
        ax.set(title="",
               xlabel=choose("工况内时间（天）", "Time within case (days)"),
               ylabel=choose("柱顶以下中心深度（cm）", "Center depth below column top (cm)"))
        relabel_legend(ax, ["Eh", "NO$_3$", "NH$_4$", "Fe", "Mn",
                           choose("中心或宽度触边", "Center / width at grid edge"),
                           choose("多通道共享中心", "Shared-center fit")], anchor=(.5, -.30), ncol=3)
    elif name == "01_direct_boundary_maps":
        fig.set_size_inches(6.8, 5.5)
        for ax, bottom, case, letter in zip(axes[:2], [.55, .12], ["3.2", "3.3"], ["a", "b"]):
            ax.set_position([.10, bottom, .74, .29])
            ax.set(title=f"({letter}) Case {case}", xlabel=choose("工况内时间（天）", "Time within case (days)"),
                   ylabel=choose("柱顶以下深度（cm）", "Depth below column top (cm)"))
        axes[2].set_position([.875, .12, .02, .72])
        axes[2].set_ylabel(choose("实测 Eh（mV）", "Observed Eh (mV)"), fontsize=8.5)
        legend = fig.legends[0]
        handles = getattr(legend, "legend_handles", None) or legend.legendHandles
        legend.remove()
        fig.legend(handles, [choose("sigmoid 中心 $c_t$", "Sigmoid center $c_t$"),
                             choose("工作区间（SD 25 mV）", "Working interval (SD 25 mV)"),
                             choose("单变点间隙中点", "Step-model split midpoint"),
                             choose("中心诊断未通过", "Fails center diagnostic"),
                             choose("宽度触搜索边界", "Width at search limit")],
                   loc="upper center", bbox_to_anchor=(.5, .995), ncol=2,
                   frameon=False, fontsize=8.5, columnspacing=1.0, handlelength=1.8)
    elif name == "B2_localization_validation":
        fig.set_size_inches(6.8, 3.8)
        for ax, left in zip(axes, [.10, .60]):
            ax.set_position([left, .22, .35, .59])
        axes[0].set(title=choose("(a) 已知中心的定位误差", "(a) Center localization error"),
                    ylabel=choose("中心 RMSE（cm）", "Center RMSE (cm)"))
        axes[1].set(title=choose("(b) 工作区间覆盖", "(b) Working-interval coverage"),
                    ylabel=choose("覆盖率（%）", "Coverage (%)"))
        labels = choose(["同形\n独立", "同形\n相关", "同形\n$t_5$", "erf\n形态", "浅层\n凹陷", "电极\n偏差"],
                        ["iid", "corr.", "$t_5$", "erf", "dip", "bias"])
        for ax in axes:
            ax.set_xticks(np.arange(6), labels=labels)
        relabel_legend(axes[0], ["Sigmoid", choose("Sigmoid＋趋势", "Sigmoid + trend"), choose("单变点", "Step model")],
                       anchor=(1.20, 1.27), ncol=3)
        relabel_legend(axes[1], [choose("名义水平 95%", "Nominal level: 95%")], loc="upper right", anchor=(1, 1))
    elif name == "B2_grid_resolution_diagnostic":
        fig.set_size_inches(6.8, 6.2)
        for ax, bottom in zip(axes, [.72, .42, .12]):
            ax.set_position([.13, bottom, .83, .20])
            ax.set_xlabel(choose("中心搜索步长（cm）", "Center search step (cm)"))
        axes[0].set(title=choose("(a) 同形、独立噪声：覆盖率", "(a) Matched iid noise: coverage"),
                    ylabel=choose("覆盖率（%）", "Coverage (%)"))
        axes[1].set(title=choose("(b) 网格点包络：零宽区间", "(b) Grid-point hull: zero-width intervals"),
                    ylabel=choose("零宽区间比例（%）", "Zero-width intervals (%)"))
        axes[2].set(title=choose("(c) 细网格仍不能消除失配偏差", "(c) Fine grids do not remove misspecification bias"),
                    ylabel=choose("覆盖率（%）", "Coverage (%)"))
        relabel_legend(axes[0], [choose("网格点包络", "Grid-point hull"),
                               choose("半格外扩（仅诊断）", "Half-cell expansion (diagnostic)"),
                               choose("名义水平 95%", "Nominal level: 95%")], loc="lower left", anchor=(.02, .02))
        relabel_legend(axes[1], [choose("同形、独立噪声", "Matched iid noise"),
                               choose("浅层局部凹陷", "Shallow local dip"), choose("单电极偏差", "Single-sensor bias")],
                       loc="upper right", anchor=(1, 1))
        relabel_legend(axes[2], [choose("凹陷：包络", "Local dip: grid-point hull"),
                               choose("凹陷：半格外扩", "Local dip: half-cell expansion"),
                               choose("电极偏差：包络", "Sensor bias: grid-point hull"),
                               choose("电极偏差：半格外扩", "Sensor bias: half-cell expansion")],
                       loc="upper right", anchor=(1, .84), ncol=2)
    elif name == "fixed_threshold_diagnostic":
        fig.set_size_inches(6.8, 5.5)
        for j, (ax, example) in enumerate(zip(axes[:3], fig._paper_info["examples"])):
            ax.set_position([.09 + j * .31, .54, .24, .34])
            ax.set_title(f"({chr(97 + j)}) Case {example['case']}; " +
                         choose(f"第 {example['time']:.2f} 天", f"day {example['time']:.2f}"))
            if j == 0:
                ax.set_ylabel(choose("柱顶以下深度（cm）", "Depth below column top (cm)"))
        ax = axes[3]
        ax.set_position([.09, .16, .87, .20])
        ax.set(title=choose("(d) 各工况的交点类型比例", "(d) Profiles by crossing category"),
               ylabel=choose("剖面比例（%）", "Profiles (%)"),
               xlabel=choose("Eh 阈值（mV）", "Eh threshold (mV)"))
        relabel_legend(ax, [choose("无交点", "No crossing"), choose("唯一交点", "One crossing"),
                            choose("多个交点", "Multiple crossings")], anchor=(.5, 1.45), ncol=3)
    else:
        raise ValueError(name)
    # Apply the paper font scale after any new title, tick, or legend objects.
    for artist in fig.findobj(match=Text):
        artist.set_fontsize(8.5)
    if name == "B2_localization_validation":
        # Keep adjacent percentage labels legible with wider Latin fonts.
        for annotation in axes[1].texts:
            annotation.set_fontsize(7.4)
    for ax in axes:
        ax.title.set_fontsize(9)
        for collection in ax.collections:
            collection.set_rasterized(False)


FIGURE_NAMES = (
    "one_Eh_transition", "chemical_prediction", "chemical_centers_descriptive",
    "01_direct_boundary_maps", "fixed_threshold_diagnostic",
    "B2_localization_validation", "B2_grid_resolution_diagnostic",
)
RESULT_FILES = (
    "B2_boundary_estimates.csv", "B3_chemistry_metrics.csv",
    "B3_chemical_transition_centers.csv", "B3_shared_transition_diagnostic.csv", "B1_threshold_audit.csv",
    "localization_summary.csv", "grid_resolution_diagnostic.csv",
)


def export_language(language, args, input_hashes):
    out = args.out / f"figures_{language}"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "translation_manifest.json"
    records = {}
    if args.figure and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("input_sha256") == input_hashes:
            records = previous["figures"]
    original_shared_save, original_entry_save = shared.save_figure, entry.save_figure

    def paper_save(fig, unused_out, name, **unused_kwargs):
        if args.figure and name != args.figure:
            plt.close(fig)
            return []
        # Diagnostic labels are translated below. Only this pre-layout draw may
        # contain Chinese glyphs while the English font is already selected.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")
            fig.canvas.draw()
        before = numeric_signature(fig)
        paper_layout(fig, name, language)
        fig.canvas.draw()
        after = numeric_signature(fig)
        if before != after:
            raise AssertionError(f"Presentation edit altered numerical artists: {name}")
        if language == "en":
            chinese = [t.get_text() for t in fig.findobj(match=Text)
                       if re.search(r"[\u3400-\u9fff]", t.get_text())]
            if chinese:
                raise AssertionError((name, chinese))
        paths = []
        for extension in ("pdf", "svg", "png"):
            path = out / f"{name}.{extension}"
            fig.savefig(path, dpi=300, facecolor="white")
            paths.append(path)
        records[name] = {
            "canvas_inches": fig.get_size_inches().tolist(),
            "font_pt": 8.5, "panel_title_pt": 9,
            "coverage_annotation_pt": 7.4 if name == "B2_localization_validation" else None,
            "numerical_artist_sha256": after,
            "assets_sha256": {p.name: sha256(p) for p in paths},
            "plot_metadata": getattr(fig, "_paper_info", {}),
        }
        plt.close(fig)
        return paths

    shared.save_figure = entry.save_figure = paper_save
    try:
        if args.simulation_only:
            shared.setup(language=language, font=args.font)
            shared.localization_validation(args.results, out)
            shared.grid_resolution_diagnostic(args.results, out)
        else:
            plot_args = ["--data", str(args.data), "--config", str(args.config),
                         "--results", str(args.results), "--out", str(out)]
            if args.skip_numerical_validation:
                plot_args.append("--skip-numerical-validation")
            entry.main(plot_args, font_language=language, font=args.font)
    finally:
        shared.save_figure, entry.save_figure = original_shared_save, original_entry_save
        plt.close("all")
    manifest = {
        "language": language,
        "font_family": plt.rcParams["font.family"],
        "plotter_sha256": {p.name: sha256(p) for p in (
            Path(__file__), Path(entry.__file__), Path(shared.__file__))},
        "input_sha256": input_hashes,
        "strategy": "Reuse production plotters; caller-only layout and labels; numerical-artist signatures match",
        "font_scale": "Body/ticks/legends 8.5 pt; panel titles 9 pt at 6.8 inches",
        "figures": records,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        help="Input workbook; default: config data path, relative to repository root")
    parser.add_argument("--config", type=Path, default=HERE / "config_one_transition.json")
    parser.add_argument("--results", type=Path, default=PACKAGE / "results")
    parser.add_argument("--out", type=Path, default=PACKAGE / "figures" / "paper",
                        help="Output root; creates figures_zh and/or figures_en subdirectories")
    parser.add_argument("--language", choices=["zh", "en", "both"], default="both")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simulation-only", action="store_true",
                      help="Export only the two simulation figures; no workbook or config is required")
    mode.add_argument("--skip-numerical-validation", action="store_true",
                      help="Export only the five observation/chemistry figures; no simulation CSVs required")
    parser.add_argument("--font", help="Installed font family; optional override of portable font detection")
    parser.add_argument("--figure", choices=FIGURE_NAMES,
                        help="Re-export one figure, retaining existing exports of other figures")
    args = parser.parse_args(argv)
    names = (FIGURE_NAMES[-2:] if args.simulation_only else
             FIGURE_NAMES[:5] if args.skip_numerical_validation else FIGURE_NAMES)
    if args.figure:
        if args.figure not in names:
            parser.error("The requested figure is excluded by the selected export mode")
        names = (args.figure,)
    if not args.simulation_only and args.data is None:
        if not args.config.is_file():
            parser.error(f"Missing configuration: {args.config}")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        args.data = PACKAGE / Path(config.get("data", "data/Whole3.1.xlsx"))
    languages = ["zh", "en"] if args.language == "both" else [args.language]
    for language in languages:
        shared.setup(language=language, font=args.font)
    result_names = (RESULT_FILES[-2:] if args.simulation_only else
                    RESULT_FILES[:5] if args.skip_numerical_validation else RESULT_FILES)
    inputs = [args.results / name for name in result_names]
    if not args.simulation_only:
        inputs = [args.data, args.config] + inputs
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        parser.error("Missing required inputs; run the experiment pipeline first: " + ", ".join(missing))
    frozen = {path: sha256(path) for path in inputs}
    input_hashes = {path.name: digest for path, digest in frozen.items()}
    manifests = [export_language(language, args, input_hashes) for language in languages]
    if len(manifests) == 2:
        for name in names:
            if (manifests[0]["figures"][name]["numerical_artist_sha256"] !=
                    manifests[1]["figures"][name]["numerical_artist_sha256"]):
                raise AssertionError(f"Chinese/English numerical artists differ: {name}")
    if not all(sha256(path) == digest for path, digest in frozen.items()):
        raise AssertionError("An input file changed during figure export")
    print(f"Paper exports ready: {', '.join(languages)}; {len(frozen)} input files unchanged; "
          f"output root: {args.out}")


if __name__ == "__main__":
    main()
