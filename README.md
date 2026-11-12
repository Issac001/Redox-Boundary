# Redox Boundary — current manuscript reproduction

本分支对应《边界实验261111_批注答复与修订》2026-11-11 版本。一个入口从观测数据重新计算全部分析、数值表和七张图；无需本地历史结果、Excel 工作簿或 Word 文件。

主方法是约束最小二乘 sigmoid；对照是最小二乘单变点与条件于一个下降变点的贝叶斯模型。当前代码只保留这三种方法、四种模拟情景及现稿三项中心诊断。旧线性模型、AICc 判断、扩展情景和旧运行入口已从当前版本删除，历史版本可从 Git 历史查看。

## 运行

需要 Python 3.9–3.12。依赖版本与原正式数值实验一致。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/run_manuscript.py
```

默认完整执行：

1. 三工况 223 条完整十深度 Eh 剖面，三方法拟合。
2. 每条剖面的十次逐深度留一重建：共 6,690 个方法预测。
3. sigmoid、abrupt step、error function、Gompertz 四种真值；每种 500 次重复，同一模拟数据用于三方法，共 6,000 条方法评估记录。
4. 从浓度数据独立拟合 140 个化学中心，按同期记录与 Eh 中心配对。
5. 生成诊断、误差、模拟、化学比较和解释性统计；自动核对现稿冻结的五类数值表。
6. 从观测重拟合汇总 sigmoid，绘制图 2.1、图 1–5、图 C.1，每图输出 PNG、PDF、SVG。

输出到 `results_manuscript/`。为避免混入旧结果，已有非空目录不会被覆盖；再次运行请指定新目录：

```bash
python code/run_manuscript.py --output results_manuscript_new
```

仅计算全部数值、不渲染图像：

```bash
python code/run_manuscript.py --output results_numeric --no-figures
```

默认使用 Times New Roman。请在操作系统安装该字体；不随仓库分发字体。程序缺少字体时会明确报错，也可显式指定替代字体：

```bash
python code/run_manuscript.py --output results_portable --font-family "DejaVu Serif"
```

替代字体只影响显示。现稿所用的图仍应使用 Times New Roman。

## 文件与输出

| 文件 | 职责 |
| --- | --- |
| `code/run_manuscript.py` | 完整运行入口、输入检查、留一重建、模拟调度、数值回归核对与运行清单 |
| `code/config_manuscript.json` | 现稿参数和固定随机种子 |
| `code/shared/boundary_model.py` | sigmoid 拟合、数值有效性与位置工作区间 |
| `code/shared/comparison_models.py` | 三方法共同候选支持、阶跃最小二乘与贝叶斯后验预测 |
| `code/shared/comparison_simulation.py` | 四种真值生成、稳定随机流、逐重复评价 |
| `code/shared/manuscript_analysis.py` | 化学中心重拟合、三指标诊断、结果汇总及原因分析统计 |
| `code/shared/manuscript_figures_data.py` | 汇总 sigmoid 重拟合及阈值交叉统计 |
| `code/plot_manuscript_figures.py` | 七张图的统一样式和输出 |
| `code/shared/comparison_io.py` | 独立输出目录检查 |

运行目录包含 `raw/`（逐剖面/逐重复拟合与预测）、`numeric/`（论文统计表）、`figures/`（21 个图形文件及样式审计）和 `run_manifest.json`。清单记录参数、输入/代码/结果 SHA-256、运行环境及数值核对结果。输出保持完整浮点精度；图中数值保留一位小数。

完整计算后，只重绘图像可运行：

```bash
python code/plot_manuscript_figures.py --results results_manuscript
```

## 数据、复现与测试

本仓库包含经项目所有者确认可上传的现稿观测 CSV，见 [数据说明](data/README.md)。`expected_results.json` 是固定数值核对基准，绝不参与模型拟合或预测。

```bash
python -m unittest discover -s tests -v
```

37 个测试覆盖共同候选支持、后验积分、诊断、区间端点覆盖、随机流、缺失化学数据、零值、时间配对、观测权重、绘图路径和输出保护。默认入口完成后自动验证五类现稿数值表；通过 `--config` 指定其他配置时，只允许改变随机种子和模拟重复数；模型、深度和工况设置固定，避免图注或诊断口径失配。更改随机设置或观测时，清单标为非现稿标准运行，不冒用标准回归核对。

科学定义与论文对应位置见 [现稿流程说明](docs/manuscript_workflow.md)。
