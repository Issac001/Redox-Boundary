# Redox Boundary — 论文实验复现

对应论文《动态水位砂柱中氧化还原电位转变中心的定位与化学关联》的 2026-09-30 修订稿。

本仓库复现：**一条 Eh 转变中心的估计与诊断 → Eh 特征预测化学浓度 → 化学剖面转变位置 → 已知位置模拟**。一个统计中心不等于特定反应界面；逐时刻拟合也不等于已经建立中心迁移的动态模型。

仅保留本文需要的代码、配置和使用说明。原工作簿、数值核对基线、论文文件、历史实验、结果副本、图片和缓存不提交；运行时在本地生成结果。

## 环境与数据

使用 Python 3.9–3.12。验证环境为 Python 3.9.6；六个依赖包的版本固定在 requirements.txt。无需服务器或 GPU。

~~~bash
git clone git@github.com:Issac001/Redox-Boundary.git
cd Redox-Boundary
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

实测实验需要协作者提供的 Whole3.1.xlsx。该文件的再分发许可尚未确认，因此不上传。将文件放入 data/Whole3.1.xlsx，或通过 --data 指定外部路径。格式、单位和缺失处理见 [数据说明](data/README.md)。

## 完整复现

以下命令从仓库根目录运行。命令使用占位路径，请替换为自己的工作簿位置；若已放入默认 data 目录，可省略 --data。

~~~bash
python code/run_one_transition.py \
  --data /path/to/Whole3.1.xlsx \
  --output results

python code/export_paper_figures.py \
  --data /path/to/Whole3.1.xlsx \
  --results results \
  --out figures \
  --language zh
~~~

第一条命令生成实测分析、六情景模拟、配对网格诊断以及 paper_audit.json。第二条命令复用同一套绘图函数，生成七张论文图的 PDF、SVG、PNG，包括固定阈值交点诊断图；语言可选 zh、en、both。中文导出需要本机已有的中文字体；缺少字体时可先用 --language en。

默认配置为 code/config_one_transition.json。运行与绘图入口均支持 --config；配置中的相对数据路径相对于仓库根目录。通过 --output / --results / --out 指定生成文件位置，勿混用不同配置的结果。

## 无原数据时：单独复现数值模拟

~~~bash
python code/run_one_transition.py \
  --simulation-only \
  --output results_simulation

python code/export_paper_figures.py \
  --simulation-only \
  --results results_simulation \
  --out figures_simulation \
  --language en
~~~

该模式复现六个已知中心情景和网格敏感性，以及两张对应图；它不生成实测 Eh 或化学结果。每情景 500 条剖面，种子 20260923。保留原子随机流和配对抽样顺序，保证本文的模拟数字可重放。

若只重跑实测部分，使用 --skip-numerical-validation，并指定另一个输出目录；此时生成五张实测图也应给绘图入口同名选项。

## 代码结构

| 文件 | 职责 |
|---|---|
| code/run_one_transition.py | 统一计算入口；控制论文范围、重分类已存诊断、调用审计、生成无作者绝对路径的运行记录 |
| code/shared/core.py | 工作簿读取、数据口径检查、阈值交点 |
| code/shared/boundary_model.py | 可复用的约束 Sigmoid、趋势 Sigmoid、单变点与位置工作区间 |
| code/shared/run_boundary_study.py | 实测中心、分开的中心与宽度诊断、深度留出、化学预测及化学/共享位置 |
| code/shared/boundary_simulation.py | 六情景定位误差、工作区间覆盖及配对网格诊断 |
| code/plot_one_transition.py | 同一入口绘制五张实测图，并调用共享模拟图 |
| code/shared/plot_boundary_study.py | 阈值诊断、热图、模拟图及公共绘图设置 |
| code/export_paper_figures.py | 论文尺寸、语言与矢量导出；检查排版前后的数值图元一致性 |
| code/audit_paper_results.py | 论文表格与补充统计、新旧规则交集、Case 1 误差归因、参考值比较 |

四个计算核心文件保持原估计逻辑。论文范围在调用层限定：化学预测只运行 Case 3.2 / C3、Case 3.3 / C4、C5，以及本文使用的基准、同期 Eh、固定深度、固定阈值和 Sigmoid 特征。其余历史化学预测分支不执行。趋势 Sigmoid 和单变点仍用于本文的 Eh 重建与定位模拟对照，不能作为冗余删去。

逐图、逐表及正文数字的输出对应见 [论文证据与代码对应](docs/paper_reproduction_map.md)。

## 中心与宽度诊断

2026-09-30 修订只将宽度触及搜索边界从中心的否决条件中拆出。中心操作性条件仍为振幅 ≥100 mV、相对无转变模型的 ΔAICc <0、25 mV 工作区间包络宽度 ≤20 cm、中心未触及位置搜索边界。条件通过数量是诊断分类，不是准确率或校准后的定位置信度。

- `supported_center`：当前中心规则；论文图表使用此列。
- `width_grid_edge`：宽度触及搜索下限或上限，单独提示厚度尚未充分解析；不单独否决中心。
- `supported_shape`：保留旧的中心与宽度共同否决规则，供版本对照，不作为当前图形的通过标记。
- `case_summary.csv` 的 `n_center_supported` 是当前计数；`n_shape_supported` 保留旧计数；`n_center_supported_width_grid_edge` 给出当前通过与宽度触边的交集。

已有旧版结果时，无需重新拟合即可生成一份独立的新结果目录：

~~~bash
python code/run_one_transition.py \
  --reclassify-from results \
  --output results_revised

python code/audit_paper_results.py \
  --data /path/to/Whole3.1.xlsx \
  --results results_revised
~~~

重分类仅写入新的空目录，保留全部数值估计与原规则，新增中心诊断并更新汇总。代码逐值核对原 B2 列，其他实验 CSV 原样复制；`source_manifest.json` 保留来源运行记录，新的 `manifest.json` 记录来源与产物哈希。原审计文件不复制，因为其哈希对应旧 CSV；第二条命令重建新版审计。后续绘图使用 `--results results_revised`。

## 输出与核对

- results/*.csv：重新计算的估计、预测及模拟明细。
- results/paper_audit.json：覆盖论文主要数值表和正文补充统计。
- results/manifest.json：实际依赖版本、配置、原工作簿哈希、相对代码路径及结果哈希。
- figures/：按语言保存七图及图元指纹。图形读取保存结果和原工作簿，示例曲线重建时核对 SSE，阈值图核对交点计数，不在绘图时重新估计中心。

已经生成结果后，也可单独重新汇总和验证：

~~~bash
python code/audit_paper_results.py \
  --data /path/to/Whole3.1.xlsx \
  --results results
~~~

数值核对基线保留在数据持有者本地，不随代码发布。若已获授权取得相应参考 JSON，可给计算或审计入口传入 --check-reference /path/to/private/paper_metrics.json；比较容差为相对/绝对 1e-8，逻辑值、字符串、缺失和列表长度严格核对。没有参考文件也能正常生成全部实验结果。

旧基线继续核对不变的估计、预测、模拟及旧规则统计；当前规则另见审计的 `center_rules_by_case`。通过旧基线比较不代表旧基线已经验证新增规则。更新私有基线时应另存为新版本，并保留旧版来源记录。

## 解释范围

化学浓度预测是利用同期 Eh 推测同期浓度、按先后循环留出测试，不是提前预测未来化学变化。化学位置拟合直接使用浓度，属于空间对应分析，不是独立真值验证。25 mV 与卡方 95% 阈值定义的是工作区间，实测覆盖尚未校准。约 10 cm 的观测间距不会因 1 cm 或 0.1 cm 计算网格而变细。

砂柱实验背景来自 Zhang & Furman (2021), *Journal of Hydrology*, [doi:10.1016/j.jhydrol.2021.126899](https://doi.org/10.1016/j.jhydrol.2021.126899)。模型、验证与来源的对应见上述证据表；引用不代表原文规定了本研究的方向约束、网格、形状阈值或共享损失。
