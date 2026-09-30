# 当前论文的证据与复现对应

对应 2026-09-30 修订稿。正文结果顺序为 Eh 中心与诊断、化学预测、化学位置对应、已知位置模拟。全部图统一由 code/export_paper_figures.py 导出；数学估计复用 code/shared/boundary_model.py。

## 七张图

| 论文位置 / 图名 | 计算输入或输出 | 实际绘图函数 |
|---|---|---|
| 附录 D 阈值交点诊断图 / fixed_threshold_diagnostic | 工作簿；B1_threshold_audit.csv | shared/plot_boundary_study.py::fixed_threshold_diagnostic |
| 图 1 / 01_direct_boundary_maps | 工作簿；B2_boundary_estimates.csv | shared/plot_boundary_study.py::observed_boundaries |
| 图 2 / one_Eh_transition | 工作簿；B2_boundary_estimates.csv | plot_one_transition.py::main 的单剖面和位置序列 |
| 图 3 / chemical_prediction | B3_chemistry_metrics.csv | plot_one_transition.py::main 的预测增量 |
| 图 4 / chemical_centers_descriptive | B2_boundary_estimates.csv；B3_chemical_transition_centers.csv；B3_shared_transition_diagnostic.csv | plot_one_transition.py::main 的化学位置对照 |
| 图 5 / B2_localization_validation | localization_summary.csv | shared/plot_boundary_study.py::localization_validation |
| 附录图 C.1 / B2_grid_resolution_diagnostic | grid_resolution_diagnostic.csv | shared/plot_boundary_study.py::grid_resolution_diagnostic |

图 2 对应本文 Case 3.3 的指定示例剖面；十深度 Eh 中心与化学共享模型内部九深度 Eh 拟合应区分。七图统一导出，无需任何旧修订目录；阈值诊断复用原交点计算与已存审计表。

## 数值表与正文补充结果

所有路径相对于一次运行的结果目录。审计列为 paper_audit.json 中的 key。

| 论文证据 | 输出 / 审计 key |
|---|---|
| 表 1：五工况剖面数 | case_summary.csv；水文设计与进水组成来自原实验论文，不能由本代码重新“估计” |
| 表 2：训练/校准/测试划分 | code/config_one_transition.json；B3_fold_audit.csv；chemical_primary_folds |
| 表 3：中心与宽度诊断 | case_summary.csv；case_summary；当前 n_center_supported 与宽度触边独立汇总 |
| 表 4：化学预测相对改善 | B3_chemistry_metrics.csv；chemical_primary_metrics |
| 表 C.1：六情景定位与覆盖 | localization_summary.csv；localization_summary |
| 表 C.2：配对网格诊断 | grid_resolution_diagnostic.csv；grid_resolution_diagnostic |
| 表 D.1：化学观测覆盖 | B1_chemistry_audit.csv；chemical_coverage_main_depths |
| 表 D.2：阈值交点 | B1_threshold_audit.csv；threshold_crossing_counts |
| 表 D.3：留一深度误差 | B2_leave_depth_out.csv；leave_depth_out_metrics |
| 表 F.1：化学绝对误差 | B3_chemistry_metrics.csv；chemical_primary_metrics |
| 表 F.2：逐时点改善 | B3_chemistry_by_time.csv；case33_chemical_by_test_time |
| 表 E.1：新旧规则对照，以及各工况当前规则失败数 | center_rules_by_case；同时报告通过中心规则且宽度触边的交集 |
| Case 3.3 旧版形状规则单项与交集失败数 | case33_shape_rules；保留作旧版基线对照 |
| Case 1 各深度的留出 SSE 归因 | case1_lodo_replay；重新调用同一估计器，核对总 RMSE 后分解 |
| 10/25/50 mV 工作区间诊断 | B2_interval_sensitivity.csv；working_interval_sigma_sensitivity |
| 化学位置拟合的有效数量与触边数量 | case33_descriptive_chemical_centers；case33_chemical_center_totals |
| 多通道共享位置范围 | case33_shared_centers |
| 化学各指标改善时点数量 | case33_chemical_improved_time_counts |
| Fe 纳入 95 cm 的敏感性 | fe_depth_sensitivity |
| 原工作簿读取范围、唯一周期填补、无恰等阈值 | raw_loader_audit；threshold_exact_equalities |
| 表 F.1、G.1 | 文件索引和公式依据是方法说明，不是另一项数值实验 |

## 所用方法及边界

- 四参数 logistic：用平台中点 / 最大下降梯度定义 Eh 中心；[经典函数定义](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/SSfpl.html)。本代码不调用 R 的 self-start 算法。
- 固定位置与宽度时先消去线性平台参数，再搜索非线性网格：[Golub & Pereyra (2003)](https://doi.org/10.1088/0266-5611/19/2/201) 的变量投影思想。具体网格由本研究设定。
- 位置剖面损失：[Raue et al. (2009)](https://doi.org/10.1093/bioinformatics/btp358) 的参数约束评价思想；有限深度的区间覆盖由模拟另行评价。
- 岭回归和留出误差：[Hastie et al. (2009)](https://doi.org/10.1007/978-0-387-84858-7)；时间划分、相对中心特征及等时点汇总是本研究设置。
- 已知中心下评价误差与覆盖：[Morris et al. (2019)](https://doi.org/10.1002/sim.8086) 的模拟评价原则；六个生成情景不是从实测识别的反应机制。
- 化学方向假设、共享损失归一化、形状阈值和工作噪声均为本研究操作性设置。化学中心定义在方向调整后的 log1p 浓度尺度上。

## 诊断版本对应

当前中心标记为 B2_boundary_estimates.csv 中的 `supported_center`：振幅、AICc 与区间宽度要求沿用原设定，只将搜索边界条件从 `grid_edge` 改为 `b_grid_edge`。`width_grid_edge` 单独报告，不将宽度触边自动解释为中心失效，也不将通过标记解释为已知定位准确。旧规则 `supported_shape` 与旧汇总 `n_shape_supported` 明确保留。

重分类由 run_one_transition.py 的 `--reclassify-from` 完成，只生成新结果目录，不重新估计中心或重跑化学与模拟。审计中的 `center_rule_version` 与运行记录标记规则版本；新旧规则交集支持直接追踪此次改动。原数值基线仍核对旧字段，新增字段需要在新的私有基线中单独保存。

## 精简范围

不包含 B4 水文中心预测、监测报警、漂移识别、旧版 B5、历史实验副本、作者本机路径审计、论文编译文件或图像检查文件。保留原始数据的 95 cm 化学敏感性、趋势模型与单变点、固定阈值及固定深度化学对照，因为本文确实使用它们。

保留原模拟的子随机流与抽样顺序。它们保证冻结结果中的配对剖面可重放，并非可任意删除的冗余。
