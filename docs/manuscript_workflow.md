# Current manuscript: definitions and output map

The executable scope is fixed to the 2026-11-11 revision: three methods, four simulations, chemical center comparison and seven figures. `python code/run_manuscript.py` performs the complete analysis rather than a smoke subset or selection from historical results.

## Scientific definitions

The sigmoid is `a + A * expit((c - z) / w)`, with A >= 0. Center c is both its half-amplitude position and maximum negative gradient position. The logistic scale w is not the whole transition-band thickness: the 10–90% span is `2*log(9)*w`. Thus w = 8 cm corresponds to a span of approximately 35.2 cm. Error-function and Gompertz shapes have that same span. Their true positions are the maximum negative gradient positions; Gompertz half-amplitude and maximum-gradient locations differ.

The Eh comparison uses center bounds 20–90 cm, a 0.1 cm sigmoid search grid, scale candidates 1, 2, 3, 5, 8, 12, 18 cm, and at least two measurements on each side of an admissible step gap. Both frequentist methods use profile-SSE working intervals at nominal 95% and working SD 25 mV. Bayesian intervals and predictions are conditional on one descending step, with ordered plateau priors derived from independent N(0,500²) variables. The method averages over possible boundary locations, including location uncertainty inside unsampled gaps.

Only sigmoid receives the current center diagnostic: finite center strictly inside the candidate support, amplitude >=100 mV, and interval length <=20 cm. Width at a search limit is reported separately. Diagnostics do not remove observations, profiles, depths or simulation replicates from comparisons.

LODO removes one depth from a profile, refits all methods on the remaining observations and predicts at the omitted depth. RMSE pools equal held-depth weights within each profile and equal profile weights; ALL pools 223 profiles rather than giving each case equal total weight. The two endpoint depths are 10 and 95 cm. LS step predicts the deep plateau at its selected gap midpoint and below.

Chemical profiles retain z <=90 cm and at least five analyte-specific observed depths. Fitting uses log1p concentration, descending NO3 and ascending NH4/Fe/Mn, center candidates 15–90 cm by 1 cm and the same width candidates. Numerical amplitude and within-observation contrast checks reject degenerate fits. One-sided or extrapolated estimates are retained with flags. Chemical and Eh centers match only by case and the same recorded time (absolute tolerance 1e-9 day); nearest-time information is diagnostic, never a substitute pairing.

Pooled Figure 2.1 fits one descriptive curve to all 2,230 Eh measurements. Since all profiles have the same ten depths, minimizing squared error is equivalent to fitting their depth means. RMSE and R² are then evaluated against all observations. The pooled curve is not used in individual-profile estimation, LODO, simulations or chemical fits.

## Outputs used by the paper

| Paper component | Output |
| --- | --- |
| Eh centers and diagnostics; Figures 1–2 | `numeric/sigmoid_profiles_current.csv`, `center_diagnostics.csv`, `all_method_centers.csv`, `representative_profiles.csv` |
| Reconstruction; Figure 3 | `numeric/lodo_regions.csv`, `lodo_by_depth.csv`, `lodo_depth_winners.csv`, `lodo_win_counts.csv` |
| Four-scenario localization; Figure 4 | `numeric/simulation_four_scenarios.csv` |
| Chemical centers; Figure 5 | `numeric/chemical_transition_centers.csv`, `chemical_eh_exact_matches.csv`, `chemical_eh_differences.csv` |
| Cause analysis in results/discussion | `numeric/mechanism_statistics.json`; `raw/real_omission_stability.csv` |
| Pooled shape illustration; Figure 2.1 | `numeric/pooled_fit_parameters.json`, `pooled_sigmoid_predictions.csv` |
| Threshold ambiguity; Figure C.1 | Threshold records and summaries generated in `numeric/` by `manuscript_figures_data.py` |
| Individual audit/reanalysis | All fitted centers, LODO predictions, gap posteriors, simulated observations and replicate metrics under `raw/` |

The four reported simulation metrics are center RMSE in cm, coverage in percent, coverage MCSE in percentage points, and median interval length in cm. Coverage includes both interval endpoints and all 500 replicates. Simulated centers are drawn continuously between 30 and 80 cm; Gaussian noise SD is 25 mV and plateau amplitude is 800 mV. Stable seeds are derived from master seed 20261018 and the scenario identifier, making random streams independent of catalog order.

Examples are selected by chronological middle record in each case, taking the later middle when the number is even. Figure typography and tick formatting are centralized in `plot_manuscript_figures.py`; full-precision results are never rounded before analysis.

## Numerical acceptance

The 2026-11-12 refactor was checked against the established formal results: 669 method/profile fits, 6,690 LODO predictions, 6,000 simulated method/replicate records, 140 chemical centers and the current aggregate tables. Differences were within numerical tolerance, with no change in reported rankings, center diagnoses or coverage. The portable default run repeats checks on five table families using `data/manuscript/expected_results.json`.

The run writes a completion manifest only after calculation, required reference verification and requested image rendering succeed. The input configuration may vary only the seed and replicate count; changes to model/depth/case definitions are rejected rather than generating mislabeled manuscript figures. The frozen reference checksum is verified and recorded. Nonempty output directories are refused. Existing manuscript files are not edited by this numerical reproduction command.
