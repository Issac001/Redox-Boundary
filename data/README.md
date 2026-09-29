# Input workbook (not distributed)

The manuscript reanalyses collaborator-provided Whole3.1.xlsx. Redistribution permission has not been confirmed. Obtain the workbook from the data provider; it is not downloaded or included by this repository.

Place it at data/Whole3.1.xlsx (Git-ignored), or pass --data /absolute/path/Whole3.1.xlsx. Never commit the workbook.

- Worksheet: Sheet2.
- Header: the second Excel row (pandas.read_excel(..., header=1)).
- Required column names: Case, Time, Depth, Eh, WL, No., Cycle, NO3, NH4, Fe, Mn.
- Time: days within each condition; Depth and WL: cm below column top; Eh: original workbook mV.
- Nitrogen concentrations: mg N/L; Fe and Mn: mg/L. Fe/Mn are elemental concentrations, not valence-specific measurements.
- Expected depths: 10,20,30,40,50,60,70,80,90,95 cm. The soil surface is at 9 cm.
- Keep chemical zeros and missing values distinct. Do not interpolate missing chemical depths.
- Missing cycle labels are forward-filled only within their condition and recorded in the local audit.
- The loader records the number of original/excluded rows and checks complete, consistent Eh profiles.
- Chemistry primary analysis uses depths <=90 cm; 95 cm is only a sensitivity analysis. Eh fits retain all ten depths.
- A different workbook may be analyzable but is not expected to reproduce the manuscript values.

Without the workbook, python code/run_one_transition.py --simulation-only reproduces the six numerical scenarios and paired grid diagnostics. It does not reproduce observed-data results.
