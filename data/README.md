# Current-manuscript reproduction inputs

`manuscript/` contains only the inputs and numerical reference required for the 2026-11-11 manuscript. The project owner explicitly authorized uploading these reproduction data on 2026-11-12. The observations reanalyse the sand-column experiments of Zhang and Furman (2021).

| File | Contents | Role |
| --- | --- | --- |
| `eh_observations.csv` | 2,230 measurements; 223 complete ten-depth profiles | Model observations |
| `chemical_observations.csv` | 233 case/time/depth rows; 922 observed concentrations | Separate chemical-profile fits |
| `provenance.json` | Source-export and input hashes; conversion description; original workbook hash | Provenance and standard-run identification |
| `expected_results.json` | Selected columns of five frozen manuscript table families | Numerical regression reference only |

## Coordinates and units

- Cases 3.1, 3.2 and 3.3 have 77, 78 and 68 Eh profiles.
- `Time` is recorded time in days within a case. Different cases are separate experiments.
- `Depth` and `WL` are cm below the column top; the soil surface is at 9 cm.
- Eh depths are 10, 20, 30, 40, 50, 60, 70, 80, 90 and 95 cm; `Eh` is mV.
- `NO3` and `NH4` are mg N/L. `Fe` and `Mn` are filtered elemental concentrations in mg/L, not valence-specific measurements.
- Chemical observations have 13, 12 and 10 sampling events by case and are restricted to z <= 90 cm. The 95 cm outlet chemical measurements are not local depth samples in this analysis. Eh retains 95 cm.
- Empty chemical cells are missing. Zero is a valid observation; 373 zero concentrations are retained. Each analyte is fitted only with its own available depths; no interpolation is introduced.

## Preparation and provenance

The Eh CSV is a byte-identical copy of the established `B2_Eh_observations.csv` export. Metadata such as cycle identifiers is retained; it does not enter individual-profile model estimation.

The chemical CSV pivots the observed `B3_chemical_observations.csv` export from long analyte format to `Case,Time,Depth,NO3,NH4,Fe,Mn`. It introduces neither fitted concentrations nor interpolated observations. Source-export hashes and the original `Whole3.1.xlsx` hash are recorded in `provenance.json`; the original workbook and unrelated experiments are not required at runtime.

`expected_results.json` freezes current center diagnostics, region/depth RMSE, four-scenario metrics and chemical/Eh position differences. It is read only after estimates and predictions have been computed, to detect unintended changes. No reference center or reference prediction is used as model input.

Run `python code/run_manuscript.py --data-dir /path/to/csv_directory` to use another copy of the two observation files. A complete standard reproduction requires their recorded hashes and the canonical configuration. Modified inputs/settings are recorded as noncanonical runs.
