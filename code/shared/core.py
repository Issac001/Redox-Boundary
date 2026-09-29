"""Shared observation loading and threshold diagnostics for B1–B3."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Profiles:
    rows: pd.DataFrame
    meta: pd.DataFrame
    eh: np.ndarray
    depth: np.ndarray
    audit: dict


def load_profiles(path: Path, cases: list[str]) -> Profiles:
    rows = pd.read_excel(path, sheet_name="Sheet2", header=1,
                         na_values=["NA", "N/A", "na", ""])
    rows["Case"] = rows["Case"].astype(str).str.extract(r"(3\.[1234]|[12])", expand=False)
    for column in rows.columns.difference(["Case"]):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    audit = {"raw_rows": len(rows), "raw_cases": rows.groupby("Case").size().to_dict(),
             "excluded_rows": int((~rows.Case.isin(cases)).sum())}
    rows = rows[rows.Case.isin(cases)].copy()
    if rows[["Case", "Time", "Depth", "Eh", "WL", "No."]].isna().any().any():
        raise ValueError("Required observations are missing; do not silently drop them")
    if rows.duplicated(["Case", "Time", "Depth"]).any():
        raise ValueError("Duplicate profile/depth keys")
    grid = rows.pivot(index=["Case", "Time"], columns="Depth", values="Eh").sort_index()
    if grid.isna().any().any():
        raise ValueError("Incomplete Eh profiles require an explicit observation policy")
    group = rows.groupby(["Case", "Time"])
    if (group.WL.nunique() > 1).any() or (group["No."].nunique() > 1).any():
        raise ValueError("Inconsistent within-profile water level or identifier")
    meta = group.agg(WL=("WL", "first"), Cycle=("Cycle", "median"),
                     profile_no=("No.", "first")).reindex(grid.index).reset_index()
    meta["cycle_imputed"] = meta.Cycle.isna()
    meta["Cycle"] = meta.groupby("Case").Cycle.ffill()
    if meta.Cycle.isna().any():
        raise ValueError("Unknown initial cycle")
    meta["profile_id"] = meta.Case + "_" + meta.profile_no.astype(int).astype(str)
    audit.update(profiles=len(meta), depth_cm=grid.columns.tolist(),
                 cycle_forward_filled=meta.loc[meta.cycle_imputed, "profile_id"].tolist(),
                 chemical_observed_rows={c: int(rows[c].notna().sum()) for c in ["NO3", "NH4"]})
    return Profiles(rows, meta, grid.to_numpy(float), grid.columns.to_numpy(float), audit)


def crossings(y: np.ndarray, depth: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    """All linear crossings and exact exceedance thickness. Flat equality => ambiguous."""
    v = np.asarray(y) - threshold
    found, thickness = [], 0.0
    for i, gap in enumerate(np.diff(depth)):
        a, b = v[i:i + 2]
        if a == 0 and b == 0:
            return np.array([np.nan]), np.nan
        if a == 0:
            found.append(depth[i])
        if a * b < 0:
            fraction = -a / (b - a)
            found.append(depth[i] + gap * fraction)
            thickness += gap * (fraction if a > 0 else 1 - fraction)
        elif a >= 0 and b >= 0:
            thickness += gap
    if v[-1] == 0:
        found.append(depth[-1])
    return np.unique(found), float(thickness)
