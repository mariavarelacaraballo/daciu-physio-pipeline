"""Loaders: Bitbrain VersaBio exports and the public WESAD dataset."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def read_versabio(path) -> pd.DataFrame:
    """Read a VersaBio export (analog / exg / imu / din).

    Exports come either as plain CSV or as .xlsx files that store the whole CSV
    inside a single Excel column; both are handled.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = wb.active.iter_rows(values_only=True)
    cols = next(rows)[0].split(",")
    df = pd.DataFrame([r[0].split(",") for r in rows if r[0] is not None], columns=cols)
    wb.close()
    return df.apply(pd.to_numeric, errors="coerce")


# WESAD labels: 0 undefined, 1 baseline, 2 stress (TSST), 3 amusement, 4 meditation
WESAD_LABELS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
WESAD_FS = {"label": 700, "chest": 700, "BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}


def load_wesad(root, subject: str = "S2") -> dict:
    """Load one WESAD subject (Schmidt et al., 2018). Download the dataset
    yourself; it is not redistributed here (non-commercial academic licence)."""
    with open(Path(root) / subject / f"{subject}.pkl", "rb") as f:
        return pickle.load(f, encoding="latin1")


def wesad_label_at(data: dict, t: float) -> int:
    lab = data["label"]
    i = min(int(t * WESAD_FS["label"]), len(lab) - 1)
    return int(lab[i])


def wesad_majority_label(data: dict, t0: float, t1: float) -> int:
    lab = np.asarray(data["label"][int(t0 * WESAD_FS["label"]):int(t1 * WESAD_FS["label"])])
    return int(np.bincount(lab).argmax()) if lab.size else 0
