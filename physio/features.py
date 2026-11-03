"""Synchronised sliding-window feature matrix across all modalities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import bvp as _bvp
from . import eda as _eda
from . import emg as _emg
from . import imu as _imu
from . import temperature as _temp


@dataclass
class Recording:
    """One session. Each signal carries its own sampling rate; any can be None."""
    bvp: Optional[np.ndarray] = None
    fs_bvp: float = 0.0
    eda_us: Optional[np.ndarray] = None          # already in µS
    fs_eda: float = 0.0
    temp_c: Optional[np.ndarray] = None          # already in °C
    fs_temp: float = 0.0
    emg: Optional[np.ndarray] = None
    fs_emg: float = 0.0
    acc_mag: Optional[np.ndarray] = None
    fs_acc: float = 0.0
    acc_units: str = "auto"
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        d = [len(x) / fs for x, fs in [(self.bvp, self.fs_bvp), (self.eda_us, self.fs_eda),
                                       (self.temp_c, self.fs_temp), (self.emg, self.fs_emg),
                                       (self.acc_mag, self.fs_acc)] if x is not None]
        return min(d)


def _seg(x, fs, t0, t1):
    return x[int(t0 * fs):int(t1 * fs)]


def extract_features(rec: Recording, win_s: float = 30.0, step_s: float = 15.0,
                     label_fn: Optional[Callable[[float], str]] = None,
                     scr_ref_s=(60.0, 240.0)) -> pd.DataFrame:
    """Process every signal once, then cut synchronised windows (default 30 s, 50 % overlap).

    Per window: HR, HR std, RMSSD, SDNN, SCL level & slope, SCR count, skin
    temperature mean & slope, EMG RMS, and an `IMU_ok` flag (no movement event).
    `label_fn(t_centre)` returns the condition label for the window, if known.
    """
    dur = rec.duration
    out = {}

    if rec.bvp is not None:
        clean = _bvp.filter_bvp(rec.bvp, rec.fs_bvp)
        peaks = _bvp.detect_beats(clean, rec.fs_bvp)
        out["t_rr"], out["rr"] = _bvp.clean_rr(peaks, rec.fs_bvp)

    if rec.eda_us is not None:
        tonic, phasic = _eda.decompose(rec.eda_us, rec.fs_eda)
        ref = slice(int(scr_ref_s[0] * rec.fs_eda), int(scr_ref_s[1] * rec.fs_eda))
        scr, _ = _eda.detect_scr(phasic, rec.fs_eda, ref=ref)
        out["tonic"], out["t_scr"] = tonic, scr / rec.fs_eda

    if rec.emg is not None:
        out["emg_f"] = _emg.filter_emg(rec.emg, rec.fs_emg)

    if rec.acc_mag is not None:
        out["t_mov"] = _imu.motion_events(rec.acc_mag, rec.fs_acc, units=rec.acc_units)

    rows = []
    t0 = 0.0
    while t0 + win_s <= dur:
        t1, tc = t0 + win_s, t0 + win_s / 2
        row = {"t_centre": tc}

        if "rr" in out:
            m = (out["t_rr"] >= t0) & (out["t_rr"] < t1)
            h = _bvp.hrv_time_domain(out["rr"][m])
            row.update(HR_mean=h["hr_mean"], RMSSD_ms=h["rmssd_ms"], SDNN_ms=h["sdnn_ms"],
                       HR_std=np.std(60 / out["rr"][m]) if m.sum() >= 5 else np.nan)

        if "tonic" in out:
            ton = _seg(out["tonic"], rec.fs_eda, t0, t1)
            tt = t0 + np.arange(len(ton)) / rec.fs_eda
            row.update(SCL_mean=ton.mean(), SCL_slope=np.polyfit(tt, ton, 1)[0],
                       SCR_count=int(np.sum((out["t_scr"] >= t0) & (out["t_scr"] < t1))))

        if rec.temp_c is not None:
            tmp = _seg(rec.temp_c, rec.fs_temp, t0, t1)
            tt = t0 + np.arange(len(tmp)) / rec.fs_temp
            row.update(Temp_mean=tmp.mean(), Temp_slope=np.polyfit(tt, tmp, 1)[0] * 1000)

        if "emg_f" in out:
            e = _seg(out["emg_f"], rec.fs_emg, t0, t1)
            row["EMG_rms"] = float(np.sqrt(np.mean(e ** 2)))

        if "t_mov" in out:
            row["IMU_ok"] = not np.any((out["t_mov"] >= t0) & (out["t_mov"] < t1))

        if label_fn is not None:
            row["label"] = label_fn(tc)
        rows.append(row)
        t0 += step_s
    return pd.DataFrame(rows)


def zscore_per_subject(df: pd.DataFrame, cols, subject_col: str = "subject") -> pd.DataFrame:
    """Normalise each feature within each subject, so models learn changes
    relative to a person's own physiology rather than between-person offsets."""
    df = df.copy()
    df[cols] = df.groupby(subject_col)[cols].transform(lambda c: (c - c.mean()) / (c.std() + 1e-12))
    return df
