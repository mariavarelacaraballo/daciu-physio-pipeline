"""Peripheral skin temperature: NTC conversion, sliding slope and quality gates."""
from __future__ import annotations

import numpy as np


def ntc_uv_to_celsius(raw_uv, vref: float = 3.3, r_divider: float = 56e3,
                      r25: float = 10e3, beta: float = 3435.0) -> np.ndarray:
    """10 kΩ NTC thermistor (β = 3435 K) in a voltage divider -> °C (β-parameter model).

    Note: `r_divider` is not documented publicly for the VersaBio; check it
    against a reference thermometer before trusting absolute values.
    """
    v = np.clip(np.asarray(raw_uv, dtype=float) * 1e-6, 1e-6, vref - 1e-6)
    r_ntc = np.clip(r_divider * v / (vref - v), 1e-6, None)
    t_k = 1.0 / (1.0 / 298.15 + np.log(r_ntc / r25) / beta)
    return t_k - 273.15


def sliding_slope(temp_c: np.ndarray, fs: float, win_s: float = 60.0, step_s: float = 30.0,
                  start_s: float = 300.0):
    """Least-squares slope per window, in m°C/s. Skin temperature is slow, so
    windows are long; the first 5 min are skipped (sensor thermal settling)."""
    t = np.arange(len(temp_c)) / fs
    centres, slopes = [], []
    t0 = start_s
    while t0 + win_s <= t[-1]:
        i0, i1 = int(t0 * fs), int((t0 + win_s) * fs)
        slopes.append(np.polyfit(t[i0:i1], temp_c[i0:i1], 1)[0] * 1000)
        centres.append(t0 + win_s / 2)
        t0 += step_s
    return np.asarray(centres), np.asarray(slopes)


def quality_gates(temp_c: np.ndarray, fs: float, bio_range=(25.0, 40.0),
                  rac_win_s: float = 2.0, rac_frac: float = 0.20) -> np.ndarray:
    """Mask of bad samples: outside the biological skin range, or a rapid
    amplitude change (sensor detached / touched)."""
    temp_c = np.asarray(temp_c, dtype=float)
    bad = (temp_c < bio_range[0]) | (temp_c > bio_range[1])
    w = int(rac_win_s * fs)
    if 0 < w < len(temp_c):
        bad[w:] |= np.abs(temp_c[w:] - temp_c[:-w]) > rac_frac * np.ptp(temp_c)
    return bad
