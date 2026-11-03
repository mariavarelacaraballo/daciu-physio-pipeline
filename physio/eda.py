"""EDA / GSR: calibration, tonic-phasic decomposition, quality gates and SCR detection."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks


def versabio_uv_to_us(raw_uv, r_fixed: float = 6.49e6, vref: float = 3.3) -> np.ndarray:
    """Convert the VersaBio GSR channel (µV across a voltage divider) to µS.

    G = 1 / R_skin,  R_skin = R_fixed * (Vref - V) / V.
    With this topology a higher voltage means a lower skin resistance, i.e. a
    higher conductance (G grows monotonically with V).
    """
    v = np.clip(np.asarray(raw_uv, dtype=float) / 1e6, 1e-9, vref - 1e-9)
    return 1e6 * v / (r_fixed * (vref - v))


def decompose(eda: np.ndarray, fs: float, tonic_cut: float = 0.05):
    """Tonic (SCL) = 2nd-order zero-phase low-pass at 0.05 Hz; phasic (SCR) = residual."""
    b, a = butter(2, tonic_cut / (fs / 2), btype="low")
    tonic = filtfilt(b, a, eda)
    return tonic, eda - tonic


def quality_gates(eda_us: np.ndarray, fs: float, min_us: float = 0.05,
                  rac_win_s: float = 2.0, rac_frac: float = 0.20) -> np.ndarray:
    """Boolean mask of *bad* samples.

    - Contact gate: conductance below 0.05 µS means the electrode lost contact.
    - RAC gate (rapid amplitude change): |x[n+2s] - x[n]| > 20 % of the dynamic
      range is faster than any physiological EDA change -> motion / pressure artefact.
    """
    eda_us = np.asarray(eda_us, dtype=float)
    bad = eda_us < min_us
    w = int(rac_win_s * fs)
    if 0 < w < len(eda_us):
        jump = np.abs(eda_us[w:] - eda_us[:-w]) > rac_frac * np.ptp(eda_us)
        bad[w:] |= jump
    return bad


def detect_scr(phasic: np.ndarray, fs: float, ref: slice | None = None,
               k_std: float = 1.0, min_dist_s: float = 5.0, skip_first_s: float = 60.0):
    """SCR peak indices on the smoothed phasic component.

    - Smoothing: 2nd-order low-pass at 0.5 Hz (removes residual high-frequency noise).
    - Adaptive threshold: k x std of the phasic signal in a rest reference segment.
    - Refractory distance between SCRs (5 s), first minute discarded while the
      electrode-skin interface settles.
    """
    b, a = butter(2, 0.5 / (fs / 2), btype="low")
    smooth = filtfilt(b, a, phasic)
    reference = smooth[ref] if ref is not None else smooth
    peaks, _ = find_peaks(smooth, height=k_std * reference.std(), distance=int(fs * min_dist_s))
    return peaks[peaks > skip_first_s * fs], smooth
