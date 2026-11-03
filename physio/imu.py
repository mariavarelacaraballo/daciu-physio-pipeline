"""IMU as a motion-artefact detector for the other signals.

Movement corrupts BVP (pulse shape) and EDA (electrode pressure). Instead of
trying to clean those signals, the accelerometer flags the windows to discard.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

G = 9.81


def magnitude(acc_xyz: np.ndarray) -> np.ndarray:
    """Euclidean norm of a (N, 3) accelerometer array."""
    acc_xyz = np.asarray(acc_xyz, dtype=float)
    return np.sqrt((acc_xyz ** 2).sum(axis=1))


def motion_events(acc_mag: np.ndarray, fs: float, k_g: float = 0.5, min_dist_s: float = 2.0,
                  units: str = "auto") -> np.ndarray:
    """Times (s) of movement peaks: |a| above median + 0.5 g.

    Peak-based detection replaced two earlier threshold rules (mean + 2σ of the
    baseline, then median + 3·MAD): isolated movements were diluted inside
    window-level RMS, but they are exactly what ruins a BVP window.
    `units`: "g", "ms2" or "auto" (decided from the median, ≈1 g at rest).
    """
    acc_mag = np.asarray(acc_mag, dtype=float)
    med = np.median(acc_mag)
    if units == "auto":
        units = "ms2" if med > 3 else "g"
    g = G if units == "ms2" else 1.0
    peaks, _ = find_peaks(acc_mag, height=med + k_g * g, distance=int(fs * min_dist_s))
    return peaks / fs


def clean_windows(event_times: np.ndarray, duration_s: float, win_s: float = 30.0,
                  step_s: float = 15.0):
    """For each sliding window, True if it contains no movement event. Returns (t_centre, ok)."""
    centres, ok = [], []
    t0 = 0.0
    while t0 + win_s <= duration_s:
        centres.append(t0 + win_s / 2)
        ok.append(not np.any((event_times >= t0) & (event_times < t0 + win_s)))
        t0 += step_s
    return np.asarray(centres), np.asarray(ok)
