"""BVP / PPG: filtering, beat detection, RR cleaning and HRV."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import filtfilt, firwin, welch
from scipy.stats import entropy


def filter_bvp(bvp: np.ndarray, fs: float, band=(0.5, 4.0), n_taps: int = 512) -> np.ndarray:
    """Zero-phase FIR band-pass (Hamming window) keeping the cardiac band (30-240 bpm)."""
    n_taps = min(n_taps, (len(bvp) - 1) // 3)
    n_taps -= (n_taps + 1) % 2  # odd length -> type I linear-phase FIR
    coef = firwin(n_taps, list(band), pass_zero=False, fs=fs, window="hamming")
    out = filtfilt(coef, 1.0, np.asarray(bvp, dtype=float))
    return out - out.mean()


def detect_beats(bvp_clean: np.ndarray, fs: float, max_period_s: float = 1.5) -> np.ndarray:
    """Systolic peak indices with AMPD (Scholkmann et al., 2012).

    AMPD needs no amplitude threshold, so it is robust to slow amplitude drift.
    `max_period_s` bounds the largest scale (1.5 s -> HR >= 40 bpm).
    """
    from pyampd.ampd import find_peaks  # optional dependency, imported lazily

    return np.asarray(find_peaks(bvp_clean, scale=int(fs * max_period_s)))


def clean_rr(peaks: np.ndarray, fs: float, rr_range=(0.3, 2.0), max_jump: float = 0.5,
             median_tol: float | None = 0.25, median_k: int = 9):
    """Physiological RR filter.

    1. Absolute range: 0.3-2.0 s (30-200 bpm).
    2. Successive-change rule: drop beats whose RR changes > 50 % w.r.t. the
       previous one (implausible without arrhythmia -> usually a missed/extra peak).
    3. Local-median rule: drop RR deviating > 25 % from the median of the
       surrounding 9 beats. Catches runs of missed beats (doubled RR) that rule 2
       lets through after the first one. Set `median_tol=None` to disable.

    Returns (t_rr, rr) with t_rr the midpoint between the two beats, in seconds.
    """
    t_peaks = np.asarray(peaks) / fs
    rr = np.diff(t_peaks)
    t_rr = (t_peaks[1:] + t_peaks[:-1]) / 2
    keep = (rr > rr_range[0]) & (rr < rr_range[1])
    rr, t_rr = rr[keep], t_rr[keep]
    if len(rr) > 1:
        jump = np.abs(np.diff(rr)) / rr[:-1]
        keep2 = np.concatenate([[True], jump < max_jump])
        rr, t_rr = rr[keep2], t_rr[keep2]
    if median_tol is not None and len(rr) >= median_k:
        med = median_filter(rr, size=median_k, mode="nearest")
        keep3 = np.abs(rr - med) / med < median_tol
        rr, t_rr = rr[keep3], t_rr[keep3]
    return t_rr, rr


def hrv_time_domain(rr: np.ndarray) -> dict:
    """RMSSD and SDNN in ms. Short windows (30 s) only make RMSSD reliable;
    SDNN is reported for completeness (Task Force 1996 recommends >= 5 min)."""
    rr = np.asarray(rr)
    if len(rr) < 5:
        return {"hr_mean": np.nan, "rmssd_ms": np.nan, "sdnn_ms": np.nan}
    return {
        "hr_mean": 60.0 / rr.mean(),
        "rmssd_ms": float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000),
        "sdnn_ms": float(np.std(rr) * 1000),
    }


def spectral_entropy_gate(bvp_clean: np.ndarray, fs: float, win_s: float = 4.0,
                          band=(0.1, 5.0), threshold: float = 0.89):
    """Quality gate (Böttcher et al., 2022): normalised spectral entropy of the
    PSD in `band` per window. A clean pulse concentrates energy around the heart
    rate (low entropy); noise spreads it (H -> 1). Returns (t, H, is_clean)."""
    win = int(win_s * fs)
    step = win // 2
    t, h = [], []
    for i in range(0, len(bvp_clean) - win + 1, step):
        f, psd = welch(bvp_clean[i:i + win], fs=fs, nperseg=win // 2)
        p = psd[(f >= band[0]) & (f <= band[1])]
        p = p / (p.sum() + 1e-12)
        h.append(entropy(p) / np.log(p.size) if p.size > 1 else 1.0)
        t.append((i + win / 2) / fs)
    h = np.asarray(h)
    return np.asarray(t), h, h <= threshold
