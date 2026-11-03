"""Surface EMG: band-pass + mains notch, windowed RMS."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def filter_emg(emg: np.ndarray, fs: float, band=(20.0, 120.0), mains_hz: float = 50.0,
               q: float = 30.0) -> np.ndarray:
    """4th-order Butterworth band-pass (removes motion drift < 20 Hz) followed by
    a notch at the mains frequency (50 Hz in Europe). Zero-phase."""
    hi = min(band[1], 0.99 * fs / 2)
    b, a = butter(4, [band[0] / (fs / 2), hi / (fs / 2)], btype="band")
    out = filtfilt(b, a, np.asarray(emg, dtype=float))
    if mains_hz < fs / 2:
        bn, an = iirnotch(mains_hz, q, fs)
        out = filtfilt(bn, an, out)
    return out


def windowed_rms(x: np.ndarray, fs: float, win_s: float = 1.0):
    """Non-overlapping RMS (amplitude of muscle activation). Returns (t, rms)."""
    w = int(fs * win_s)
    n = len(x) // w
    seg = np.asarray(x[: n * w]).reshape(n, w)
    return (np.arange(n) + 0.5) * win_s, np.sqrt(np.mean(seg ** 2, axis=1))
