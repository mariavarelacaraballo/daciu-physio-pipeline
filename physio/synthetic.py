"""Synthetic multimodal session for demos and tests (no real participant data).

Three 10-min blocks — rest, stress, recovery — with physiologically plausible
changes: HR up and HRV down under stress, higher SCL and more SCRs, cooling
fingertips (vasoconstriction), more trapezius EMG, plus a few movement artefacts.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .features import Recording

BLOCKS = [("rest", 0, 600), ("stress", 600, 1200), ("recovery", 1200, 1800)]


def block_of(t: float) -> str:
    for name, t0, t1 in BLOCKS:
        if t0 <= t < t1:
            return name
    return BLOCKS[-1][0]


def make_session(seed: int = 0, fs_bvp=64, fs_eda=4, fs_temp=4, fs_emg=256, fs_acc=32,
                 movement_times=(305.0, 905.0, 1410.0)) -> Recording:
    rng = np.random.default_rng(seed)
    dur = BLOCKS[-1][2]
    hr_target = {"rest": 68, "stress": 88, "recovery": 72}
    hrv_sd = {"rest": 0.045, "stress": 0.018, "recovery": 0.035}

    # --- BVP: beat train with block-dependent RR ---
    beats, t = [], 0.0
    rr_prev = 60 / hr_target["rest"]
    while t < dur:
        b = block_of(t)
        rr = 0.8 * rr_prev + 0.2 * (60 / hr_target[b]) + rng.normal(0, hrv_sd[b])
        rr = float(np.clip(rr, 0.45, 1.4))
        beats.append(t)
        t += rr
        rr_prev = rr
    tb = np.arange(int(dur * fs_bvp)) / fs_bvp
    bvp = np.zeros_like(tb)
    for tk in beats:
        d = tb - tk
        m = (d >= 0) & (d < 1.0)
        bvp[m] += np.exp(-((d[m] - 0.15) / 0.07) ** 2) + 0.35 * np.exp(-((d[m] - 0.42) / 0.10) ** 2)
    bvp += 0.3 * np.sin(2 * np.pi * 0.05 * tb) + rng.normal(0, 0.03, tb.size)

    # --- EDA (µS): tonic level + SCR events ---
    te = np.arange(int(dur * fs_eda)) / fs_eda
    scl = {"rest": 2.0, "stress": 3.2, "recovery": 2.4}
    eda = np.array([scl[block_of(x)] for x in te], dtype=float)
    eda = uniform_filter1d(eda, size=int(60 * fs_eda), mode="nearest")  # slow SCL transitions
    rate = {"rest": 1.0, "stress": 5.0, "recovery": 2.0}  # SCRs per minute
    for name, t0, t1 in BLOCKS:
        for ts in rng.uniform(t0, t1, int(rate[name] * (t1 - t0) / 60)):
            d = te - ts
            m = d >= 0
            eda[m] += rng.uniform(0.15, 0.4) * (np.exp(-d[m] / 4.0) - np.exp(-d[m] / 0.75))
    eda += rng.normal(0, 0.005, te.size)

    # --- Skin temperature (°C) ---
    tt = np.arange(int(dur * fs_temp)) / fs_temp
    slope = {"rest": 0.0004, "stress": -0.0015, "recovery": 0.0012}
    temp = 32.0 + np.cumsum([slope[block_of(x)] / fs_temp for x in tt]) + rng.normal(0, 0.01, tt.size)

    # --- EMG (µV) ---
    tm = np.arange(int(dur * fs_emg)) / fs_emg
    amp = np.array([{"rest": 8, "stress": 20, "recovery": 10}[block_of(x)] for x in tm[::fs_emg]])
    emg = np.repeat(amp, fs_emg)[: tm.size] * rng.normal(0, 1, tm.size)
    emg += 15 * np.sin(2 * np.pi * 50 * tm) + 40 * np.sin(2 * np.pi * 0.3 * tm)  # mains + drift

    # --- Accelerometer magnitude (g) with a few movements ---
    ta = np.arange(int(dur * fs_acc)) / fs_acc
    acc = 1.0 + rng.normal(0, 0.01, ta.size)
    for tmv in movement_times:
        m = np.abs(ta - tmv) < 1.5
        acc[m] += 1.2 * np.exp(-((ta[m] - tmv) / 0.4) ** 2)
        mb = np.abs(tb - tmv) < 4
        bvp[mb] += rng.normal(0, 1.5, mb.sum())  # motion artefact on BVP

    return Recording(bvp=bvp, fs_bvp=fs_bvp, eda_us=eda, fs_eda=fs_eda, temp_c=temp, fs_temp=fs_temp,
                     emg=emg, fs_emg=fs_emg, acc_mag=acc, fs_acc=fs_acc, acc_units="g",
                     meta={"beats": np.asarray(beats), "movements": np.asarray(movement_times)})
