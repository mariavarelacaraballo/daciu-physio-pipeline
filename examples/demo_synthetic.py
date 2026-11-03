"""End-to-end demo on a synthetic session. Regenerates the figures in docs/figures/.

    python examples/demo_synthetic.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physio  # noqa: E402
from physio import bvp, eda, emg, imu, synthetic  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"rest": "#4C78A8", "stress": "#E45756", "recovery": "#54A24B"}
INK, MUTED = "#222222", "#6b6b6b"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
})


def shade(ax):
    for name, t0, t1 in synthetic.BLOCKS:
        ax.axvspan(t0 / 60, t1 / 60, color=COLORS[name], alpha=0.07, lw=0)


rec = synthetic.make_session(seed=0)
dur = rec.duration

# ---- Processing -------------------------------------------------------------
bvp_f = bvp.filter_bvp(rec.bvp, rec.fs_bvp)
peaks = bvp.detect_beats(bvp_f, rec.fs_bvp)
t_rr, rr = bvp.clean_rr(peaks, rec.fs_bvp)
tonic, phasic = eda.decompose(rec.eda_us, rec.fs_eda)
scr, phasic_s = eda.detect_scr(phasic, rec.fs_eda, ref=slice(60 * 4, 240 * 4))
emg_f = emg.filter_emg(rec.emg, rec.fs_emg)
t_emg, emg_rms = emg.windowed_rms(emg_f, rec.fs_emg, 5.0)
t_mov = imu.motion_events(rec.acc_mag, rec.fs_acc, units="g")
df = physio.extract_features(rec, label_fn=synthetic.block_of)

# ---- Figure 1: processed signals -------------------------------------------
fig, axs = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
axs[0].plot(t_rr / 60, 60 / rr, color=INK, lw=0.8)
axs[0].set_ylabel("HR (bpm)")
axs[0].set_title("BVP → beats (AMPD) → cleaned RR → heart rate")
te = np.arange(len(tonic)) / rec.fs_eda / 60
axs[1].plot(te, rec.eda_us, color="#bbbbbb", lw=0.7, label="raw")
axs[1].plot(te, tonic, color=INK, lw=1.2, label="tonic (SCL)")
axs[1].scatter(te[scr], rec.eda_us[scr], s=7, color="#E45756", zorder=3, label="SCR")
axs[1].set_ylabel("EDA (µS)")
axs[1].set_title("EDA → tonic / phasic decomposition → SCR detection")
axs[1].legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
tt = np.arange(len(rec.temp_c)) / rec.fs_temp / 60
axs[2].plot(tt, rec.temp_c, color=INK, lw=0.9)
axs[2].set_ylabel("Skin T (°C)")
axs[2].set_title("Peripheral temperature (vasoconstriction under stress)")
axs[3].plot(t_emg / 60, emg_rms, color=INK, lw=0.9)
axs[3].set_ylabel("EMG RMS (µV)")
axs[3].set_title("EMG → 20–120 Hz band-pass + 50 Hz notch → RMS")
ta = np.arange(len(rec.acc_mag)) / rec.fs_acc / 60
axs[4].plot(ta, rec.acc_mag, color=INK, lw=0.6)
for tm in t_mov:
    for a in axs:
        a.axvline(tm / 60, color="#E45756", lw=0.8, ls=":")
axs[4].set_ylabel("|acc| (g)")
axs[4].set_title("IMU → movement events → windows flagged for BVP/EDA")
axs[4].set_xlabel("Time (min)")
for a in axs:
    shade(a)
handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in COLORS.values()]
fig.legend(handles, COLORS.keys(), loc="upper right", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT / "pipeline_signals.png", dpi=150)
plt.close(fig)

# ---- Figure 2: windowed features by condition -------------------------------
valid = df[df["IMU_ok"]]
feats = [("HR_mean", "HR (bpm)"), ("RMSSD_ms", "RMSSD (ms)"), ("SCL_mean", "SCL (µS)"),
         ("SCR_count", "SCRs / window"), ("Temp_slope", "Temp slope (m°C/s)"), ("EMG_rms", "EMG RMS (µV)")]
fig, axs = plt.subplots(2, 3, figsize=(11, 5.5))
order = list(COLORS)
for ax, (f, lab) in zip(axs.flat, feats):
    data = [valid.loc[valid["label"] == c, f].dropna() for c in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color=INK, lw=1.2))
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS[c]); patch.set_alpha(0.55); patch.set_edgecolor(COLORS[c])
    for i, d in enumerate(data, 1):
        ax.scatter(np.random.default_rng(i).normal(i, 0.06, len(d)), d, s=6, color=COLORS[order[i - 1]], alpha=0.8)
    ax.set_xticks(range(1, 4), order)
    ax.set_title(lab)
fig.suptitle("30 s windows (50 % overlap), movement-free only", x=0.01, ha="left", color=MUTED, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "features_by_condition.png", dpi=150)
plt.close(fig)

# ---- Figure 3: BVP beat detection zoom --------------------------------------
fig, ax = plt.subplots(figsize=(11, 2.8))
t0, t1 = 200, 215
tb = np.arange(len(bvp_f)) / rec.fs_bvp
m = (tb >= t0) & (tb < t1)
ax.plot(tb[m], bvp_f[m], color=INK, lw=1)
pk = peaks[(peaks / rec.fs_bvp >= t0) & (peaks / rec.fs_bvp < t1)]
ax.scatter(pk / rec.fs_bvp, bvp_f[pk], color="#E45756", s=22, zorder=3)
ax.set_xlabel("Time (s)")
ax.set_title("Filtered BVP with AMPD-detected systolic peaks")
ax.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "bvp_peaks.png", dpi=150)
plt.close(fig)

print(f"Detected {len(peaks)} beats (ground truth {len(rec.meta['beats'])}); "
      f"{(~df['IMU_ok']).sum()} of {len(df)} windows flagged by IMU")
print(df.groupby("label")[[f for f, _ in feats]].mean().round(2))
print(f"Figures written to {OUT}")
