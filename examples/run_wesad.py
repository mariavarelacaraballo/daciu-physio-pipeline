"""Run the pipeline on the public WESAD dataset (wrist E4 + chest EMG) and test
baseline-vs-stress classification across subjects.

1. Download WESAD: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
2. python examples/run_wesad.py /path/to/WESAD [S2 S3 ...]

Leave-subjects-out CV (GroupKFold) + permutation test: the model never sees the
test subject during training.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physio import Recording, extract_features  # noqa: E402
from physio.features import zscore_per_subject  # noqa: E402
from physio.io import WESAD_FS, WESAD_LABELS, load_wesad, wesad_majority_label  # noqa: E402
from physio.ml import default_models, evaluate  # noqa: E402

ALL_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]  # S1 and S12 do not exist
WIN, STEP = 30.0, 15.0
FEATURES = ["HR_mean", "RMSSD_ms", "SCL_mean", "SCL_slope", "SCR_count", "Temp_mean", "Temp_slope", "EMG_rms"]
GROUPS = {
    "All features": FEATURES,
    "Without cardiac (wrist BVP)": [f for f in FEATURES if f not in ("HR_mean", "RMSSD_ms")],
    "Cardiac only (HR, RMSSD)": ["HR_mean", "RMSSD_ms"],
    "EDA only": ["SCL_mean", "SCL_slope", "SCR_count"],
    "Skin temperature only": ["Temp_mean", "Temp_slope"],
    "EMG only": ["EMG_rms"],
}


def ecg_hr_reference(ecg, fs=700):
    """Chest-ECG heart rate, used only to audit the wrist BVP estimate."""
    b, a = butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
    e = filtfilt(b, a, ecg)
    r, _ = find_peaks(e, distance=int(0.33 * fs), height=0.4 * np.percentile(e, 98))
    return (r[1:] + r[:-1]) / 2 / fs, np.diff(r) / fs


def subject_features(root, sid):
    d = load_wesad(root, sid)
    w, c = d["signal"]["wrist"], d["signal"]["chest"]
    acc_g = np.linalg.norm(w["ACC"], axis=1) / 64.0  # E4 ACC: 1/64 g per count
    rec = Recording(bvp=w["BVP"].ravel(), fs_bvp=WESAD_FS["BVP"],
                    eda_us=w["EDA"].ravel(), fs_eda=WESAD_FS["EDA"],
                    temp_c=w["TEMP"].ravel(), fs_temp=WESAD_FS["TEMP"],
                    emg=c["EMG"].ravel(), fs_emg=WESAD_FS["chest"],
                    acc_mag=acc_g, fs_acc=WESAD_FS["ACC"], acc_units="g")
    label = lambda tc: WESAD_LABELS.get(wesad_majority_label(d, tc - WIN / 2, tc + WIN / 2), "other")
    df = extract_features(rec, win_s=WIN, step_s=STEP, label_fn=label)
    t_e, rr_e = ecg_hr_reference(c["ECG"].ravel())
    df["HR_ecg"] = [60 / rr_e[(t_e >= tc - WIN / 2) & (t_e < tc + WIN / 2)].mean() for tc in df["t_centre"]]
    df["subject"] = sid
    return df


def main():
    root = Path(sys.argv[1])
    subjects = sys.argv[2:] or ALL_SUBJECTS
    frames = []
    for sid in subjects:
        print(f"processing {sid} ...", flush=True)
        frames.append(subject_features(root, sid))
    df = pd.concat(frames, ignore_index=True)
    audit = df[df["label"].isin(["baseline", "stress"])].dropna(subset=["HR_mean", "HR_ecg"])
    audit = audit.assign(abs_err=(audit["HR_mean"] - audit["HR_ecg"]).abs())
    print("Wrist-BVP HR vs chest-ECG HR, mean absolute error (bpm):")
    print(audit.groupby("label")["abs_err"].agg(["mean", "median", "count"]).round(1))
    df = df[df["label"].isin(["baseline", "stress"]) & df["IMU_ok"]].dropna(subset=FEATURES)
    df = zscore_per_subject(df, FEATURES)
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "wesad_features.csv", index=False)
    audit.groupby("label")["abs_err"].agg(["mean", "median", "count"]).to_csv(out / "wesad_hr_audit.csv")

    y = (df["label"] == "stress").astype(int)
    res = evaluate(df[FEATURES], y, groups=df["subject"], n_permutations=100)
    print(res.round(3).to_string(index=False))
    res.to_csv(out / "wesad_results.csv", index=False)

    # Ablation: which modalities carry the signal? (Random Forest only)
    rf = {"Random Forest": default_models()["Random Forest"]}
    rows = []
    for name, cols in GROUPS.items():
        r = evaluate(df[cols], y, groups=df["subject"], models=rf, n_permutations=50)
        rows.append({"features": name, "balanced_accuracy": r.iloc[0]["balanced_accuracy"],
                     "chance": r.iloc[0]["chance"], "p_value": r.iloc[0]["p_value"]})
    abl = pd.DataFrame(rows)
    print(abl.round(3).to_string(index=False))
    abl.to_csv(out / "wesad_ablation.csv", index=False)
    plot(audit, abl, len(subjects))


def plot(audit, abl, n_subjects):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
    for lab, col in [("baseline", "#4C78A8"), ("stress", "#E45756")]:
        d = audit[audit["label"] == lab]
        a1.scatter(d["HR_ecg"], d["HR_mean"], s=6, alpha=0.5, color=col, label=lab)
    lim = [40, 140]
    a1.plot(lim, lim, color="#888", lw=0.8, ls="--")
    a1.set(xlim=lim, ylim=lim, xlabel="HR from chest ECG (bpm)", ylabel="HR from wrist BVP (bpm)")
    a1.set_title("Wrist BVP vs ECG, 30 s windows", loc="left", fontweight="bold")
    a1.legend(frameon=False)
    abl = abl.iloc[::-1]
    a2.barh(abl["features"], abl["balanced_accuracy"], color="#4C78A8", height=0.6)
    a2.scatter(abl["chance"], abl["features"], color="#222", marker="|", s=200, label="permutation chance", zorder=3)
    for y_, v in enumerate(abl["balanced_accuracy"]):
        a2.text(v + 0.01, y_, f"{v:.2f}", va="center", fontsize=9)
    a2.set(xlim=(0.4, 1.0), xlabel="Balanced accuracy (leave-subjects-out)")
    a2.set_title("Baseline vs stress: modality ablation", loc="left", fontweight="bold")
    a2.legend(frameon=False, loc="lower right")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"WESAD, {n_subjects} subjects", x=0.01, ha="left", color="#666", fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1] / "docs" / "figures" / "wesad_results.png", dpi=150)


if __name__ == "__main__":
    main()
