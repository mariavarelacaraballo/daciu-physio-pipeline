import numpy as np
import pytest

from physio import bvp, eda, emg, extract_features, imu, synthetic, temperature


@pytest.fixture(scope="module")
def rec():
    return synthetic.make_session(seed=1)


def test_beat_detection_matches_ground_truth(rec):
    peaks = bvp.detect_beats(bvp.filter_bvp(rec.bvp, rec.fs_bvp), rec.fs_bvp)
    assert abs(len(peaks) - len(rec.meta["beats"])) / len(rec.meta["beats"]) < 0.02


def test_rr_filter_removes_implausible_intervals():
    fs = 100
    peaks = np.array([0, 80, 160, 170, 250, 330, 600, 680]) * 1  # 0.1 s and 2.7 s gaps are artefacts
    _, rr = bvp.clean_rr(peaks, fs)
    assert np.all((rr > 0.3) & (rr < 2.0))


def test_hrv_known_values():
    rr = np.array([0.8, 0.9, 0.8, 0.9, 0.8, 0.9])
    h = bvp.hrv_time_domain(rr)
    assert h["rmssd_ms"] == pytest.approx(100.0)
    assert h["hr_mean"] == pytest.approx(60 / 0.85)


def test_spectral_entropy_separates_pulse_from_noise():
    fs = 64
    t = np.arange(0, 60, 1 / fs)
    pulse = np.sin(2 * np.pi * 1.2 * t)
    noise = np.random.default_rng(0).normal(size=t.size)
    assert bvp.spectral_entropy_gate(pulse, fs)[2].mean() > 0.9
    assert bvp.spectral_entropy_gate(noise, fs)[2].mean() < 0.1


def test_eda_decomposition_sums_to_signal(rec):
    tonic, phasic = eda.decompose(rec.eda_us, rec.fs_eda)
    np.testing.assert_allclose(tonic + phasic, rec.eda_us)


def test_eda_calibration_is_monotonic():
    g = eda.versabio_uv_to_us([0.5e6, 1.0e6, 2.0e6])
    assert np.all(np.diff(g) > 0)  # G grows monotonically with V for this divider
    assert np.all(g > 0)


def test_eda_contact_gate():
    x = np.full(400, 2.0)
    x[100:120] = 0.01
    assert eda.quality_gates(x, fs=4)[100:120].all()


def test_ntc_conversion_at_25c():
    # At 25 °C the NTC equals R25 = 10 kΩ -> V = Vref * R / (R + R_div)
    v = 3.3 * 10e3 / (10e3 + 56e3)
    assert temperature.ntc_uv_to_celsius(v * 1e6) == pytest.approx(25.0, abs=1e-6)


def test_emg_filter_removes_mains_and_drift():
    fs = 256
    t = np.arange(0, 10, 1 / fs)
    x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 0.3 * t)
    assert np.std(emg.filter_emg(x, fs)[fs:-fs]) < 0.05


def test_imu_finds_movements(rec):
    ev = imu.motion_events(rec.acc_mag, rec.fs_acc, units="g")
    for true_t in rec.meta["movements"]:
        assert np.min(np.abs(ev - true_t)) < 1.0


def test_features_reflect_stress(rec):
    df = extract_features(rec, label_fn=synthetic.block_of)
    m = df[df["IMU_ok"]].groupby("label").mean(numeric_only=True)
    assert m.loc["stress", "HR_mean"] > m.loc["rest", "HR_mean"]
    assert m.loc["stress", "RMSSD_ms"] < m.loc["rest", "RMSSD_ms"]
    assert m.loc["stress", "SCL_mean"] > m.loc["rest", "SCL_mean"]
    assert m.loc["stress", "Temp_slope"] < 0 < m.loc["recovery", "Temp_slope"]
    assert (~df["IMU_ok"]).sum() >= len(rec.meta["movements"])
