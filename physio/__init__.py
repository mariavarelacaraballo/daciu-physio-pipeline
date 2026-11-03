"""Multimodal wearable physiology pipeline (BVP, EDA, skin temperature, EMG, IMU)."""
from . import bvp, eda, emg, features, imu, io, ml, temperature  # noqa: F401
from .features import Recording, extract_features  # noqa: F401

__version__ = "1.0.0"
