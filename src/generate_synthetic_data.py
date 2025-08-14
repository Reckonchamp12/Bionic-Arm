"""
Generate synthetic EMG, EEG, and IMU motion streams.

Real dataset is proprietary to Dyne Research. This synthetic generator
mimics key temporal/statistical patterns for demo purposes ONLY.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_SAMPLES = 30_000                 # ~30s @ 1 kHz EMG; EEG downsampled later
FS_EMG = 1000
FS_EEG = 250                       # EEG lower sampling
FS_IMU = 200

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

def pink_noise(n, rng):
    """1/f noise via filtering white noise."""
    # simple approximation: cumulative sum of white noise (random walk), then detrend
    x = rng.normal(0, 1, n).astype(np.float32)
    y = np.cumsum(x)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return y

def emg_signal(n, fs, class_id):
    # class-dependent bursts; EMG is wideband (20–450 Hz). Use AM/FM + noise.
    t = np.arange(n)/fs
    base = np.sin(2*np.pi*(80 + 20*class_id)*t) * (0.2 + 0.05*class_id)
    bursts = (RNG.random(n) < (0.02 + 0.01*class_id)).astype(np.float32) * RNG.normal(1.5, 0.2, n)
    noise = 0.4*RNG.normal(0, 1, n)
    return (base + bursts + noise).astype(np.float32)

def eeg_signal(n, fs, class_id):
    # alpha/beta rhythms vary by class; add 1/f background
    t = np.arange(n)/fs
    alpha = np.sin(2*np.pi*10*t) * (0.8 if class_id in [0,4] else 0.3)
    beta  = np.sin(2*np.pi*20*t) * (0.8 if class_id in [1,2,3] else 0.3)
    back  = 0.5 * pink_noise(n, RNG)
    noise = 0.2 * RNG.normal(0, 1, n)
    return (alpha + beta + back + noise).astype(np.float32)

def imu_signal(n, fs, class_id):
    # IMU accel/gyro spikes on motion classes
    accel = 0.02 * np.cumsum(RNG.normal(0, 1, n)).astype(np.float32)
    gyro  = 0.02 * np.cumsum(RNG.normal(0, 1, n)).astype(np.float32)
    if class_id != 4:  # not rest
        accel += RNG.normal(0.2, 0.05, n).astype(np.float32)
        gyro  += RNG.normal(0.2, 0.05, n).astype(np.float32)
    return accel, gyro

def scheduled_classes(n, fs):
    # Cycle classes every few seconds with random transitions
    classes = [0,1,2,3,4]  # open, close, supinate, pronate, rest
    seg_len = int(2.5 * fs)
    schedule = np.zeros(n, dtype=np.int32)
    idx = 0
    while idx < n:
        c = classes[RNG.integers(0, len(classes))]
        L = min(seg_len + RNG.integers(-fs//2, fs//2), n - idx)
        schedule[idx:idx+L] = c
        idx += L
    return schedule

def resample_like(base_len, src_fs, dst_fs, x):
    # simple decimation or linear interpolation as needed
    import math
    duration = base_len/src_fs
    t_dst = np.linspace(0, duration, int(duration*dst_fs), endpoint=False)
    t_src = np.linspace(0, duration, base_len, endpoint=False)
    return np.interp(t_dst, t_src, x).astype(np.float32)

def main():
    # master timeline = EMG
    cls = scheduled_classes(N_SAMPLES, FS_EMG)

    # EMG: one channel (demo); could be multi-channel with stacking
    emg = np.array([emg_signal(N_SAMPLES, FS_EMG, c) for c in cls]).reshape(-1)

    # EEG & IMU on their native rates, then export their own CSVs
    n_eeg = int(N_SAMPLES * FS_EEG / FS_EMG)
    n_imu = int(N_SAMPLES * FS_IMU / FS_EMG)
    cls_eeg = resample_like(N_SAMPLES, FS_EMG, FS_EEG, cls.astype(np.float32)).round().astype(int)
    cls_imu = resample_like(N_SAMPLES, FS_EMG, FS_IMU, cls.astype(np.float32)).round().astype(int)

    eeg = np.zeros(n_eeg, dtype=np.float32)
    for k in range(5):
        mask = (cls_eeg == k)
        eeg[mask] = eeg_signal(mask.sum(), FS_EEG, k)

    acc = np.zeros(n_imu, dtype=np.float32)
    gyr = np.zeros(n_imu, dtype=np.float32)
    for k in range(5):
        mask = (cls_imu == k)
        a, g = imu_signal(mask.sum(), FS_IMU, k)
        acc[mask] = a
        gyr[mask] = g

    # write CSVs
    pd.DataFrame({"t": np.arange(N_SAMPLES)/FS_EMG, "emg": emg, "label": cls}).to_csv(OUT/"emg_sample.csv", index=False)
    pd.DataFrame({"t": np.arange(len(eeg))/FS_EEG, "eeg": eeg}).to_csv(OUT/"eeg_sample.csv", index=False)
    pd.DataFrame({"t": np.arange(len(acc))/FS_IMU, "acc": acc, "gyr": gyr}).to_csv(OUT/"motion_sample.csv", index=False)
    print(f"Wrote synthetic data to {OUT.resolve()}")

if __name__ == "__main__":
    main()
