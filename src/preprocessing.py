"""
Bandpass/normalize → windowing → save dataset arrays for training.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from fusion import fuse

OUT_NPZ = Path("data/processed/windows.npz")

def bandpass(x, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def make_windows(fused: pd.DataFrame, fs=1000, win_ms=200, hop_ms=50):
    win = int(win_ms * fs / 1000)
    hop = int(hop_ms * fs / 1000)
    X, y = [], []
    sig = fused.copy()

    # Basic filtering
    sig["emg"] = bandpass(sig["emg"].values, fs, 20, 450)
    sig["eeg"] = bandpass(sig["eeg"].fillna(method="ffill").fillna(0).values, fs, 1, 45)
    for c in ["acc", "gyr"]:
        sig[c] = sig[c].fillna(method="ffill").fillna(0).values

    # Z-score normalization per channel
    for c in ["emg", "eeg", "acc", "gyr"]:
        v = sig[c].values
        sig[c] = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)

    # Rolling windows
    V = sig[["emg", "eeg", "acc", "gyr"]].values
    labels = sig["label"].values.astype(int)
    for s in range(0, len(sig) - win, hop):
        e = s + win
        w = V[s:e]
        lab = np.bincount(labels[s:e], minlength=5).argmax()
        X.append(w)         # shape: (win, 4)
        y.append(lab)

    X = np.stack(X).astype(np.float32)  # (N, T, C=4)
    y = np.array(y, dtype=np.int64)
    return X, y

def main():
    fused = fuse()
    X, y = make_windows(fused, fs=1000, win_ms=200, hop_ms=50)
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, X=X, y=y)
    print("Saved windows:", OUT_NPZ.resolve(), "X:", X.shape, "y:", y.shape)

if __name__ == "__main__":
    main()
