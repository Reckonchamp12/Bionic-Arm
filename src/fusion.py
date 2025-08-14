"""
Fuse raw EMG/EEG/IMU streams into a single, time-aligned dataframe.
"""
import pandas as pd
from pathlib import Path
import numpy as np

def fuse(raw_dir="data/raw", out_csv="data/processed/fused_signals.csv"):
    raw_dir = Path(raw_dir)
    emg = pd.read_csv(raw_dir/"emg_sample.csv")     # columns: t, emg, label
    eeg = pd.read_csv(raw_dir/"eeg_sample.csv")     # columns: t, eeg
    mot = pd.read_csv(raw_dir/"motion_sample.csv")  # columns: t, acc, gyr

    # resample eeg & mot onto emg timeline (simple nearest merge-asof)
    emg = emg.sort_values("t")
    eeg = eeg.sort_values("t")
    mot = mot.sort_values("t")

    eeg_on_emg = pd.merge_asof(emg[["t"]], eeg, on="t", direction="nearest")
    mot_on_emg = pd.merge_asof(emg[["t"]], mot, on="t", direction="nearest")

    fused = pd.concat(
        [emg[["t", "emg", "label"]], eeg_on_emg[["eeg"]], mot_on_emg[["acc", "gyr"]]],
        axis=1
    )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(out_csv, index=False)
    return fused

if __name__ == "__main__":
    df = fuse()
    print("Fused shape:", df.shape)
