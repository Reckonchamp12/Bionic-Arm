import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

RES = Path("results")
RES.mkdir(exist_ok=True, parents=True)

def plot_signals(fused_csv="data/processed/fused_signals.csv"):
    df = pd.read_csv(fused_csv).head(5000)
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(3,1,1); ax1.plot(df["emg"]); ax1.set_title("EMG")
    ax2 = plt.subplot(3,1,2); ax2.plot(df["eeg"]); ax2.set_title("EEG")
    ax3 = plt.subplot(3,1,3); ax3.plot(df["acc"], label="acc"); ax3.plot(df["gyr"], label="gyr"); ax3.set_title("IMU (acc/gyr)")
    ax3.legend()
    plt.tight_layout()
    fp = RES/"signals.png"; plt.savefig(fp, dpi=150); plt.close(fig)
    print("Saved", fp)

def plot_spectrogram_emg(fused_csv="data/processed/fused_signals.csv", fs=1000):
    df = pd.read_csv(fused_csv)
    x = df["emg"].values[:10000]
    fig = plt.figure(figsize=(10,4))
    plt.specgram(x, NFFT=256, Fs=fs, noverlap=128)
    plt.title("EMG Spectrogram")
    plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)")
    fp = RES/"spectrogram_emg.png"; plt.savefig(fp, dpi=150); plt.close(fig)
    print("Saved", fp)

def plot_training_curve(log_json="results/metrics.json"):
    if not Path(log_json).exists():
        print("No metrics.json found for training curve.")
        return
    with open(log_json, "r") as f:
        m = json.load(f)
    epochs = list(range(1, len(m["train_loss"])+1))
    fig = plt.figure(figsize=(7,4))
    plt.plot(epochs, m["train_loss"], marker="o", label="train")
    if "val_loss" in m: plt.plot(epochs, m["val_loss"], marker="o", label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss"); plt.legend()
    fp = RES/"training_curve.png"; plt.savefig(fp, dpi=150); plt.close(fig)
    print("Saved", fp)

def plot_confusion(y_true, y_pred, labels=("open","close","supinate","pronate","rest")):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    fp = RES/"confusion_matrix.png"; plt.savefig(fp, dpi=150); plt.close(fig)
    print("Saved", fp)
