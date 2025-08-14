# Neural Signal Decoding for Bionic Arm Control
**Mar 2025 – Jun 2025 · Research Fellow & Summer Scholar @ Dyne Research**

Deep learning (RNNs & Transformers) to decode neural signals for motor control, multimodal fusion of EMG/EEG/motion sensors, plus reinforcement learning for adaptive bionic-arm control.

> **Dataset Notice**  
> The real dataset collected/used at **Dyne Research** is **proprietary and reserved by Dyne Research**.  
> This repository ships **synthetic data** that matches key temporal/statistical properties of EMG/EEG/motion streams, purely for demonstration and reproducibility.

---

## Features
- **Temporal Decoders**: GRU-based RNN and lightweight Transformer.
- **Multimodal Fusion**: EMG + EEG + IMU (accel/gyro) aligned and windowed.
- **Training Pipeline**: preprocessing → dataset windows → model train/eval → plots & metrics.
- **Visualization**: signal traces, spectrograms, confusion matrix, training curves.
- **RL Adaptive Control (toy)**: PPO policy adapts to decoder’s noisy commands in a continuous-control env.

---

## Quick Start
```bash
# 1) Create environment (optional)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Generate synthetic data
python src/generate_synthetic_data.py

# 4) Preprocess & build windows (fusion + labels)
python src/preprocessing.py

# 5) Train (Transformer by default). Artifacts go to results/
python src/train.py --model transformer --epochs 20

# 6) Evaluate + plots
python src/evaluate.py

# 7) (Optional) Run RL adaptive control demo
python src/rl_control.py --decoder_ckpt results/transformer_best.pt

bionic_arm_control/
│
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ data/
│  ├─ raw/
│  │  ├─ emg_sample.csv
│  │  ├─ eeg_sample.csv
│  │  └─ motion_sample.csv
│  └─ processed/
│     ├─ fused_signals.csv
│     └─ windows.npz
│
├─ results/
│  ├─ signals.png
│  ├─ spectrogram_emg.png
│  ├─ training_curve.png
│  ├─ confusion_matrix.png
│  ├─ metrics.json
│  ├─ rl_training_curve.png
│  └─ transformer_best.pt
│
└─ src/
   ├─ __init__.py
   ├─ generate_synthetic_data.py
   ├─ preprocessing.py
   ├─ datasets.py
   ├─ models.py
   ├─ fusion.py
   ├─ train.py
   ├─ evaluate.py
   ├─ visualization.py
   └─ rl_control.py

