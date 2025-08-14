import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from datasets import WindowDataset
from models import TransformerDecoder, RNNDecoder
from visualization import plot_confusion

def main():
    out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load("data/processed/windows.npz")
    X, y = data["X"], data["y"]
    ds = WindowDataset(X, y)
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    # load best model if exists
    ckpt = None
    for name in ["transformer_best.pt", "rnn_best.pt"]:
        p = out_dir / name
        if p.exists():
            ckpt = p; break

    if ckpt is None:
        print("No checkpoint found in results/. Run train.py first.")
        return

    if "transformer" in ckpt.name:
        model = TransformerDecoder(in_ch=X.shape[-1])
    else:
        model = RNNDecoder(in_ch=X.shape[-1])

    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    all_p = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb)
            all_p.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_pred = np.concatenate(all_p)
    print(classification_report(y, y_pred, target_names=["open","close","supinate","pronate","rest"]))
    plot_confusion(y, y_pred)

    # save summary
    with open(out_dir/"summary.txt", "w") as f:
        f.write("Checkpoint: " + ckpt.name + "\n")
        f.write("See confusion_matrix.png and metrics.json for details.\n")
    print("Evaluation artifacts saved in results/")

if __name__ == "__main__":
    main()
