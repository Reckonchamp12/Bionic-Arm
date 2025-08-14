import argparse, json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from datasets import WindowDataset
from models import RNNDecoder, TransformerDecoder
from visualization import plot_training_curve

def set_seed(s=123):
    torch.manual_seed(s); np.random.seed(s)

def load_data(npz_path="data/processed/windows.npz"):
    d = np.load(npz_path)
    return d["X"], d["y"]

def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    losses = []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return np.mean(losses)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    losses = []
    crit = nn.CrossEntropyLoss()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = crit(logits, y)
        p = torch.argmax(logits, dim=1)
        losses.append(float(loss.item()))
        all_y.append(y.cpu().numpy()); all_p.append(p.cpu().numpy())
    y_true = np.concatenate(all_y); y_pred = np.concatenate(all_p)
    acc = accuracy_score(y_true, y_pred)
    return np.mean(losses), acc, y_true, y_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rnn","transformer"], default="transformer")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--npz", default="data/processed/windows.npz")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_data(args.npz)
    ds = WindowDataset(X, y)

    n_val = int(0.2 * len(ds))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    if args.model == "rnn":
        model = RNNDecoder(in_ch=X.shape[-1])
    else:
        model = TransformerDecoder(in_ch=X.shape[-1])
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"train_loss": [], "val_loss": [], "val_acc": []}
    best = {"acc": -1, "path": None}

    for epoch in range(1, args.epochs+1):
        tl = train_one_epoch(model, train_loader, opt, crit, device)
        vl, va, _, _ = evaluate(model, val_loader, device)
        metrics["train_loss"].append(tl); metrics["val_loss"].append(vl); metrics["val_acc"].append(va)
        print(f"Epoch {epoch:02d} | train_loss {tl:.4f} | val_loss {vl:.4f} | val_acc {va:.3f}")

        if va > best["acc"]:
            best["acc"] = va
            best["path"] = out_dir / f"{args.model}_best.pt"
            torch.save(model.state_dict(), best["path"])

    with open(out_dir/"metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # quick plot
    plot_training_curve(str(out_dir/"metrics.json"))
    print("Best val_acc:", best["acc"], "checkpoint:", best["path"])

if __name__ == "__main__":
    main()
