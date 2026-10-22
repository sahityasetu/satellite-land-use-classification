"""Train one model and keep the checkpoint with the best validation accuracy.

Run from the project root:
    python -m src.train --model resnet18 --epochs 8
    python -m src.train --model cnn --epochs 20
    python -m src.train --model resnet18 --epochs 8 --fraction 0.05     # data-efficiency experiment
Outputs: models/<run>.pt and reports/runs/<run>.json (history + test metrics)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.data import get_loaders
from src.models import build_model, count_params, get_device

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def run(model_name: str, epochs: int, lr: float, fraction: float, size: int):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    loaders, classes = get_loaders(size=size, train_fraction=fraction)
    model = build_model(model_name, len(classes)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(loaders["train"]))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    run_name = f"{model_name}_frac{fraction:g}"
    ckpt = ROOT / "models" / f"{run_name}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    history, best_acc, start = [], -1.0, time.time()
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loaders["train"], optimizer, scheduler, loss_fn, device)
        p, y = predict(model, loaders["val"], device)
        val_acc = float((p.argmax(1) == y).mean())
        history.append({"epoch": epoch, "train_loss": round(loss, 4), "val_acc": round(val_acc, 4)})
        print(f"{run_name} epoch {epoch:2d}  train_loss {loss:.4f}  val_acc {val_acc:.4f}  ({time.time() - start:.0f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model_name, "classes": classes, "size": size, "state_dict": model.state_dict()}, ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    p, y = predict(model, loaders["test"], device)
    result = {
        "run": run_name, "model": model_name, "train_fraction": fraction,
        "train_images": len(loaders["train"].dataset), "params": count_params(model),
        "epochs": epochs, "image_size": size, "train_minutes": round((time.time() - start) / 60, 1),
        "best_val_acc": round(best_acc, 4),
        "test_acc": round(float((p.argmax(1) == y).mean()), 4),
        "test_macro_f1": round(f1_score(y, p.argmax(1), average="macro"), 4),
        "history": history,
    }
    out = ROOT / "reports" / "runs" / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=None)
    a = ap.parse_args()
    # fine-tuning wants a smaller learning rate than training from scratch
    lr = a.lr or (1e-3 if a.model == "resnet18" else 3e-3)
    # ResNet-18 downsamples 32x; upscaling 64 -> 128 px keeps a 4x4 map in the last layer
    size = a.size or (128 if a.model == "resnet18" else 64)
    run(a.model, a.epochs, lr, a.fraction, size)


if __name__ == "__main__":
    main()
