"""Evaluate the trained models and make the report figures.

Run from the project root after training:  python -m src.evaluate
Outputs: reports/metrics.json and reports/figures/*.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data import MEAN, STD, get_loaders
from src.gradcam import GradCAM
from src.models import build_model, get_device, last_conv_layer
from src.train import predict

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
BEST = "resnet18_frac1"
BLUE, GREY = "#2a78d6", "#898781"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def load_runs() -> pd.DataFrame:
    runs = [json.loads(p.read_text()) for p in sorted((ROOT / "reports" / "runs").glob("*.json"))]
    return pd.DataFrame(runs)


def load_model(run: str, device):
    ckpt = torch.load(ROOT / "models" / f"{run}.pt", map_location=device)
    model = build_model(ckpt["model"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def plot_data_efficiency(runs: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, label, color in [("resnet18", "ResNet-18 (ImageNet pretrained, fine-tuned)", BLUE),
                                ("cnn", "Small CNN (trained from scratch)", GREY)]:
        r = runs[runs.model == model].sort_values("train_images")
        ax.plot(r.train_images, r.test_acc * 100, "o-", color=color, lw=2, ms=7, label=label)
        for _, row in r.iterrows():
            ax.annotate(f"{row.test_acc:.1%}", (row.train_images, row.test_acc * 100), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=color)
    ax.set_xscale("log")
    ax.set_xticks(sorted(runs.train_images.unique()))
    ax.set_xticklabels([f"{n:,}" for n in sorted(runs.train_images.unique())])
    ax.set_xlabel("Labelled training images (log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Transfer learning needs far fewer labels", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color="#e1e0d9")
    fig.tight_layout()
    fig.savefig(FIG / "data_efficiency.png", dpi=160)


def plot_confusion(y, pred, classes):
    cm = confusion_matrix(y, pred, normalize="true") * 100
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.5:
                ax.text(j, i, f"{cm[i, j]:.0f}" if i == j else f"{cm[i, j]:.1f}", ha="center", va="center",
                        fontsize=8, color="white" if cm[i, j] > 60 else "#0b0b0b")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix, ResNet-18 (% of each actual class)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=160)


def denorm(x):
    img = x.cpu().numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    return img.clip(0, 1)


def pick_correct(y, pred, seed=0):
    """One random correctly classified test image per class."""
    rng = np.random.default_rng(seed)
    return [int(rng.choice(np.where((y == c) & (pred == c))[0])) for c in np.unique(y)]


def pick_errors(y, pred, n=8, seed=0):
    rng = np.random.default_rng(seed)
    wrong = np.where(y != pred)[0]
    return [int(i) for i in rng.choice(wrong, size=min(n, len(wrong)), replace=False)]


def plot_gradcam(model, dataset, classes, y, pred, device, path, title, pick):
    cam = GradCAM(model, last_conv_layer(model))
    idx = pick(y, pred)
    fig, axes = plt.subplots(2, len(idx), figsize=(2.1 * len(idx), 5.2))
    for col, i in enumerate(idx):
        x, target = dataset[i]
        heat, cls, prob = cam(x.unsqueeze(0).to(device))
        img = denorm(x)
        axes[0, col].imshow(img)
        axes[1, col].imshow(img)
        axes[1, col].imshow(heat, cmap="jet", alpha=0.45)
        ok = cls == target
        axes[0, col].set_title(f"true: {classes[target]}", fontsize=8)
        mark = "" if ok else "✗ "
        axes[1, col].set_title(f"{mark}pred: {classes[cls]} ({prob:.0%})", fontsize=8,
                               color="#0b0b0b" if ok else "#b23c12", fontweight="normal" if ok else "bold")
        for a in axes[:, col]:
            a.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(h_pad=1.5)
    fig.savefig(path, dpi=160)


def plot_history(runs: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for run, label, color in [("resnet18_frac1", "ResNet-18", BLUE), ("cnn_frac1", "Small CNN", GREY)]:
        h = pd.DataFrame(runs.set_index("run").loc[run, "history"])
        ax.plot(h.epoch, h.val_acc * 100, "o-", color=color, lw=2, ms=4, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Training curves, full training set", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e1e0d9")
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=160)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    device = get_device()
    runs = load_runs()
    plot_data_efficiency(runs)
    plot_history(runs)

    model, ckpt = load_model(BEST, device)
    loaders, classes = get_loaders(size=ckpt["size"])
    p, y = predict(model, loaders["test"], device)
    pred = p.argmax(1)
    report = classification_report(y, pred, target_names=classes, output_dict=True, digits=4)
    plot_confusion(y, pred, classes)

    cm = confusion_matrix(y, pred)
    np.fill_diagonal(cm, 0)
    top_confusions = [{"actual": classes[i], "predicted": classes[j], "count": int(cm[i, j])}
                      for i, j in zip(*np.unravel_index(np.argsort(-cm, axis=None)[:5], cm.shape))]

    ds = loaders["test"].dataset
    plot_gradcam(model, ds, classes, y, pred, device, FIG / "gradcam_correct.png",
                 "Grad-CAM: where the model looks (one correct example per class)", pick_correct)
    plot_gradcam(model, ds, classes, y, pred, device, FIG / "gradcam_errors.png",
                 "Grad-CAM on mistakes (✗ = wrong prediction)", pick_errors)

    summary = runs.drop(columns="history").sort_values(["model", "train_images"]).to_dict("records")
    metrics = {
        "best_model": BEST, "test_images": int(len(y)), "test_accuracy": round(float((pred == y).mean()), 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4), "errors": int((pred != y).sum()),
        "per_class_f1": {c: round(report[c]["f1-score"], 4) for c in classes},
        "top_confusions": top_confusions, "runs": summary,
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
