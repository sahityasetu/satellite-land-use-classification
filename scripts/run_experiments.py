"""Train both models on 5%, 10%, 25% and 100% of the training labels (8 runs, ~1.5 h on an M2 laptop).

Run from the project root:  python scripts/run_experiments.py
Smaller training sets get more epochs so every run sees a reasonable number of updates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import run  # noqa: E402

EPOCHS = {"cnn": {0.05: 40, 0.1: 30, 0.25: 25, 1.0: 20}, "resnet18": {0.05: 15, 0.1: 12, 0.25: 10, 1.0: 8}}
LR = {"cnn": 3e-3, "resnet18": 1e-3}
SIZE = {"cnn": 64, "resnet18": 128}

if __name__ == "__main__":
    for model in ["resnet18", "cnn"]:
        for fraction in [1.0, 0.25, 0.1, 0.05]:
            run(model, EPOCHS[model][fraction], LR[model], fraction, SIZE[model])
