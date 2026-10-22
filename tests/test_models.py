import numpy as np
import pandas as pd
import torch

from src.data import subsample
from src.evaluate import pick_correct, pick_errors
from src.gradcam import GradCAM
from src.models import SmallCNN, build_model, last_conv_layer


def test_small_cnn_output_shape():
    model = SmallCNN(n_classes=10)
    assert model(torch.randn(4, 3, 64, 64)).shape == (4, 10)


def test_resnet_head_replaced():
    model = build_model("resnet18", 10)
    assert model(torch.randn(2, 3, 128, 128)).shape == (2, 10)


def test_gradcam_heatmap_shape_and_range():
    model = SmallCNN(n_classes=10)
    heat, cls, prob = GradCAM(model, last_conv_layer(model))(torch.randn(1, 3, 64, 64))
    assert heat.shape == (64, 64)
    assert 0 <= heat.min() and heat.max() <= 1
    assert 0 <= cls < 10 and 0 < prob <= 1


def test_subsample_is_stratified():
    df = pd.DataFrame({"path": range(1000), "label": ["a"] * 800 + ["b"] * 200})
    part = subsample(df, 0.1)
    assert len(part) == 100
    assert part["label"].value_counts().to_dict() == {"a": 80, "b": 20}


def test_example_pickers():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    assert all(y[i] == pred[i] for i in pick_correct(y, pred))
    assert set(pick_errors(y, pred)) == {1, 5}
