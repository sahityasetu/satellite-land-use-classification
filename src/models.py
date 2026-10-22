"""Two models: a small CNN trained from scratch (baseline) and a pretrained ResNet-18 fine-tuned on EuroSAT."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """4 conv blocks (32-64-128-256 filters) + global average pooling. ~1.2M parameters."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128), conv_block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str, n_classes: int) -> nn.Module:
    if name == "cnn":
        return SmallCNN(n_classes)
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, n_classes))
        return model
    raise ValueError(f"unknown model {name}")


def last_conv_layer(model: nn.Module) -> nn.Module:
    """Layer used by Grad-CAM: the last convolutional block of either model."""
    return model.layer4 if hasattr(model, "layer4") else model.features[-1]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
