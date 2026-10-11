"""Heterogeneous client architectures + fixed student architecture. PyTorch CPU."""
import torch
import torch.nn as nn


class CNN_S(nn.Module):  # small CNN: 2 conv
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


class CNN_D(nn.Module):  # deeper CNN: 3 conv
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


ARCH_FAMILIES = ["cnn_s", "cnn_d", "mlp"]


def make_arch(name, n_classes=10):
    return {"cnn_s": CNN_S, "cnn_d": CNN_D, "mlp": MLP}[name](n_classes)


def make_student(n_classes=10):
    """Fixed student architecture used for every distilled subset model."""
    return CNN_D(n_classes)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
