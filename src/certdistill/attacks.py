"""Byzantine behaviors. Two hooks: poison_data() and corrupt_message().

Data attacks (applied before local training):
  - labelflip : y -> 9 - y
  - backdoor  : 30% of local samples get a 3x3 white trigger (bottom-right), label -> 0
Message attacks (applied to the honest message before sending):
  - logit_invert   : z -> -3 * z            (distillation methods)
  - logit_targeted : one-hot(target) * 10   (distillation methods)
  - logit_random   : N(0, 25)               (distillation methods)
  - update_negate  : delta -> -3 * delta     (FedAvg-style methods)
"""
import numpy as np
import torch

DATA_ATTACKS = {"labelflip", "backdoor"}
LOGIT_ATTACKS = {"logit_invert", "logit_targeted", "logit_random"}
DELTA_ATTACKS = {"update_negate"}


def poison_data(X, y, attack, rng, target_class=0, frac=0.3):
    X = X.copy()
    y = y.copy()
    if attack == "labelflip":
        y = 9 - y
    elif attack == "backdoor":
        n = len(X)
        idx = rng.choice(n, size=int(frac * n), replace=False)
        X[idx, :, 24:27, 24:27] = 1.0
        y[idx] = target_class
    return X, y


def corrupt_logits(z, attack, target_class=0):
    z = z.clone()
    if attack == "logit_invert":
        return -3.0 * z
    if attack == "logit_targeted":
        out = torch.zeros_like(z)
        out[:, target_class] = 10.0
        return out
    if attack == "logit_random":
        return torch.randn_like(z) * 5.0
    raise ValueError(attack)


def corrupt_delta(delta, attack):
    if attack == "update_negate":
        return {k: -3.0 * v for k, v in delta.items()}
    raise ValueError(attack)


def add_trigger(X):
    """Apply the backdoor trigger to a batch (for ASR evaluation)."""
    X = X.copy()
    X[:, :, 24:27, 24:27] = 1.0
    return X
