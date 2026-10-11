"""Baselines: B1 FedDF-mean, B2 FedRAD (median), B3 Cao-homogeneous (FedAvg ensemble)."""
import numpy as np
import torch
import torch.nn as nn

from certdistill.models import make_arch, make_student
from certdistill.attacks import (poison_data, corrupt_logits, corrupt_delta,
                                 DATA_ATTACKS, LOGIT_ATTACKS, DELTA_ATTACKS)
from certdistill.core import train_local, client_logits, distill_student, set_seed


class FedDF:
    """Single-student distillation FL. robust in {'mean','median'}."""

    def __init__(self, client_data, arch_of_client, probe_X, robust="mean",
                 rounds=6, local_epochs=3, distill_epochs=3, tau=2.0,
                 attack=None, malicious=None, seed=0, cached_logits=None):
        self.client_data = client_data
        self.arch_of_client = arch_of_client
        self.probe_X = probe_X
        self.robust = robust
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.distill_epochs = distill_epochs
        self.tau = tau
        self.attack = attack
        self.malicious = set(malicious or [])
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cached_logits = cached_logits
        self.student = make_student()
        self.client_models = {i: make_arch(a) for i, a in enumerate(arch_of_client)}

    def _aggregate(self, zs):
        if self.robust == "mean":
            return zs.mean(dim=0)
        if self.robust == "median":
            return zs.median(dim=0).values
        raise ValueError(self.robust)

    def _needs_training(self, i):
        return (self.cached_logits is None or
                (i in self.malicious and self.attack in DATA_ATTACKS))

    def run(self, ckpt=None):
        start_r = ckpt["round"] + 1 if ckpt else 0
        if ckpt:
            self.student.load_state_dict(ckpt["student"])
            for i, sd in ckpt.get("client_models", {}).items():
                self.client_models[i].load_state_dict(sd)
        for r in range(start_r, self.rounds):
            zs = []
            for i in range(len(self.client_data)):
                if not self._needs_training(i):
                    z = self.cached_logits[r][i].clone()
                else:
                    X, y = self.client_data[i]
                    if i in self.malicious and self.attack in DATA_ATTACKS:
                        X, y = poison_data(X, y, self.attack, self.rng)
                    m = self.client_models[i]
                    train_local(m, X, y, self.local_epochs, seed=self.seed + r * 1000 + i)
                    z = client_logits(m, self.probe_X)
                if i in self.malicious and self.attack in LOGIT_ATTACKS:
                    z = corrupt_logits(z, self.attack)
                zs.append(z)
            targets = self._aggregate(torch.stack(zs))
            distill_student(self.student, targets, self.probe_X,
                            self.distill_epochs, tau=self.tau,
                            seed=self.seed + r * 100)
            yield r
        return self.student

    def state(self, r):
        d = {"round": r, "student": self.student.state_dict()}
        if self.cached_logits is None:
            d["client_models"] = {i: m.state_dict() for i, m in self.client_models.items()}
        return d

    @torch.no_grad()
    def predict(self, Xte, batch=1024):
        self.student.eval()
        Xt = torch.from_numpy(Xte)
        out = []
        for i in range(0, len(Xt), batch):
            out.append(self.student(Xt[i:i + batch]).argmax(dim=1))
        return torch.cat(out).numpy()


class CaoHomogeneous:
    """Cao et al. (AAAI'21) style: homogeneous arch, per-subset FedAvg, majority vote."""

    def __init__(self, client_data, subsets, attack=None, malicious=None,
                 rounds=6, local_epochs=3, seed=0):
        self.client_data = client_data
        self.subsets = subsets
        self.attack = attack
        self.malicious = set(malicious or [])
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.models = [make_arch("cnn_d") for _ in subsets]

    def run(self, ckpt=None):
        start_r = ckpt["round"] + 1 if ckpt else 0
        if ckpt:
            for s, sd in zip(self.models, ckpt["models"]):
                s.load_state_dict(sd)
        for r in range(start_r, self.rounds):
            for k, subset in enumerate(self.subsets):
                g = self.models[k]
                deltas = []
                for i in subset:
                    X, y = self.client_data[i]
                    if i in self.malicious and self.attack in DATA_ATTACKS:
                        X, y = poison_data(X, y, self.attack, self.rng)
                    m = make_arch("cnn_d")
                    m.load_state_dict(g.state_dict())
                    train_local(m, X, y, self.local_epochs,
                                seed=self.seed + r * 1000 + i)
                    delta = {n: m.state_dict()[n] - g.state_dict()[n]
                             for n in m.state_dict()}
                    if i in self.malicious and self.attack in DELTA_ATTACKS:
                        delta = corrupt_delta(delta, self.attack)
                    deltas.append(delta)
                avg = {n: torch.stack([d[n] for d in deltas]).mean(dim=0)
                       for n in deltas[0]}
                g.load_state_dict({n: g.state_dict()[n] + avg[n] for n in avg})
            yield r
        return self.models

    def state(self, r):
        return {"round": r, "models": [m.state_dict() for m in self.models]}

    @torch.no_grad()
    def votes(self, Xte, batch=1024):
        Xt = torch.from_numpy(Xte)
        all_v = []
        for s in self.models:
            s.eval()
            v = []
            for i in range(0, len(Xt), batch):
                v.append(s(Xt[i:i + batch]).argmax(dim=1))
            all_v.append(torch.cat(v).numpy())
        return np.stack(all_v)
