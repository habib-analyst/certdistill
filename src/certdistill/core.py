"""CertDistill core: subset training + trimmed-mean distillation + students."""
import copy
import numpy as np
import torch
import torch.nn as nn

from certdistill.models import make_arch, make_student
from certdistill.attacks import poison_data, corrupt_logits, DATA_ATTACKS, LOGIT_ATTACKS


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def trimmed_mean(logits, alpha):
    """logits: (m, P, C) tensor of client logits. Returns (P, C) robust targets."""
    m = logits.shape[0]
    k = int(alpha * m)
    if k == 0:
        return logits.mean(dim=0)
    s, _ = torch.sort(logits, dim=0)
    return s[k:m - k].mean(dim=0)


def train_local(model, X, y, epochs, lr=1e-3, batch=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def client_logits(model, probe_X, batch=512):
    model.eval()
    out = []
    Xt = torch.from_numpy(probe_X)
    for i in range(0, len(Xt), batch):
        out.append(model(Xt[i:i + batch]))
    return torch.cat(out, dim=0)  # (P, C)


def distill_student(student, targets, probe_X, epochs, tau=2.0, lr=1e-3,
                    batch=256, seed=0):
    """KL distillation toward soft targets (targets are raw logits)."""
    g = torch.Generator().manual_seed(seed)
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    kl = nn.KLDivLoss(reduction="batchmean")
    Xt = torch.from_numpy(probe_X)
    Tt = targets  # (P, C) logits
    n = len(Xt)
    soft = torch.softmax(Tt / tau, dim=1)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            logp = torch.log_softmax(student(Xt[b]) / tau, dim=1)
            loss = kl(logp, soft[b]) * (tau * tau)
            loss.backward()
            opt.step()
    return student


class CertDistill:
    """Heterogeneous-architecture certified ensemble via robust distillation."""

    def __init__(self, client_data, arch_of_client, subsets, probe_X,
                 alpha=1 / 3, rounds=6, local_epochs=3, distill_epochs=3,
                 tau=2.0, attack=None, malicious=None, seed=0,
                 cached_logits=None):
        self.client_data = client_data
        self.arch_of_client = arch_of_client
        self.subsets = subsets            # list of lists of client idx
        self.n_clients = len(client_data)
        self.probe_X = probe_X
        self.alpha = alpha
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.distill_epochs = distill_epochs
        self.tau = tau
        self.attack = attack              # None or attack name
        self.malicious = set(malicious or [])
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # cached_logits: {round: tensor (N, P, C)} of honest-trained client logits.
        # Honest clients skip training and reuse the cache; Byzantine clients
        # with data attacks still train (their data differs). Methodologically
        # identical to from-scratch training (same data, same seeds) — it only
        # isolates the attack effect from honest-training noise.
        self.cached_logits = cached_logits
        self.students = [make_student() for _ in subsets]
        # per-client persistent models (warm start across rounds)
        self.client_models = {i: make_arch(a) for i, a in enumerate(arch_of_client)}

    def _needs_training(self, i):
        # Only Byzantine clients with data attacks need (re)training;
        # everyone else reuses cached honest logits.
        return (self.cached_logits is None or
                (i in self.malicious and self.attack in DATA_ATTACKS))

    def _client_round(self, i, r):
        if not self._needs_training(i):
            z = self.cached_logits[r][i].clone()
        else:
            X, y = self.client_data[i]
            model = self.client_models[i]
            if i in self.malicious and self.attack in DATA_ATTACKS:
                X, y = poison_data(X, y, self.attack, self.rng)
            train_local(model, X, y, self.local_epochs, seed=self.seed + r * 1000 + i)
            z = client_logits(model, self.probe_X)
        if i in self.malicious and self.attack in LOGIT_ATTACKS:
            z = corrupt_logits(z, self.attack)
        return z

    def run(self, ckpt=None, save_logits_to=None):
        start_r = 0
        if ckpt:
            start_r = ckpt["round"] + 1
            for s, sd in zip(self.students, ckpt["students"]):
                s.load_state_dict(sd)
            for i, sd in ckpt.get("client_models", {}).items():
                self.client_models[i].load_state_dict(sd)
        for r in range(start_r, self.rounds):
            round_logits = {}
            for k, subset in enumerate(self.subsets):
                zs = torch.stack([self._client_round(i, r) for i in subset])  # (m,P,C)
                for i, z in zip(subset, zs):
                    round_logits[i] = z
                targets = trimmed_mean(zs, self.alpha)
                distill_student(self.students[k], targets, self.probe_X,
                                self.distill_epochs, tau=self.tau,
                                seed=self.seed + r * 100 + k)
            if save_logits_to:
                all_z = torch.stack([round_logits[i] for i in range(self.n_clients)])
                torch.save(all_z, save_logits_to.format(r=r))
            yield r  # checkpoint hook
        return self.students

    def state(self, r):
        d = {"round": r, "students": [s.state_dict() for s in self.students]}
        if self.cached_logits is None:
            # full-training run: keep client models for resume
            d["client_models"] = {i: m.state_dict() for i, m in self.client_models.items()}
        return d

    @torch.no_grad()
    def votes(self, Xte, batch=1024):
        """Returns (K, Ntest) vote matrix."""
        Xt = torch.from_numpy(Xte)
        all_v = []
        for s in self.students:
            s.eval()
            v = []
            for i in range(0, len(Xt), batch):
                v.append(s(Xt[i:i + batch]).argmax(dim=1))
            all_v.append(torch.cat(v).numpy())
        return np.stack(all_v)
