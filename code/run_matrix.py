"""Resumable experiment matrix for certdistill.

Usage:
  python run_matrix.py --pilot        # 1 quick config, timing calibration
  python run_matrix.py                # full matrix, resumable
  python run_matrix.py --resume       # same (default behavior: skip done)

Each config checkpoints per round to checkpoints/<name>.pt.
Results -> results/<name>.json. DONE marker when the whole matrix finishes.
Launch detached: setsid nohup ./venv/bin/python code/run_matrix.py > logs/run.log 2>&1 &
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from certdistill.data import load_mnist, make_federation
from certdistill.models import ARCH_FAMILIES
from certdistill.core import CertDistill, set_seed
from certdistill.baselines import FedDF, CaoHomogeneous
from certdistill.eval import summarize_ensemble, accuracy, vote_flip_rate, vote_stats
from certdistill.attacks import add_trigger
from certdistill import stratified_subsets

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(BASE, "checkpoints")
RES = os.path.join(BASE, "results")
os.makedirs(CKPT, exist_ok=True)
os.makedirs(RES, exist_ok=True)

B_GRID = [0, 1, 2, 4, 6, 10, 18]
N_CLIENTS = 60
K = 20
SUBSET_SEED = 0
TRAIN_SEED = 1000


def malicious_set(n_clients, frac, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_clients)
    b = int(round(frac * n_clients))
    return sorted(order[:b].tolist())


def build_configs(pilot=False):
    cfgs = []
    if pilot:
        cfgs.append(dict(name="pilot_clean", method="certdistill", attack=None,
                          frac=0.0, alpha=1 / 3, probe=500, k=20, rounds=2,
                          tau=2.0, iid=True))
        return cfgs
    # --- main certdistill matrix ---
    attacks = ["labelflip", "logit_invert", "logit_targeted", "logit_random", "backdoor"]
    cfgs.append(dict(name="cd_clean", method="certdistill", attack=None, frac=0.0,
                      alpha=1 / 3, probe=2000, k=20, rounds=4, tau=2.0, iid=True,
                      save_logits=True))
    for atk in attacks:
        frac = 0.3
        cfgs.append(dict(name=f"cd_{atk}_{int(frac*100)}", method="certdistill",
                          attack=atk, frac=frac, alpha=1 / 3, probe=2000,
                          k=20, rounds=4, tau=2.0, iid=True, cache=True))
    # --- baselines (trimmed for CPU budget) ---
    for atk, frac in [(None, 0.0), ("logit_targeted", 0.2)]:
        nm = "feddf_mean_" + ("clean" if atk is None else f"{atk}_20")
        cfgs.append(dict(name=nm, method="feddf_mean", attack=atk, frac=frac,
                          probe=2000, rounds=4, tau=2.0, iid=True, cache=True))
    for atk, frac in [(None, 0.0), ("logit_targeted", 0.2)]:
        nm = "fedrad_" + ("clean" if atk is None else f"{atk}_20")
        cfgs.append(dict(name=nm, method="fedrad", attack=atk, frac=frac,
                          probe=2000, rounds=4, tau=2.0, iid=True, cache=True))
    for atk, frac in [(None, 0.0), ("labelflip", 0.2), ("update_negate", 0.2)]:
        nm = "cao_" + ("clean" if atk is None else f"{atk}_20")
        cfgs.append(dict(name=nm, method="cao", attack=atk, frac=frac,
                          rounds=4, iid=True))
    # --- ablations ---
    for alpha in (0.0, 1 / 6):
        cfgs.append(dict(name=f"cd_alpha{alpha:.3f}_targeted_20", method="certdistill",
                          attack="logit_targeted", frac=0.2, alpha=alpha,
                          probe=2000, k=20, rounds=4, tau=2.0, iid=True, cache=True))
    cfgs.append(dict(name="cd_probe500_clean", method="certdistill", attack=None,
                      frac=0.0, alpha=1 / 3, probe=500, k=20, rounds=4,
                      tau=2.0, iid=True, cache=True))
    for kk in (10,):
        cfgs.append(dict(name=f"cd_k{kk}_clean", method="certdistill", attack=None,
                          frac=0.0, alpha=1 / 3, probe=2000, k=kk, rounds=4,
                          tau=2.0, iid=True, cache=True))
        cfgs.append(dict(name=f"cd_k{kk}_targeted_20", method="certdistill",
                          attack="logit_targeted", frac=0.2, alpha=1 / 3,
                          probe=2000, k=kk, rounds=4, tau=2.0, iid=True, cache=True))
    cfgs.append(dict(name="cd_noniid_clean", method="certdistill", attack=None,
                      frac=0.0, alpha=1 / 3, probe=2000, k=20, rounds=4,
                      tau=2.0, iid=False))
    for tau in (4.0,):
        cfgs.append(dict(name=f"cd_tau{tau:g}_clean", method="certdistill",
                          attack=None, frac=0.0, alpha=1 / 3, probe=2000,
                          k=20, rounds=4, tau=tau, iid=True, cache=True))
    return cfgs


def load_cached_logits(rounds, probe_size):
    """Load cd_clean's per-round honest client logits; slice to probe_size."""
    cache = {}
    for r in range(rounds):
        p = os.path.join(CKPT, f"cd_clean_logits_r{r}.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"cache missing: {p} (run cd_clean first)")
        t = torch.load(p, map_location="cpu", weights_only=False)  # (N, 2000, C)
        cache[r] = t[:, :probe_size]
    return cache


def run_config(cfg, Xtr, ytr, Xte, yte):
    set_seed(TRAIN_SEED)
    torch.set_num_threads(2)
    t0 = time.time()
    client_data, probe_full = make_federation(
        Xtr, ytr, N_CLIENTS, max(2000, cfg.get("probe", 2000)),
        iid=cfg.get("iid", True), seed=7)
    # subsample clients for CPU speed (300 samples/client is plenty for MNIST)
    spc = cfg.get("samples_per_client", 300)
    rng_sp = np.random.default_rng(TRAIN_SEED + 5)
    sub_data = []
    for (X, y) in client_data:
        if len(X) > spc:
            idx = rng_sp.choice(len(X), spc, replace=False)
            sub_data.append((X[idx], y[idx]))
        else:
            sub_data.append((X, y))
    client_data = sub_data
    probe_size = cfg.get("probe", 2000)
    probe_X = probe_full[:probe_size]
    arch_of_client = [ARCH_FAMILIES[i % 3] for i in range(N_CLIENTS)]
    mal = malicious_set(N_CLIENTS, cfg["frac"], seed=2000)
    ckpt_path = os.path.join(CKPT, cfg["name"] + ".pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False) if os.path.exists(ckpt_path) else None

    use_cache = cfg.get("cache", False)
    cached = load_cached_logits(cfg["rounds"], probe_size) if use_cache else None
    save_logits_to = (os.path.join(CKPT, "cd_clean_logits_r{r}.pt")
                      if cfg.get("save_logits", False) else None)

    method = cfg["method"]
    if method == "certdistill":
        subsets = stratified_subsets(N_CLIENTS, 20, arch_of_client, seed=SUBSET_SEED)
        if cfg["k"] == 10:  # pair triples -> sextuples
            subsets = [a + b for a, b in zip(subsets[0::2], subsets[1::2])]
        algo = CertDistill(client_data, arch_of_client, subsets, probe_X,
                           alpha=cfg["alpha"], rounds=cfg["rounds"],
                           local_epochs=cfg.get("local_epochs", 3),
                           distill_epochs=2, tau=cfg["tau"],
                           attack=cfg["attack"], malicious=mal, seed=TRAIN_SEED,
                           cached_logits=cached)
        for r in algo.run(ckpt, save_logits_to=save_logits_to):
            torch.save(algo.state(r), ckpt_path)
        votes = algo.votes(Xte)
        summ = summarize_ensemble(votes, yte, subsets, mal, cfg["alpha"], B_GRID)
        # backdoor ASR on triggered test set
        Xte_trig = add_trigger(Xte)
        v_trig = algo.votes(Xte_trig)
        st = vote_stats(v_trig)
        summ["backdoor_asr"] = float((st["pred"] == 0).mean())
        # vote-flip vs clean run (only when K matches)
        clean_votes_path = os.path.join(RES, "cd_clean.votes.npy")
        if cfg["name"] != "cd_clean" and os.path.exists(clean_votes_path):
            cv = np.load(clean_votes_path)
            if cv.shape == votes.shape:
                summ["vote_flip_vs_clean"] = vote_flip_rate(votes, cv)
            else:
                summ["vote_flip_vs_clean"] = None  # K differs; not comparable
        if cfg["name"] == "cd_clean":
            np.save(clean_votes_path, votes)
        result = {"config": cfg, "malicious": mal, **summ}
    elif method in ("feddf_mean", "fedrad"):
        robust = "mean" if method == "feddf_mean" else "median"
        algo = FedDF(client_data, arch_of_client, probe_X, robust=robust,
                     rounds=cfg["rounds"], local_epochs=cfg.get("local_epochs", 3),
                     distill_epochs=2, tau=cfg["tau"],
                     attack=cfg["attack"], malicious=mal, seed=TRAIN_SEED,
                     cached_logits=cached)
        for r in algo.run(ckpt):
            torch.save(algo.state(r), ckpt_path)
        pred = algo.predict(Xte)
        Xte_trig = add_trigger(Xte)
        pred_trig = algo.predict(Xte_trig)
        result = {"config": cfg, "malicious": mal,
                  "accuracy": accuracy(pred, yte),
                  "backdoor_asr": float((pred_trig == 0).mean())}
    elif method == "cao":
        arch_h = ["cnn_d"] * N_CLIENTS
        rng = np.random.default_rng(SUBSET_SEED)
        order = rng.permutation(N_CLIENTS)
        subsets = [sorted(order[i * 3:(i + 1) * 3].tolist()) for i in range(20)]
        atk = cfg["attack"]
        algo = CaoHomogeneous(client_data, subsets, attack=atk, malicious=mal,
                              rounds=cfg["rounds"],
                              local_epochs=cfg.get("local_epochs", 3),
                              seed=TRAIN_SEED)
        for r in algo.run(ckpt):
            torch.save(algo.state(r), ckpt_path)
        votes = algo.votes(Xte)
        summ = summarize_ensemble(votes, yte, subsets, mal, 0.0, B_GRID)
        result = {"config": cfg, "malicious": mal, **summ}
    else:
        raise ValueError(method)
    result["wall_seconds"] = time.time() - t0
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    print("loading MNIST...", flush=True)
    Xtr, ytr, Xte, yte = load_mnist()
    print(f"train {Xtr.shape} test {Xte.shape}", flush=True)
    for cfg in build_configs(pilot=args.pilot):
        out = os.path.join(RES, cfg["name"] + ".json")
        if os.path.exists(out):
            print(f"SKIP {cfg['name']} (done)", flush=True)
            continue
        print(f"RUN {cfg['name']} {cfg['method']} attack={cfg['attack']} frac={cfg['frac']}", flush=True)
        res = run_config(cfg, Xtr, ytr, Xte, yte)
        with open(out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"DONE {cfg['name']} acc={res.get('accuracy', float('nan')):.4f} "
              f"wall={res['wall_seconds'] / 60:.1f}min", flush=True)
    if not args.pilot:
        open(os.path.join(BASE, "PIPELINE_DONE"), "w").write("done\n")
        print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
