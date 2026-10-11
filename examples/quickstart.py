"""Quickstart: train a CertDistill ensemble on MNIST and print certificates.

Expected output (clean, CPU):
  ensemble accuracy: ~0.96
  certified accuracy @ B=2 (worst-case): ~0.90
  certified accuracy @ B=6 (worst-case): ~0.55
Exact numbers depend on seeds; see README for the measured table.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from certdistill import (CertDistill, ARCH_FAMILIES, stratified_subsets,
                         summarize_ensemble)
from certdistill.data import load_mnist, make_federation

N_CLIENTS, K = 60, 20
B_GRID = [0, 1, 2, 4, 6, 10]

print("loading MNIST...")
Xtr, ytr, Xte, yte = load_mnist()
client_data, probe_full = make_federation(Xtr, ytr, N_CLIENTS, 2000, seed=7)
probe_X = probe_full[:2000]
arch_of_client = [ARCH_FAMILIES[i % 3] for i in range(N_CLIENTS)]
subsets = stratified_subsets(N_CLIENTS, K, arch_of_client, seed=0)

algo = CertDistill(client_data, arch_of_client, subsets, probe_X,
                   alpha=1 / 3, rounds=4, distill_epochs=2, tau=2.0, seed=1000)
for r in algo.run():
    print(f"round {r} done")

votes = algo.votes(Xte)
summ = summarize_ensemble(votes, yte, subsets, [], 1 / 3, B_GRID)
print(f"ensemble accuracy: {summ['accuracy']:.4f}")
for b in B_GRID:
    print(f"certified accuracy @ B={b} (worst-case): {summ['cert_worst'][str(b)]:.4f}")
print(f"placement-conditional certified accuracy: {summ['cert_cond']:.4f}")
