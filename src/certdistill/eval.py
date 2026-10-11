"""Evaluation: votes, margins, worst-case + placement-conditional certificates."""
import numpy as np


def vote_stats(votes):
    """votes: (K, N). Returns dict with per-sample n_a, n_b, pred, margin."""
    K, N = votes.shape
    n_a = np.zeros(N, dtype=np.int64)
    n_b = np.zeros(N, dtype=np.int64)
    pred = np.zeros(N, dtype=np.int64)
    for j in range(N):
        counts = np.bincount(votes[:, j], minlength=10)
        order = np.argsort(-counts, kind="stable")
        pred[j] = order[0]
        n_a[j] = counts[order[0]]
        n_b[j] = counts[order[1]]
    return {"pred": pred, "n_a": n_a, "n_b": n_b,
            "margin": (n_a - n_b).astype(np.int64)}


def certified_accuracy(stats, yte, B):
    """Worst-case certificate: correct AND margin > 2B."""
    ok = (stats["pred"] == yte) & (stats["margin"] > 2 * B)
    return float(ok.mean())


def compromised_count(subsets, malicious, alpha):
    """t = #subsets where Byzantine count exceeds trimming capacity."""
    mal = set(malicious)
    t = 0
    per_subset_b = []
    for s in subsets:
        b = sum(1 for i in s if i in mal)
        per_subset_b.append(b)
        if b > alpha * len(s):
            t += 1
    return t, per_subset_b


def conditional_certified_accuracy(stats, yte, t):
    """Placement-conditional certificate: correct AND margin > 2t."""
    ok = (stats["pred"] == yte) & (stats["margin"] > 2 * t)
    return float(ok.mean())


def accuracy(pred, yte):
    return float((pred == yte).mean())


def vote_flip_rate(votes_attack, votes_clean):
    """Fraction of (subset, sample) votes differing from the clean run."""
    return float((votes_attack != votes_clean).mean())


def summarize_ensemble(votes, yte, subsets, malicious, alpha, B_grid):
    stats = vote_stats(votes)
    t, per_subset_b = compromised_count(subsets, malicious, alpha)
    return {
        "accuracy": accuracy(stats["pred"], yte),
        "mean_margin": float(stats["margin"].mean()),
        "frac_margin_ge1": float((stats["margin"] >= 1).mean()),
        "t_effective": t,
        "per_subset_byzantine": per_subset_b,
        "cert_worst": {str(B): certified_accuracy(stats, yte, B) for B in B_grid},
        "cert_cond": conditional_certified_accuracy(stats, yte, t),
        "margin_hist": np.bincount(np.clip(stats["margin"], 0, 20)).tolist(),
    }
