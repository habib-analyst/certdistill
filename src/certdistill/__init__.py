"""certdistill: certified Byzantine robustness for model-heterogeneous FL."""
from certdistill.core import CertDistill, trimmed_mean, set_seed
from certdistill.baselines import FedDF, CaoHomogeneous
from certdistill.eval import (vote_stats, certified_accuracy,
                              conditional_certified_accuracy,
                              compromised_count, summarize_ensemble)
from certdistill.models import ARCH_FAMILIES, make_arch, make_student
from certdistill.attacks import add_trigger

__version__ = "0.1.0"
__all__ = ["CertDistill", "FedDF", "CaoHomogeneous", "trimmed_mean", "set_seed",
           "vote_stats", "certified_accuracy", "conditional_certified_accuracy",
           "compromised_count", "summarize_ensemble", "ARCH_FAMILIES",
           "make_arch", "make_student", "add_trigger"]


def stratified_subsets(n_clients, k, arch_of_client, seed=0):
    """Partition clients into k stratified subsets (one arch family each)."""
    import numpy as np
    by_arch = {a: [] for a in ARCH_FAMILIES}
    for i, a in enumerate(arch_of_client):
        by_arch[a].append(i)
    rng = np.random.default_rng(seed)
    for a in by_arch:
        rng.shuffle(by_arch[a])
    return [[by_arch[a][j] for a in ARCH_FAMILIES] for j in range(k)]
