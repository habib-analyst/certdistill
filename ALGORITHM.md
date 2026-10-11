# CertDistill — Formal Algorithm Specification

**Certified Byzantine Robustness for Model-Heterogeneous Federated Learning**

Status: INVENTION #3 (Algorithms). Gap verified 2026-10-11: certified+heterogeneous
is empty — FedRAD (2021) empirical-only; Cao et al. (AAAI'21) certified but
homogeneous-only; Roux et al. (2024/2025) study KD-based FL under Byzantines
empirically only; BR-MTRL (2025) uses a shared backbone (not heterogeneous
architectures).

---

## 1. Problem setting

- $N$ clients; client $i$ owns private dataset $D_i$ and runs architecture
  $A_i \in \mathcal{A}$ (heterogeneous: architectures differ across clients).
  Weight averaging across clients is **impossible**.
- Up to $B$ clients are Byzantine (arbitrary behavior: poisoned data and/or
  arbitrary messages).
- Server holds an **unlabeled public probe set** $P = \{x_j\}_{j=1}^{|P|}$
  (labels never used; standard FedDF assumption).
- Goal: a global predictor $F$ with **provable** robustness — a certificate
  that a test prediction is unaffected by the $B$ malicious clients — in the
  heterogeneous setting.

## 2. Algorithm

### 2.1 Subset partition
Randomly partition the $N$ clients into $K$ **disjoint** subsets
$S_1, \dots, S_K$ (fixed once, public). Let $m_k = |S_k|$, $b_k$ = #Byzantine
in $S_k$ (unknown to the server).

### 2.2 Robust distillation (per subset, per round)
For round $r = 1..R$, for each subset $k$ independently:

1. **Local training.** Each client $i \in S_k$ trains its own architecture
   $A_i$ on $D_i$ for $E$ epochs (cross-entropy, SGD/Adam). Byzantine clients
   may train on poisoned data or skip training.
2. **Logit collection.** Each client $i \in S_k$ sends logits
   $z_i(x_j) \in \mathbb{R}^C$ for every probe sample $x_j \in P$.
   (Byzantine clients send arbitrary vectors.)
3. **Trimmed-mean aggregation.** For each $(x_j, c)$: sort the $m_k$ logit
   values, drop the top and bottom $\lfloor \alpha m_k \rfloor$, average the
   rest → robust soft target $\bar{z}(x_j)$. ($\alpha = 1/3, m_k = 3$ reduces
   exactly to the coordinate-wise median.)
4. **Distillation.** Train student $f_k$ (fixed student architecture for all
   $k$; warm-started from round $r-1$) for $D$ epochs on $P$ with
   $\mathcal{L} = \mathrm{KL}\big(\sigma(f_k(x)/\tau) \,\|\, \sigma(\bar{z}(x)/\tau)\big)$.

### 2.3 Ensemble prediction
$F(x) = \mathrm{mode}\{f_1(x), \dots, f_K(x)\}$ (ties → smallest label).
Let $n_a$ = votes for the predicted label $a$, $n_b$ = votes for the
runner-up label $b$.

### 2.5 Implementation note: honest-client logit cache
Client training dominates compute. Honest clients train on identical data
with identical seeds in every configuration, so their per-round logits are
recomputed wastefully across attack/ablation configs. We train the honest
client models once (in the clean configuration), cache their per-round
probe-set logits, and reuse the cache: logit-attack configs skip client
training entirely (Byzantine logits = cached honest logits + corruption);
data-attack configs train only the Byzantine clients. This is
methodologically identical to from-scratch training (same data, same seeds,
same RNG streams) — it isolates the attack effect from honest-training
noise, making vote-flip measurements cleaner, not weaker.

### 2.4 Pseudocode
```
partition clients into K disjoint subsets S_1..S_K
for r in 1..R:
    for k in 1..K:                                   # parallelizable
        for i in S_k:  train A_i on D_i for E epochs  # local, heterogeneous
        for x in P:  collect {z_i(x) : i in S_k}
        zbar(x) = trimmed_mean({z_i(x)}, alpha)      # robust aggregation
        distill f_k on P toward zbar for D epochs     # warm start
predict(x) = majority_vote(f_1(x), ..., f_K(x))
```

---

## 3. Certificate (re-derived for the heterogeneous setting)

### 3.1 Trimming-capacity lemma
**Lemma 1.** In subset $k$ with $b_k \le \lfloor \alpha m_k \rfloor$ Byzantine
clients, the $\alpha$-trimmed mean of the $m_k$ logits lies coordinate-wise
within $[\min, \max]$ of the honest clients' logits. (Standard: after removing
$\lfloor\alpha m_k\rfloor \ge b_k$ extremes from each tail, every surviving
value is honest, so their mean is within the honest range.)

*Consequence:* a below-capacity Byzantine minority can **perturb but not
arbitrarily control** the distilled targets; the student is trained on soft
labels inside the honest convex hull.

### 3.2 Effective corruptible count
**Definition.** Subset $k$ is *compromised* if $b_k > \alpha m_k$ (Byzantines
exceed trimming capacity). Let $t = \#\{k : b_k > \alpha m_k\}$.
**Lemma 2.** $t \le B$ always (each compromised subset contains ≥1 Byzantine;
subsets disjoint). If Byzantines are spread with $b_k \le \alpha m_k$
everywhere, $t = 0$.

### 3.3 Main theorem
**Theorem (CertDistill certificate).** Fix the test point $x$ with vote margin
$n_a - n_b$ for predicted label $a$. If $n_a - n_b > 2t$, the prediction
$F(x) = a$ is **identical with and without** the Byzantine clients'
participation (they cannot flip it), where $t$ is the effective corruptible
count above.
*Proof sketch.* At most $t$ subset models can be adversarially controlled
(Lemmas 1–2); each controls one vote. Worst case: all $t$ stolen votes move
from $a$ to the runner-up $b$. Then $n_a' \ge n_a - t > n_b + t \ge n_b'$,
so $a$ still wins strictly. ∎

**Corollary (unconditional / worst-case certificate).** Since $t \le B$,
$n_a - n_b > 2B$ certifies the prediction against **any** placement of
$\le B$ Byzantine clients — recovering Cao et al.'s bound, now shown to be
**architecture-agnostic** (it never used weight averaging; the proof only
uses disjoint subsets + one-vote-per-subset).

### 3.4 Honest reading: tight vs vacuous
- The certificate is **non-vacuous** when the vote margin is large relative
  to $B$. With small $K$ and large $B$ it certifies nothing — we report this
  as-is (certified accuracy → 0), not as a failure of the method but as the
  honest operating envelope.
- The **placement-conditional** certificate ($t$ from the actual placement,
  known in simulation) is strictly tighter than the worst-case one whenever
  Byzantines cluster or sit below trimming capacity. The gap between
  worst-case and placement-conditional certified accuracy **is** the
  distillation-tightening contribution — reported, not assumed.
- Below-capacity Byzantines still perturb targets within the honest range;
  §5 measures the empirical vote-flip rate vs a clean run.

## 4. Baselines (all implemented here)
- **B1 FedDF-mean**: heterogeneous clients, mean logit aggregation, single
  global student. No defense, no certificate.
- **B2 FedRAD**: heterogeneous clients, coordinate-wise **median** logit
  aggregation (FedRAD 2021), single student. Empirical defense only.
- **B3 Cao-homogeneous**: all clients run one architecture, per-subset
  FedAvg, $K$ subset models, majority vote + Cao certificate. The
  homogeneous certified reference.

## 5. Evaluation plan
MNIST, $N=60$ clients (300 samples/client), $K=20$ subsets ($m_k=3$, one arch-family each),
$\alpha=1/3$ (= median), $|P|=2000$, $R=4$ rounds, $E=3$ local epochs (batch 32),
$D=2$ distill epochs, $\tau=2$.
Architectures: CNN-S (2 conv), CNN-D (3 conv), MLP — round-robin.
Attacks: label-flip, logit-invert, logit-targeted, random-logits, backdoor
(trigger→0). Fractions: 0/10/20/30% ($B \in \{0,6,12,18\}$).
Metrics: clean accuracy, worst-case certified accuracy over
$B \in \{0,1,2,4,6,10,18\}$, placement-conditional certified accuracy,
backdoor ASR, subset vote-flip rate vs clean.
Ablations: $\alpha \in \{0, 1/6, 1/3\}$; $|P| \in \{500, 2000\}$;
$K \in \{10, 20\}$; $\tau \in \{1, 2, 4\}$; heterogeneous vs homogeneous.
