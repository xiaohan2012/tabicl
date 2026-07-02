# TabICL: A Tabular Foundation Model for In-Context Learning on Large Data

**Authors:** Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan (SODA & Sierra teams, INRIA; ENS/PSL)
**Venue:** ICML 2025 (PMLR 267) · [arXiv:2502.05564](https://arxiv.org/abs/2502.05564)

> Summary digested from the paper PDF. This is the model implemented in `src/tabicl/`.

## TL;DR

TabICL is a **transformer foundation model for tabular classification** that does in-context learning (ICL): the training set is fed as context and predictions are made in a single forward pass, no gradient updates. It is pretrained **exclusively on synthetic data**. Its contribution over TabPFNv2 is **scalability**: a *column-then-row* embedding that collapses the feature dimension into one vector per row *before* the expensive ICL attention, letting it handle **up to 500K samples / 500 features** on affordable hardware. On 200 TALENT classification datasets it is **on par with TabPFNv2 while up to 10× faster**; on the 53 datasets with >10K samples it **beats both TabPFNv2 and CatBoost**.

## 1. Problem & Motivation

- GBDTs (XGBoost, CatBoost) have long dominated tabular ML. Tabular foundation models using ICL (TabPFN → TabPFNv2) recently challenged that dominance, but TabPFNv2's **alternating column-wise and row-wise attention never collapses dimensions**, giving complexity `O(m²n + n²m)` — computationally prohibitive beyond ~10K samples.
- Question the paper asks: *can ICL be scaled to large tables and still pay off?*
- Extra challenge vs. text: table cells are **ambiguous without metadata** (column names, types). TabICL sidesteps this by learning **distribution-aware** embeddings that act as implicit feature identifiers.

## 2. Architecture — three transformers, "embedding then ICL"

A table `X ∈ R^{n×m}` (n rows, m features) flows through two embedding stages that produce one vector per row, then a final ICL transformer. **Labels enter only in stage 3.**

### Stage 1 — Distribution-aware column-wise embedding (`TF_col`)
- Each column `c_j ∈ R^n` is embedded independently but through **one shared Set Transformer** (a hypernetwork), rather than per-column embedding modules → enables cross-table transfer.
- Framed as a *set-input* problem. Pipeline: `U = Lin(c)` (project each cell to `d=128`) → **ISAB** → linear heads produce per-cell weight/bias `W, B` → `e_j = W ⊙ c_j + B`.
- **ISAB (Induced Self-Attention Block):** `M = MAB₁(V_I, U_train, U_train)` (k inducing vectors attend to training cells → `M ∈ R^{k×d}`), then `V = MAB₂(U, M, M)` (all cells attend back to M). Reduces self-attention from `O(n²)` to `O(n)`. Config: **k=128 inducing vectors, 4 heads, 3 ISAB blocks.**
- **Leakage guard:** only *train* cells are keys/values in `MAB₁`, so column statistics depend only on training data.
- **What it learns (Fig. 3):** embeddings cluster columns by skewness/kurtosis — they encode distributional identity, serving as feature identifiers without needing column names.
- Complexity: **`O(nkm)`**, linear in n.

### Stage 2 — Context-aware row-wise interaction (`TF_row`)
- A **3-layer, 8-head** transformer over the feature axis. **4 learnable [CLS] tokens** are prepended to each row; their outputs are **concatenated → a `4d = 512` vector per row.** (`d=128` is kept inside `TF_col`/`TF_row` to save memory; the 4× width appears only here.)
- **Representation collapse & RoPE:** because features are identified only by distribution, identically-distributed features are interchangeable, so permutation-invariant attention can collapse distinct rows to the same vector (e.g. the `balance-scale` dataset, Fig. 4). Fix: **Rotary Positional Embedding (RoPE), base = 100,000** breaks the symmetry. This *gives up* column-permutation invariance, which is then **approximately restored by ensembling over column permutations** at inference.
- Complexity: **`O(m²n)`**, cheap because `m ≪ n`.

### Stage 3 — Dataset-wise in-context learning (`TF_icl`)
- Training labels are **one-hot encoded, linearly mapped into the `4d` space, and added** to the training row embeddings → `H_train`. Test rows get no label.
- A **12-layer, 4-head** transformer processes `[H_train ; H_test]` with a mask: **train rows attend to each other; test rows attend only to train rows.** A **2-layer MLP** decoder maps test outputs → class probabilities.
- Complexity: **`O(n²)`**, but paid **once** since the feature dimension is already collapsed.

**Overall complexity: `O(m²n + n²)`** vs. TabPFNv2's `O(m²n + n²m)`.

## 3. Pretraining (synthetic only) & Inference

### Synthetic prior — Structural Causal Models (SCMs)
- Sample a DAG shaped like a fully-connected MLP; each feature `c = f(Pa(c)) + ε`. Two enrichments over TabPFN v1:
  1. **Tree-based SCMs** — at each DAG layer an **XGBoost** regressor maps parents → children, injecting tree inductive biases and hierarchical interactions.
  2. **Diverse activation functions** — 15 activations beyond {Identity, Tanh, LeakyReLU, ELU}, plus Gaussian-process random-kernel activations, with random per-layer choice, standardization, and rescaling.
- Mixture: **70% MLP-SCMs + 30% tree-based SCMs.**

### Curriculum learning (3 stages, grow sequence length)
1. micro-batch `N_B=4`, fixed size **1,024**, **160K steps**
2. `N_B=1`, size log-uniform **1K–40K**, **2K steps** (activation checkpointing >10K)
3. `N_B=1`, size uniform **40K–60K**, **50 steps**, training **only `TF_icl`** with other components frozen

Each step = 512 datasets, features ≤100, classes ≤10. FlashAttention + AMP. **Total: ~20 days on 3× A100 (40GB)** (16 / 3 / 1 days for stages 1/2/3).

### Hierarchical class extension (>10 classes)
Recursively partition classes into balanced subgroups of ≤10, forming a tree of depth `r = ⌈log₁₀ k⌉`. Internal nodes predict subgroup probabilities; final class probability = product along the root-to-leaf path. **All sub-classifiers reuse the same row embeddings `H` and the same `TF_icl`**, so the extra cost is small.

### Memory-efficient inference
Peak activation memory is modeled by polynomial regression (`α₁·batch + α₂·seq_len + α₃·batch·seq_len + α₄`), enabling dynamic batch sizing plus CPU/disk offloading. Result: **100K samples × 500 features on ~5 GB GPU + 32 GB RAM.**

## 4. Experiments

- **Benchmark:** TALENT — **200 classification datasets** (120 binary, 80 multiclass), 30+ baselines. 15 datasets used by TabPFNv2 for tuning are excluded; main analysis on the **171 datasets with ≤10 classes.**
- Split 64/16/20. **TabICL & TabPFNv2 use training data only (no validation tuning)** — a self-imposed disadvantage. Both **ensemble 32 predictions** (shuffled columns/classes, z-norm ± power transform).

**Key results:**
- **Accuracy:** TabICL has the **best median relative accuracy** over a tuned MLP across all datasets, statistically tied with TabPFNv2; both beat all other methods by a wide margin (critical-difference).
- **Speed:** geom-mean train+inference **1.1 s per 1K samples** for TabICL, vs ~3 min (CatBoost, CPU tuning) / ~7 min (RealMLP, ModernNCA, GPU tuning) — roughly **two orders of magnitude faster** than tuned methods.
- **vs TabPFNv2:** **1.5× faster on small, 3–10× on large** datasets (~5× average on large). E.g. 10K×100 → ~20 s vs 1 m 40 s; 1K×10 → 1 s vs 2 s.
- **Large data:** TabPFNv2 was pretrained only up to 2048 samples and can fail above 30K (subsampled to 30K here); TabICL predicts on the full data. On the **53 datasets >10K samples, TabICL surpasses both TabPFNv2 and CatBoost.**
- **Calibration:** on log loss, TabICL & TabPFNv2 significantly outperform accuracy-tuned competitors (reliable probabilities).
- **>10 classes:** with the hierarchical strategy, TabICL gets the **2nd-best** normalized accuracy across 12 many-class datasets; **TabPFNv2 cannot natively handle >10 classes.**

**Ablations:**
- **Tree-based SCMs help** (Fig. 9 — improvements in accuracy, AUC, log loss).
- **Curriculum learning helps on large data:** TabICL's average rank improves **11.4 (9th) → 7.46 (2nd) → 6.95 (1st)** across the three stages (with a slight drop on some small datasets).

## 5. Limitations

1. Slow inference like other foundation models (caching could help).
2. **Classification only** — regression is left to future work (feasible with similar methodology).
3. **RoPE violates column-permutation invariance** (mitigated, not eliminated, by permutation ensembling).
4. Evaluation inherits TALENT's design choices (single models, holdout tuning; ensembling others could narrow gaps).
5. **Categorical features and missing values** are handled by mean imputation / not passed explicitly, unlike TabPFNv2 which infers categoricals internally.

## Takeaway

TabICL shows that ICL's advantage **persists at scale**: even with abundant training data, synthetic pretraining induces implicit priors that make a single forward pass competitive with (and faster than) heavily-tuned tree and deep-learning baselines — extending tabular foundation models by roughly an order of magnitude in dataset size.
