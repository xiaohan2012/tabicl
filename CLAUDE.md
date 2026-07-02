# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TabICL is a tabular foundation model for in-context learning on classification tasks (ICML 2025, [arXiv:2502.05564](https://arxiv.org/abs/2502.05564)). Like TabPFN, `fit()` does minimal work while `predict()` runs the full model — in-context learning happens at prediction time by attending to the labeled training rows. Pretrained checkpoints (~100MB) are auto-downloaded from Hugging Face Hub on first use.

## Commands

Development uses [Hatch](https://hatch.pypa.io/) with `uv` as the installer.

```bash
pip install -e .              # editable install for local dev

hatch test                    # run the full test suite (what CI runs)
hatch test tests/test_sklearn.py::test_sklearn_compatible_estimator   # single test
hatch run types:check         # mypy type checking

# Training (reproduce the paper's 3-stage curriculum learning)
bash scripts/train_stage1.sh  # edit the /path/to placeholders first
# stages 2 and 3 continue from the previous checkpoint
```

Notes:
- `pyproject.toml` sets `pythonpath = "src"` so tests import `tabicl`, not `src.tabicl`.
- The test suite is sklearn's estimator conformance checks (`parametrize_with_checks`) run against `TabICLClassifier(n_estimators=2)`. Any change to the sklearn interface must keep these green across Python 3.9–3.13.
- Training requires `swig` (installed in CI) and typically a GPU; it is launched via `torchrun`.

## Architecture

The package has two largely independent halves: **inference** (`model/` + `sklearn/`, shipped to end users) and **training/data generation** (`prior/` + `train/`, used only to pretrain checkpoints).

### The model: three sequential stages (`src/tabicl/model/`)

`TabICL` (`model/tabicl.py`) is the orchestrator. Its `forward` composes three submodules in order — the feature dimension is collapsed to one vector per row *before* the expensive ICL attention, which is what lets TabICL scale (`O(m²n + n²)` vs. TabPFNv2's `O(m²n + n²m)`):

1. **`ColEmbedding` (`embedding.py`) — column-wise, inter-sample.** Maps each scalar cell to a `d=128` vector whose embedding reflects the statistics of its whole column. Columns share one `SetTransformer` (a hypernetwork) built from induced self-attention blocks (ISAB), giving linear cost in the number of rows. **Leakage guard:** `train_size` is threaded down so the induced column summary only sees training rows.
2. **`RowInteraction` (`interaction.py`) — row-wise, inter-feature.** Prepends 4 learnable CLS tokens, runs a transformer across the feature axis with **rotary positional encoding (RoPE)**, and concatenates the 4 CLS outputs into one `4d = 512` vector per row. RoPE (`model/rope.py`) breaks the symmetry between identically-distributed features that would otherwise collapse distinct rows; this sacrifices column-permutation invariance, restored approximately by ensembling over permutations at inference.
3. **`ICLearning` (`learning.py`) — dataset-wise ICL.** Training labels are one-hot encoded, projected into the `4d` space, and **added** to training row embeddings (labels enter *only* here). A 12-layer transformer processes `[train ; test]` with a mask (train↔train attention, test→train only), then a 2-layer MLP decoder emits class logits. **Hierarchical classification** (`ClassNode`) handles datasets with more than `max_classes=10` classes by recursively splitting classes into a tree.

Shared building blocks live in `encoders.py` (`Encoder`, `SetTransformer`), `layers.py` (attention blocks, `SkippableLinear`, `OneHotAndLinear`, `ClassNode`), and `attention.py`.

### Memory-efficient inference (`model/inference.py`, `inference_config.py`)

A distinguishing feature: `MemoryEstimator` predicts peak activation memory per component from profiled regression coefficients, `InferenceManager` dynamically picks batch sizes, offloads intermediates to CPU, and recovers from OOM by shrinking the batch. `InferenceConfig` / `MgrConfig` expose fine-grained control. When touching the forward passes, keep the `mgr_config` plumbing intact — each module's `forward` accepts it to drive this memory management.

### sklearn interface (`src/tabicl/sklearn/`)

`TabICLClassifier` (`classifier.py`) is the public API. `fit()` loads the checkpoint and preprocesses; `predict()` runs the model over an **ensemble** of transformed dataset views (`n_estimators=32` by default) and averages. Ensemble diversity comes from `EnsembleGenerator` (`preprocessing.py`): different normalization methods (`none`/`power`/`quantile`/`robust`), feature permutations (`latin`/`shift`/`random`), and cyclic class-label shifts. `TransformToNumerical` ordinal-encodes categoricals and imputes missing values when the input is a DataFrame (numpy input is assumed already encoded). `sklearn_utils.py` holds version-agnostic sklearn shims.

### Training & synthetic priors (`src/tabicl/prior/`, `src/tabicl/train/`)

TabICL is pretrained purely on synthetic data generated on the fly from structural causal models. `prior/` samples SCMs (`mlp_scm.py`, `tree_scm.py`, mixed via `prior_config.py`/`hp_sampling.py`), turns regression targets into classification (`reg2cls.py`), and serves batches (`dataset.py`, `genload.py`). `train/run.py` is the `torchrun` entry point (supports DDP); `train_config.py` builds the CLI arg parser and `optim.py` provides LR schedulers. The 3-stage curriculum (scripts/) grows sequence length and freezes earlier components across stages.

## Conventions

- All model modules use `from __future__ import annotations` and are documented with NumPy-style docstrings on every class — match that density when adding parameters.
- `checkpoint_version` selects among published checkpoints; `'tabicl-classifier-v1.1-0506.ckpt'` is the current default and best-performing one. `'tabicl-classifier-v1-0208.ckpt'` reproduces the original paper results.
- Version lives in `src/tabicl/__about__.py` (Hatch reads it from there).

## Note on stray files

The untracked `TABICL_ARCHITECTURE.md` and `TABICL_col_embedding_notes.md` at the repo root are personal scratch notes and may be out of date (some describe a `layers_info` intervention API from a different fork that does **not** exist in this code). Treat the source as ground truth, not those files.
