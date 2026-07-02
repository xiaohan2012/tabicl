"""Reproduction of TabICL paper Figure 3.

"Learned column-wise embeddings encode statistical properties of feature
distributions." We feed columns through TabICL's column-wise embedding
(TF_col), capture the intermediate M matrix (output of MAB1 in the *final*
ISAB block), summarize it to one vector per column, project to 2D with PCA,
and color by each column's skewness / kurtosis.

Paper: Qu et al., "TabICL: A Tabular Foundation Model for In-Context Learning
on Large Data", ICML 2025 (arXiv:2502.05564), Section 3.2 / Figure 3.
"""

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # TabICL Figure 3 — distribution-aware column embeddings

    **Claim (paper §3.2):** the shared Set Transformer `TF_col` learns column
    embeddings that encode *distributional identity* — columns cluster by
    skewness / kurtosis, acting as implicit feature identifiers.

    **What we capture:** `M = MAB₁(V_I, U_train, U_train) ∈ ℝ^{k×d}`, the output
    of the first attention block of the **final** ISAB in `TF_col`
    (`layers.py:558`). We grab it with a forward hook on
    `col_embedder.tf_col.blocks[-1].multihead_attn1`, sum over the `k`
    inducing vectors → one `d`-vector per column, then PCA to 2D.
    """)
    return


@app.cell
def _():
    import os

    # The tree-SCM prior uses xgboost; torch + xgboost both loading OpenMP segfaults
    # DMatrix construction on macOS. Must be set before torch/xgboost pull in libomp.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import platform

    import numpy as np
    import torch
    from scipy import stats
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return PCA, go, make_subplots, np, stats, torch


@app.cell
def _(mo, torch):
    # GPU for the embedding pass (CUDA > MPS > CPU); synthetic generation stays on CPU.
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    is_script = mo.app_meta().mode == "script"
    mo.md(f"**Embedding device:** `{device}` &nbsp;&nbsp; **script mode:** `{is_script}`")
    return device, is_script


@app.cell
def _(torch):
    from huggingface_hub import hf_hub_download
    from tabicl import TabICL

    # Load the v1.1 checkpoint once and keep only the column embedder.
    _ckpt_path = hf_hub_download("jingang/TabICL-clf", "tabicl-classifier-v1.1-0506.ckpt")
    _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=True)
    _model = TabICL(**_ckpt["config"])
    _model.load_state_dict(_ckpt["state_dict"])
    _model.eval()
    col_embedder = _model.col_embedder
    return (col_embedder,)


@app.cell
def _(mo, torch):
    def capture_col_M(col_embedder, cols, device, batch_size=512, train_size=None):
        """Run TF_col and capture M from the final ISAB block via a forward hook.

        cols: (N, T) float tensor of raw columns.
        Returns: (N, d) numpy array — M summed over the k inducing vectors.
        """
        col_embedder = col_embedder.to(device).eval()
        captured = []

        def hook(_module, _inp, out):
            # out: (B, k, d) -> sum over inducing vectors -> (B, d)
            captured.append(out.sum(dim=1).detach().to("cpu"))

        handle = col_embedder.tf_col.blocks[-1].multihead_attn1.register_forward_hook(hook)
        n = cols.shape[0]
        with torch.no_grad(), mo.status.progress_bar(
            total=n, title="Capturing M", subtitle="TF_col forward pass"
        ) as bar:
            for start in range(0, n, batch_size):
                batch = cols[start : start + batch_size].to(device).unsqueeze(-1)  # (B, T, 1)
                col_embedder._compute_embeddings(batch, train_size)
                bar.update(increment=batch.shape[0])
        handle.remove()

        return torch.cat(captured, dim=0).float().numpy()

    return (capture_col_M,)


@app.cell
def _(mo, np, stats, torch):
    from tabicl.prior.dataset import PriorDataset

    # GPU generation is ~5x faster here (benchmarked); joblib parallelism is actually
    # slower for these small per-dataset SCMs, so n_jobs=1. PriorDataset supports
    # only cpu/cuda (not mps), so pick cuda when available else cpu.
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_synthetic_columns(n_columns, seq_len, seed):
        """Pool raw columns from the SCM prior (mix_scm) until we have n_columns."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        prior = PriorDataset(
            batch_size=32,
            batch_size_per_gp=4,
            min_features=2,
            max_features=100,
            max_classes=10,
            min_seq_len=None,
            max_seq_len=seq_len,
            prior_type="mix_scm",
            n_jobs=1,
            device=gen_device,
        )
        cols, skew, kurt = [], [], []
        with mo.status.progress_bar(
            total=n_columns, title="Generating synthetic columns", subtitle="SCM prior (mix_scm)"
        ) as bar:
            while len(cols) < n_columns:
                prev = min(len(cols), n_columns)
                X, _y, d, _sl, _ts = prior.get_batch()
                for i in range(X.shape[0]):
                    di = int(d[i])
                    Xi = X[i, :, :di].float().cpu().numpy()  # (T, di); .cpu() for cuda tensors
                    for j in range(di):
                        c = Xi[:, j]
                        if not np.all(np.isfinite(c)) or np.std(c) < 1e-8:
                            continue
                        cols.append(c)
                        skew.append(float(stats.skew(c)))
                        kurt.append(float(stats.kurtosis(c)))  # excess kurtosis
                bar.update(increment=min(len(cols), n_columns) - prev)
        columns = torch.tensor(np.stack(cols[:n_columns]), dtype=torch.float32)
        return columns, np.array(skew[:n_columns]), np.array(kurt[:n_columns])


    return (generate_synthetic_columns,)


@app.cell
def _(mo):
    # Heavy compute is gated behind this button (see EXPENSIVE.md guidance).
    n_columns_slider = mo.ui.slider(2000, 40000, value=8000, step=1000, label="columns (n)")
    seq_len_slider = mo.ui.slider(256, 2048, value=1024, step=256, label="rows per column (seq_len)")
    run_button = mo.ui.run_button(label="Run capture")
    mo.hstack(
        [n_columns_slider, seq_len_slider, run_button],
        justify="start",
        align="center",
        gap=2,
    )
    return n_columns_slider, run_button, seq_len_slider


@app.cell
def _(
    PCA,
    capture_col_M,
    col_embedder,
    device,
    generate_synthetic_columns,
    is_script,
    mo,
    n_columns_slider,
    run_button,
    seq_len_slider,
):
    # Small, auto-running defaults in script mode; slider-driven + button-gated interactively.
    n_columns = 400 if is_script else n_columns_slider.value
    seq_len = 256 if is_script else seq_len_slider.value

    mo.stop(
        not (is_script or run_button.value),
        mo.md("👆 Set the sliders, then click **Run capture** to generate columns and embed them."),
    )

    with mo.status.spinner(title=f"Generating {n_columns} synthetic columns…"):
        syn_columns, syn_skew, syn_kurt = generate_synthetic_columns(n_columns, seq_len, seed=42)
    with mo.status.spinner(title="Capturing M and running PCA…"):
        syn_M = capture_col_M(col_embedder, syn_columns, device)
        syn_coords = PCA(n_components=2, random_state=42).fit_transform(syn_M)

    mo.md(f"Captured **M** for **{len(syn_M)}** columns · shape `{syn_M.shape}`.")
    return syn_columns, syn_coords, syn_kurt, syn_skew


@app.cell
def _(go, np):
    def scatter_figure(coords, color_vals, label, cmin, cmax):
        idx = np.arange(len(coords))
        fig = go.Figure(
            go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker=dict(
                    size=5,
                    color=color_vals,
                    colorscale="Spectral_r",
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(title=label),
                    opacity=0.8,
                ),
                customdata=np.stack([idx, color_vals], axis=1),
                hovertemplate="col %{customdata[0]}<br>" + label + " %{customdata[1]:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"TabICL column embeddings (M) — colored by {label}",
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            height=520,
            template="plotly_white",
        )
        return fig

    return (scatter_figure,)


@app.cell
def _(mo, scatter_figure, syn_coords, syn_kurt, syn_skew):
    syn_fig_skew = scatter_figure(syn_coords, syn_skew, "skewness", -1.5, 1.5)
    syn_fig_kurt = scatter_figure(syn_coords, syn_kurt, "kurtosis", 0.0, 5.0)
    syn_plot = mo.ui.plotly(syn_fig_skew)
    syn_plot_kurt = mo.ui.plotly(syn_fig_kurt)
    mo.hstack([syn_plot, syn_plot_kurt], widths="equal", gap=1)
    return (syn_plot,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Column distributions (the paper's insets, made interactive)

    The panels below reproduce Figure 3's hand-drawn distribution insets: the
    histograms of representative columns (min / median / max skew, and max
    kurtosis). **Select points** in the scatter above (box/lasso) to instead
    inspect the distributions of your selection.
    """)
    return


@app.cell
def _(make_subplots, np, syn_columns, syn_kurt, syn_plot, syn_skew):
    # Prefer the user's scatter selection; otherwise show representative columns.
    selected = []
    if syn_plot.value:
        for _p in syn_plot.value:
            if isinstance(_p, dict) and "customdata" in _p:
                selected.append(int(_p["customdata"][0]))

    if selected:
        show_idx = selected[:8]
        titles = [f"col {i} · skew {syn_skew[i]:.2f} · kurt {syn_kurt[i]:.2f}" for i in show_idx]
    else:
        rep = {
            "min skew": int(np.argmin(syn_skew)),
            "median skew": int(np.argsort(syn_skew)[len(syn_skew) // 2]),
            "max skew": int(np.argmax(syn_skew)),
            "max kurt": int(np.argmax(syn_kurt)),
        }
        show_idx = list(rep.values())
        titles = [f"{k} · skew {syn_skew[i]:.2f} · kurt {syn_kurt[i]:.2f}" for k, i in rep.items()]

    n = len(show_idx)
    ncols = 2 if n <= 4 else 4
    nrows = int(np.ceil(n / ncols))
    hist_fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles)
    for pos, ci in enumerate(show_idx):
        r, c = pos // ncols + 1, pos % ncols + 1
        hist_fig.add_histogram(x=syn_columns[ci].numpy(), nbinsx=40, row=r, col=c, showlegend=False)
    hist_fig.update_layout(height=240 * nrows, template="plotly_white", bargap=0.02)
    hist_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Bonus: real columns from TabArena (via OpenML)

    Same pipeline on real data. TabArena has no usable PyPI package, so we
    fetch a few of its curated datasets directly from OpenML by task id.
    Real columns are standardized before embedding (skew/kurtosis are
    shift/scale invariant, so the coloring is unaffected). Network-gated —
    does not auto-run in script mode.
    """)
    return


@app.cell
def _(np, stats, torch):
    import openml

    def load_tabarena_columns(task_ids, max_rows=2000, seed=42):
        """Pool standardized numeric columns from TabArena datasets (via OpenML)."""
        rng = np.random.default_rng(seed)
        cols, skew, kurt, names = [], [], [], []
        for tid in task_ids:
            task = openml.tasks.get_task(tid, download_data=True, download_qualities=False)
            ds = task.get_dataset()
            Xdf, _y, _cat, _names = ds.get_data(target=ds.default_target_attribute)
            num = Xdf.select_dtypes(include="number").dropna(axis=1, how="any")
            if len(num) > max_rows:
                num = num.iloc[rng.choice(len(num), max_rows, replace=False)]
            for name in num.columns:
                c = num[name].to_numpy(dtype=np.float64)
                if not np.all(np.isfinite(c)) or np.std(c) < 1e-8:
                    continue
                skew.append(float(stats.skew(c)))
                kurt.append(float(stats.kurtosis(c)))
                cols.append((c - c.mean()) / c.std())  # standardize before embedding
                names.append(f"{ds.name}:{name}")
        # Pad/truncate all columns to a common length for stacking.
        T = min(len(c) for c in cols)
        columns = torch.tensor(np.stack([c[:T] for c in cols]), dtype=torch.float32)
        return columns, np.array(skew), np.array(kurt), names

    return (load_tabarena_columns,)


@app.cell
def _(
    PCA,
    capture_col_M,
    col_embedder,
    device,
    load_tabarena_columns,
    mo,
    scatter_figure,
    tabarena_run,
    tabarena_select,
):
    mo.stop(not tabarena_run.value, mo.md("👆 Pick datasets and click **Fetch & embed TabArena**."))

    with mo.status.spinner(title="Fetching TabArena datasets from OpenML…"):
        ta_columns, ta_skew, ta_kurt, ta_names = load_tabarena_columns(list(tabarena_select.value))
    with mo.status.spinner(title="Capturing M and running PCA…"):
        ta_M = capture_col_M(col_embedder, ta_columns, device)
        ta_coords = PCA(n_components=2, random_state=42).fit_transform(ta_M)

    ta_fig_skew = scatter_figure(ta_coords, ta_skew, "skewness", -1.5, 1.5)
    ta_fig_kurt = scatter_figure(ta_coords, ta_kurt, "kurtosis", 0.0, 5.0)
    ta_fig_skew.update_layout(title=f"TabArena real columns ({len(ta_M)}) — skewness")
    ta_fig_kurt.update_layout(title=f"TabArena real columns ({len(ta_M)}) — kurtosis")
    mo.hstack([ta_fig_skew, ta_fig_kurt], widths="equal", gap=1)

    return


@app.cell
def _(mo):
    # TabArena-v0.1 classification datasets (OpenML task ids); label shows total feature count.
    tabarena_tasks = {
        "Bioresponse (1776f)": 363620,
        "hiva_agnostic (1617f)": 363677,
        "kddcup09_appetency (212f)": 363683,
        "APSFailure (170f)": 363616,
        "MIC (111f)": 363711,
        "taiwanese_bankruptcy_prediction (94f)": 363706,
        "NATICUSdroid (86f)": 363689,
        "coil2000_insurance_policies (85f)": 363624,
        "polish_companies_bankruptcy (64f)": 363694,
        "splice (60f)": 363702,
        "Diabetes130US (47f)": 363630,
        "qsar-biodeg (41f)": 363696,
        "anneal (38f)": 363614,
        "students_dropout_and_academic_success (36f)": 363704,
        "hazelnut-spread-contaminant-detection (30f)": 363674,
        "Marketing_Campaign (25f)": 363684,
        "in_vehicle_coupon_recommendation (24f)": 363681,
        "credit_card_clients_default (23f)": 363627,
        "heloc (23f)": 363676,
        "jm1 (21f)": 363712,
        "customer_satisfaction_in_airline (21f)": 363628,
        "credit-g (20f)": 363626,
        "churn (19f)": 363623,
        "online_shoppers_intention (17f)": 363691,
        "seismic-bumps (15f)": 363700,
        "Is-this-a-good-customer (13f)": 363682,
        "bank-marketing (13f)": 363618,
        "HR_Analytics_Job_Change_of_Data_Scientists (12f)": 363679,
        "SDSS17 (11f)": 363699,
        "GiveMeSomeCredit (10f)": 363673,
        "E-CommereShippingData (10f)": 363632,
        "Bank_Customer_Churn (10f)": 363619,
        "website_phishing (9f)": 363707,
        "Amazon_employee_access (9f)": 363613,
        "diabetes (8f)": 363629,
        "maternal_health_risk (6f)": 363685,
        "Fitness_Club (6f)": 363671,
        "blood-transfusion-service-center (4f)": 363621,
    }
    tabarena_select = mo.ui.multiselect(
        options=tabarena_tasks,
        # default to a feature-rich mix so the scatter is dense (Bioresponse alone ~1776 cols)
        value=["Bioresponse (1776f)", "APSFailure (170f)", "credit-g (20f)", "churn (19f)"],
        label="TabArena datasets",
    )
    tabarena_run = mo.ui.run_button(label="Fetch & embed TabArena")
    mo.hstack([tabarena_select, tabarena_run], justify="start")

    return tabarena_run, tabarena_select


if __name__ == "__main__":
    app.run()
