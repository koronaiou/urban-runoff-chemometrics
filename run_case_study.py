#!/usr/bin/env python3
"""
run_case_study.py
=================
Machine-learning case study for the chemometrics chapter.

Purpose
-------
This script does NOT attempt source apportionment. It demonstrates a validation
framework for identifying compounds with reproducible site-associated variation,
and reports honestly what that framework can and cannot deliver at n = 40. The
transferable product is the procedure; the site-level findings are a worked
example.

What it does
------------
1. Multi-seed, fully nested, event-grouped cross-validation in which the
   algorithm, the feature count and the hyperparameters are all selected inside
   each training partition.
2. Records which algorithm each fold selected, so the stability of the algorithm
   ranking can be reported rather than assumed.
3. A restricted permutation test.
4. Per-site recall with confidence intervals and a confusion matrix.
5. SHAP explanations for one target site: a class-specific beeswarm over all
   out-of-fold samples, plus waterfall plots for correctly classified samples.
6. A convergence test between the SHAP ranking and an independently derived
   marker list (VIP + variance partitioning) supplied as a CSV or via --vip.

Preprocessing matches the chemometrics chapter exactly (occurrence filter,
half-minimum imputation, log10, max-scaling) but is fitted inside each training
partition; see preprocessing.py.

Run
---
    python run_case_study.py --quick                 # ~5 min, smoke test
    python run_case_study.py                         # defaults, ~1-3 h
    python run_case_study.py --seeds 10 --trials 60 --perm 500 --jobs 8

Runtime scales as seeds x folds x trials x inner_splits. Use --jobs to give the
tree ensembles more cores.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
)

warnings.filterwarnings("ignore")

from compound_annotation import CLASS_LABEL, annotate
from data_prep import load_dataset
from nested_selection import (
    ALGORITHMS, build_pipeline, default_config, get_outer_splits_seeded,
    nested_run, permutation_test, LabelCodec,
)

# Markers reported in the chemometrics chapter (VIP > 1 and site variance > 20%).
# Override with --vip-csv pointing to a file with a `compound` column.
DEFAULT_VIP_MARKERS = [
    "Perfluorooctanoic acid (PFOA)",
    "8H-Perfluorooctane",
    "Perfluoro-2-methyl-3-(propan-2-yl)pentan-3-yl",
    "1-Hydroperfluoroheptane",
    "Perfluorononanoic acid (PFNA)",
    "Perfluorodecanoic acid (PFDA)",
    "Octadecyl hydrogen sulfate",
    "Stearoyl Ethanolamide",
    "Tectorigenin",
    "Grandiflorenic acid",
    "Cocaine",
]

SITE_COLORS = {
    "1 - Agias Sofias Str": "#2E5E8E",
    "2 - Ethnikis Aminis Str": "#C24B3A",
    "3 - Lagkada Str": "#3F7C56",
    "4 - Thessaloniki Port": "#B8892B",
}


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def all_metrics(y_true, y_pred, labels):
    kw = dict(labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, **kw),
        "precision_macro": precision_score(y_true, y_pred, **kw),
        "recall_macro": recall_score(y_true, y_pred, **kw),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
    }


# =========================================================================== #
def step_multiseed(Xr, y, meta, seeds, trials, scheme, jobs, outdir, inner_splits=3):
    banner(f"STEP 1  Multi-seed nested cross-validation "
           f"({len(seeds)} seeds x {trials} trials, algorithm selected inside folds)")
    labels = sorted(pd.unique(y))
    runs, fold_tables, per_seed = [], [], []

    for s in seeds:
        t0 = time.time()
        r = nested_run(Xr, y, meta, scheme=scheme, n_trials=trials, seed=s,
                       inner_splits=inner_splits, n_jobs=jobs, verbose=False)
        runs.append(r)
        fold_tables.append(r["fold_table"])
        m = all_metrics(y, r["oof_pred"], labels)
        b = all_metrics(y, r["oof_baseline"], labels)
        per_seed.append({"seed": s, **{f"model_{k}": v for k, v in m.items()},
                         **{f"baseline_{k}": v for k, v in b.items()}})
        picks = r["fold_table"]["algorithm"].value_counts().to_dict()
        print(f"  seed {s:2d}: BA={m['balanced_accuracy']:.3f}  F1={m['f1_macro']:.3f}  "
              f"[{time.time()-t0:.0f}s]  algorithms chosen: {picks}", flush=True)

    ps = pd.DataFrame(per_seed)
    folds = pd.concat(fold_tables, ignore_index=True)
    ps.to_csv(f"{outdir}/table_ml1_per_seed_metrics.csv", index=False)
    folds.to_csv(f"{outdir}/table_ml2_fold_selections.csv", index=False)

    print("\n  Across seeds (this is the headline estimate):")
    for k in ["accuracy", "balanced_accuracy", "f1_macro",
              "precision_macro", "recall_macro", "mcc"]:
        v = ps[f"model_{k}"]
        print(f"    {k:20s} {v.mean():.3f} +/- {v.std(ddof=1):.3f}   "
              f"(range {v.min():.3f}-{v.max():.3f})")
    print(f"    {'baseline BA':20s} {ps['baseline_balanced_accuracy'].mean():.3f}")

    print("\n  Algorithm selected by the inner loop, across all folds and seeds:")
    vc = folds["algorithm"].value_counts()
    for a in ALGORITHMS:
        n = int(vc.get(a, 0))
        print(f"    {a:20s} {n:4d} / {len(folds)}  ({n/len(folds)*100:5.1f}%)")
    print(f"\n  Feature count k: median {int(folds['k'].median())}, "
          f"range {int(folds['k'].min())}-{int(folds['k'].max())}")
    return ps, folds, runs


def fig_algorithm_stability(folds, ps, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    vc = folds["algorithm"].value_counts().reindex(ALGORITHMS).fillna(0)
    ax.barh(range(len(ALGORITHMS)), vc.values / len(folds) * 100,
            color="#2E5E8E", edgecolor="white", linewidth=.6)
    ax.set_yticks(range(len(ALGORITHMS)))
    ax.set_yticklabels(ALGORITHMS, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Folds in which this algorithm was selected (%)")
    ax.set_title("a  Which algorithm the inner loop chose", loc="left", fontweight="bold")
    for i, v in enumerate(vc.values / len(folds) * 100):
        ax.text(v + .8, i, f"{v:.0f}%", va="center", fontsize=7)

    ax = axes[1]
    v = ps["model_balanced_accuracy"]
    ax.plot(ps["seed"], v, "o-", color="#2E5E8E", ms=6, lw=1.5, label="Nested estimate")
    ax.axhline(v.mean(), color="#2E5E8E", ls=":", lw=1.2,
               label=f"Mean = {v.mean():.3f}")
    ax.fill_between(ps["seed"], v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1),
                    color="#2E5E8E", alpha=.12, label=f"±1 SD = {v.std(ddof=1):.3f}")
    ax.axhline(0.25, color="#C24B3A", ls="--", lw=1.3, label="Chance / majority baseline")
    ax.set_xlabel("Seed"); ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("b  Stability across repeated runs", loc="left", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{figdir}/FigML1_algorithm_stability.png")
    plt.close(fig)


# =========================================================================== #
def step_site_performance(runs, y, target_site, outdir, figdir):
    banner("STEP 2  Per-site performance")
    labels = sorted(pd.unique(y))
    short = [l.split(" - ")[0] for l in labels]

    # recall per site per seed
    rec = {l: [] for l in labels}
    for r in runs:
        cm = confusion_matrix(y, r["oof_pred"], labels=labels)
        for i, l in enumerate(labels):
            rec[l].append(cm[i, i] / cm.sum(1)[i])
    rows = []
    for l in labels:
        v = np.array(rec[l])
        k = int(round(v.mean() * 10)); n = 10
        lo, hi = st.beta.interval(0.95, k + .5, n - k + .5)
        rows.append({"site": l, "mean_recall": v.mean(), "sd_recall": v.std(ddof=1),
                     "ci_low": lo, "ci_high": hi, "n_samples": n})
    rt = pd.DataFrame(rows)
    rt.to_csv(f"{outdir}/table_ml3_per_site_recall.csv", index=False)
    print(rt.round(3).to_string(index=False))

    print("\n  Pairwise overlap of 95% intervals (do NOT claim a site is "
          "distinguishable\n  from another when its interval overlaps):")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            ov = not (rt.ci_high[i] < rt.ci_low[j] or rt.ci_high[j] < rt.ci_low[i])
            print(f"    {short[i]} vs {short[j]}: "
                  f"{'OVERLAPS - not separable' if ov else 'separable'}")

    # pooled confusion over the modal prediction across seeds
    preds = np.array([r["oof_pred"] for r in runs])
    modal = np.array([pd.Series(preds[:, i]).mode()[0] for i in range(len(y))])
    cm = confusion_matrix(y, modal, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        f"{outdir}/table_ml4_confusion_modal.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("bl", ["#FFFFFF", "#2E5E8E"])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() * .6 else "#222")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(short)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(short)
    ax.set_xlabel("Predicted site"); ax.set_ylabel("True site"); ax.grid(False)
    ax.set_title("a  Confusion matrix (modal prediction across seeds)",
                 loc="left", fontweight="bold")

    ax = axes[1]
    ax.bar(range(len(labels)), rt.mean_recall,
           color=[SITE_COLORS.get(l, "#888") for l in labels],
           edgecolor="white", linewidth=.8)
    ax.errorbar(range(len(labels)), rt.mean_recall,
                yerr=[rt.mean_recall - rt.ci_low, rt.ci_high - rt.mean_recall],
                fmt="none", ecolor="#333", lw=1.2, capsize=5)
    ax.axhline(0.25, color="#C24B3A", ls="--", lw=1.3)
    ax.text(len(labels) - .55, .27, "chance", fontsize=7, color="#C24B3A", ha="right")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(short)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Recall (95% Jeffreys CI)")
    ax.set_title("b  Per-site recall — intervals overlap",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{figdir}/FigML2_site_performance.png")
    plt.close(fig)
    return rt, modal


# =========================================================================== #
def step_permutation(Xr, y, meta, scheme, n_perm, mode, seed, jobs, outdir, figdir,
                     trials=8):
    banner(f"STEP 3  Restricted permutation test (mode={mode}, {n_perm} permutations)")
    r = permutation_test(Xr, y, meta, scheme=scheme, n_perm=n_perm, seed=seed,
                         mode=mode, n_trials=trials, n_jobs=jobs, verbose=True)
    print(f"\n    observed ........... {r['observed']:.3f}")
    print(f"    null mean +/- sd ... {r['null_mean']:.3f} +/- {r['null_sd']:.3f}")
    print(f"    null 95th pct ...... {r['null_q95']:.3f}")
    print(f"    p-value ............ {r['p_value']:.4f}")
    if mode == "fixed":
        print("\n    NOTE: in 'fixed' mode the null does not include the cost of\n"
              "    selecting an algorithm. Re-run with --perm-mode nested for a\n"
              "    null that reflects the full selection procedure.")
    pd.DataFrame({"null": r["null"]}).to_csv(
        f"{outdir}/table_ml5_permutation_null.csv", index=False)
    with open(f"{outdir}/permutation_summary.json", "w") as f:
        json.dump({k: v for k, v in r.items() if k != "null"}, f, indent=2)

    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.hist(r["null"], bins=25, color="#B8C4CE", edgecolor="white", linewidth=.6,
            label="Null: site labels shuffled within each rain event")
    ax.axvline(r["observed"], color="#C24B3A", lw=2,
               label=f"Observed = {r['observed']:.3f}")
    ax.axvline(r["null_q95"], color="#2E5E8E", ls="--", lw=1.3,
               label=f"Null 95th pct = {r['null_q95']:.3f}")
    ax.set_xlabel("Balanced accuracy"); ax.set_ylabel("Permutations")
    ax.set_title(f"Restricted permutation test (p = {r['p_value']:.4f})",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=6.8)
    fig.tight_layout()
    fig.savefig(f"{figdir}/FigML3_permutation.png")
    plt.close(fig)
    return r


# =========================================================================== #
def _normalise_sv(sv):
    """Coerce any SHAP return shape to (n_samples, n_features, n_classes)."""
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    if sv.ndim == 2:
        sv = sv[:, :, None]
    return sv


def _background(pipe, Xr, tr, max_bg=32):
    """Transformed training data used as the reference distribution."""
    Ztr = pipe.named_steps["select"].transform(
        pipe.named_steps["prep"].transform(Xr[tr]))
    if "scale" in pipe.named_steps:
        Ztr = pipe.named_steps["scale"].transform(Ztr)
    if len(Ztr) > max_bg:
        idx = np.linspace(0, len(Ztr) - 1, max_bg).astype(int)
        Ztr = Ztr[idx]
    return Ztr


def _safe_shap(clf, Z, pipe, Xr, tr, backend="permutation", n_fallback=0, seed=0):
    """
    Compute SHAP values for one fold.

    backend="permutation" (default) is model-agnostic: it only calls the
    model's predict_proba/decision_function, which is the same code path
    already exercised during cross-validation. It avoids SHAP's compiled
    tree-traversal code, which can terminate the interpreter outright
    (an access violation, not a Python exception) when the installed SHAP
    build does not match the installed XGBoost/LightGBM/CatBoost build.
    Because the matrices here are small (tens of samples, <=30 features) the
    model-agnostic route costs seconds per fold, so it is the sensible default.

    backend="tree" uses TreeExplainer and is faster, but carries that risk.
    backend="kernel" is the older model-agnostic sampler.

    The permutation and kernel estimators draw their reference samples
    stochastically. `seed` is passed to the explainer so that repeated runs of
    this script reproduce identical attribution values, not merely identical
    rankings; without it the third decimal place drifts between runs and the
    output is not bit-reproducible.

    Returns (n_samples, n_features, n_classes), or None if this fold cannot be
    explained, so one failure does not abort the analysis.
    """
    import shap
    f = (clf.predict_proba if hasattr(clf, "predict_proba")
         else clf.decision_function)
    try:
        if backend == "tree":
            return _normalise_sv(shap.TreeExplainer(clf).shap_values(Z))
        bg = _background(pipe, Xr, tr)
        if backend == "kernel":
            ke = shap.KernelExplainer(f, shap.kmeans(bg, min(8, len(bg))))
            try:
                return _normalise_sv(ke.shap_values(Z, nsamples=200, silent=True,
                                                    seed=seed))
            except TypeError:
                return _normalise_sv(ke.shap_values(Z, nsamples=200))
        try:
            ex = shap.PermutationExplainer(f, bg, seed=seed)
        except TypeError:          # older shap without the seed argument
            np.random.seed(seed)
            ex = shap.PermutationExplainer(f, bg)
        try:
            return _normalise_sv(ex(Z, max_evals=2 * Z.shape[1] + 1).values)
        except TypeError:
            return _normalise_sv(ex(Z).values)
    except Exception as e:
        print(f"      (SHAP unavailable for one fold: "
              f"{type(e).__name__}: {e})", flush=True)
        return None


def step_shap_target(Xr, y, meta, compounds, runs, target_site, scheme,
                     jobs, outdir, figdir, top_n=15, min_stability=0.5,
                     shap_backend="permutation"):
    banner(f"STEP 4  SHAP explanations for the target site: {target_site}")
    import shap

    labels = sorted(pd.unique(y))
    ci = labels.index(target_site)
    codec = LabelCodec(y)
    yc = codec.encode(y)
    n_comp = len(compounds)

    sv_target = np.full((len(runs), len(y), n_comp), np.nan)
    xv_target = np.full((len(runs), len(y), n_comp), np.nan)
    n_failed = 0

    for ri, r in enumerate(runs):
        splits, _, _ = get_outer_splits_seeded(scheme, y, meta, r["seed"])
        for fi, (tr, te) in enumerate(splits):
            cfg = r["configs"][fi]
            pipe = build_pipeline(cfg, r["seed"], jobs)
            pipe.fit(Xr[tr], yc[tr])
            kept = pipe.named_steps["prep"].get_kept_indices()
            sel_local = pipe.named_steps["select"].get_support(indices=True)
            orig_idx = kept[sel_local]

            Z = pipe.named_steps["select"].transform(
                pipe.named_steps["prep"].transform(Xr[te]))
            if "scale" in pipe.named_steps:
                Z = pipe.named_steps["scale"].transform(Z)
            clf = pipe.named_steps["clf"]
            sv = _safe_shap(clf, Z, pipe, Xr, tr, backend=shap_backend,
                            n_fallback=len(orig_idx),
                            seed=r["seed"] * 7919 + fi)
            if sv is None:
                n_failed += 1
                continue
            c = min(ci, sv.shape[2] - 1)
            for a, gi in enumerate(te):
                sv_target[ri, gi, orig_idx] = sv[a, :, c]
                xv_target[ri, gi, orig_idx] = Z[a]

    mean_sv = np.nanmean(sv_target, axis=0)
    mean_xv = np.nanmean(xv_target, axis=0)
    coverage = (~np.isnan(sv_target)).any(axis=0).mean(axis=0)
    # A compound absent from a fold's model contributes exactly zero to that
    # sample's prediction, so unselected entries are treated as 0 rather than
    # skipped. Averaging only over the folds where a compound happened to be
    # selected would inflate compounds picked up once with an extreme value.
    imp = np.nan_to_num(np.abs(mean_sv), nan=0.0).mean(axis=0)

    if n_failed:
        print(f"  NOTE: SHAP could not be computed for {n_failed} fold(s); "
              f"they are excluded from the ranking.")
    if np.all(np.isnan(sv_target)):
        print("  SHAP could not be computed for any fold - skipping step 4.")
        return None

    cls, prio, _ = annotate(compounds)
    gi_df = pd.DataFrame({
        "compound": compounds, "mean_abs_shap_target": imp,
        "selection_stability": coverage, "source_class": cls,
        "priority_contaminant": prio,
    }).sort_values("mean_abs_shap_target", ascending=False).reset_index(drop=True)
    gi_df.to_csv(f"{outdir}/table_ml6_shap_target_site.csv", index=False)

    print(f"  compounds selected in >={min_stability*100:.0f}% of folds: "
          f"{(coverage >= min_stability).sum()}")
    gi_df = gi_df[gi_df.selection_stability >= min_stability].reset_index(drop=True)
    print(f"  ranking restricted to those, so a compound selected once with an\n"
          f"  extreme value cannot top the list")
    print(f"\n  top {top_n} compounds for '{target_site}':")
    for _, rr in gi_df.head(top_n).iterrows():
        print(f"    {rr['mean_abs_shap_target']:.4f}  stab={rr['selection_stability']*100:3.0f}%  "
              f"{rr['compound'][:44]:44s} {rr['source_class']}")

    # ---- beeswarm for the target class ----------------------------------
    top_n = min(top_n, len(gi_df))
    top_idx = [compounds.index(c) for c in gi_df.head(top_n)["compound"]]
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * top_n + 1.8))
    rng = np.random.default_rng(0)
    for row, j in enumerate(top_idx[::-1]):
        s = mean_sv[:, j]; xvals = mean_xv[:, j]
        ok = ~np.isnan(s)
        s, xvals = s[ok], xvals[ok]
        if len(s) == 0:
            continue
        norm = ((xvals - np.nanmin(xvals)) /
                (np.nanmax(xvals) - np.nanmin(xvals) + 1e-12))
        jitter = rng.normal(0, .085, len(s))
        ax.scatter(s, row + jitter, c=norm, cmap="coolwarm", s=26,
                   edgecolor="white", linewidth=.35, vmin=0, vmax=1, zorder=3)
    ax.axvline(0, color="#333", lw=.9)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([c[:44] for c in gi_df.head(top_n)["compound"]][::-1], fontsize=7.5)
    ax.set_xlabel(f"SHAP value for the '{target_site}' class\n"
                  "(positive = pushes the sample towards this site)")
    ax.set_title(f"SHAP summary for {target_site} — all out-of-fold samples",
                 loc="left", fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])          # required by matplotlib < 3.6
    cb = fig.colorbar(sm, ax=ax, pad=.015, aspect=30)
    cb.set_label("Relative compound abundance (low → high)", fontsize=7.5)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    fig.tight_layout()
    fig.savefig(f"{figdir}/FigML4_shap_beeswarm_target.png")
    plt.close(fig)

    # ---- waterfalls for correctly classified target samples --------------
    preds = np.array([r["oof_pred"] for r in runs])
    modal = np.array([pd.Series(preds[:, i]).mode()[0] for i in range(len(y))])
    tp = np.where((y == target_site) & (modal == target_site))[0]
    show = tp[:3]
    if len(show):
        fig, axes = plt.subplots(1, len(show), figsize=(5.0 * len(show), 4.6))
        axes = np.atleast_1d(axes)
        for ax, gi in zip(axes, show):
            s = mean_sv[gi]
            ok = np.where(~np.isnan(s))[0]
            order = ok[np.argsort(np.abs(s[ok]))[-10:]]
            cols = ["#C24B3A" if s[j] > 0 else "#2E5E8E" for j in order]
            ax.barh(range(len(order)), s[order], color=cols,
                    edgecolor="white", linewidth=.5)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([compounds[j][:32] for j in order], fontsize=6.8)
            ax.axvline(0, color="#333", lw=.8)
            ax.set_xlabel("SHAP value", fontsize=7.5)
            ax.set_title(f"{meta['Sample'].values[gi]}\n"
                         f"{meta['event_date'].values[gi]}", fontsize=8,
                         fontweight="bold")
        fig.suptitle(f"Local explanations for correctly classified {target_site} "
                     "samples (red pushes towards the site)",
                     fontweight="bold", x=.04, ha="left", y=1.02)
        fig.tight_layout()
        fig.savefig(f"{figdir}/FigML5_shap_local_target.png")
        plt.close(fig)
    print(f"\n  correctly classified {target_site} samples: {len(tp)}/10 "
          f"(waterfalls drawn for {len(show)})")
    return gi_df


# =========================================================================== #
def step_convergence(gi_df, vip_markers, compounds, outdir, figdir, top_n=20):
    banner("STEP 5  Convergence with the independent VIP / variance marker list")
    present = [m for m in vip_markers if m in compounds]
    missing = [m for m in vip_markers if m not in compounds]
    if missing:
        print(f"  WARNING: {len(missing)} marker name(s) not found in the data: {missing}")

    gi_df = gi_df.reset_index(drop=True)
    gi_df["shap_rank"] = np.arange(1, len(gi_df) + 1)
    sub = gi_df[gi_df.compound.isin(present)][
        ["shap_rank", "compound", "mean_abs_shap_target",
         "selection_stability", "source_class"]]
    overlap = int((sub.shap_rank <= top_n).sum())
    N, K, n = len(compounds), top_n, len(present)
    expected = K * n / N
    p = st.hypergeom.sf(overlap - 1, N, K, n)

    sub.to_csv(f"{outdir}/table_ml7_vip_shap_convergence.csv", index=False)
    print(sub.round(4).to_string(index=False))
    print(f"\n  Markers in the SHAP top {top_n} of {N}: {overlap}/{n}")
    print(f"  Expected by chance ..................: {expected:.2f}")
    print(f"  Hypergeometric p ....................: {p:.3g}")
    print("\n  Two methodologically independent analyses (VIP + variance "
          "partitioning\n  vs out-of-fold SHAP) converging on the same compounds is "
          "stronger evidence\n  than either ranking alone, and does not depend on the "
          "classifier being accurate.")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    s = sub.sort_values("shap_rank")
    ax.scatter(s.shap_rank, range(len(s)), s=70, color="#2E5E8E",
               edgecolor="white", linewidth=.6, zorder=3)
    ax.axvline(top_n + .5, color="#C24B3A", ls="--", lw=1.3,
               label=f"SHAP top {top_n}")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([c[:42] for c in s.compound], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(f"Rank by out-of-fold SHAP importance (of {N} compounds)")
    ax.set_title(f"Independent marker list vs SHAP ranking\n"
                 f"{overlap}/{n} in the top {top_n}; {expected:.1f} expected by chance; "
                 f"p = {p:.2g}", loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{figdir}/FigML6_vip_shap_convergence.png")
    plt.close(fig)
    return {"overlap": overlap, "n_markers": n, "top_n": top_n,
            "expected": expected, "p_value": float(p)}


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--perm", type=int, default=300)
    ap.add_argument("--perm-mode", choices=["fixed", "nested"], default="fixed")
    ap.add_argument("--perm-trials", type=int, default=8)
    ap.add_argument("--scheme", default="lodo_date",
                    choices=["lodo_date", "loeo_episode", "loeo_rainday", "stratified5"])
    ap.add_argument("--target-site", default="3 - Lagkada Str")
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--shap-backend", default="permutation",
                    choices=["permutation", "tree", "kernel"],
                    help="permutation (default) avoids SHAP's compiled tree code, "
                         "which can crash the interpreter on some Windows "
                         "SHAP/XGBoost/LightGBM combinations; tree is faster")
    ap.add_argument("--skip-shap", action="store_true",
                    help="Skip steps 4 and 5 entirely")
    ap.add_argument("--min-stability", type=float, default=0.5,
                    help="Minimum fraction of folds a compound must be selected "
                         "in before it is eligible for the SHAP ranking")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--vip-csv", default=None,
                    help="CSV with a `compound` column; defaults to the built-in list")
    ap.add_argument("--datadir", default=".")
    ap.add_argument("--outdir", default="results_ml")
    ap.add_argument("--figdir", default="figures_ml")
    ap.add_argument("--quick", action="store_true",
                    help="Fast smoke test: 2 seeds, 5 trials, 20 permutations")
    args = ap.parse_args()

    if args.quick:
        args.seeds, args.trials, args.perm = 2, 5, 20

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)
    setup_style()
    t0 = time.time()

    banner("STEP 0  Dataset")
    try:
        import shap as _shap, sklearn as _sk, matplotlib as _mpl
        print(f"  versions: python={__import__('sys').version.split()[0]}  "
              f"sklearn={_sk.__version__}  shap={_shap.__version__}  "
              f"matplotlib={_mpl.__version__}  numpy={np.__version__}")
    except Exception as _e:
        print(f"  (version probe failed: {_e})")
    X, y, meta, compounds = load_dataset(args.datadir)
    Xr = X.values
    print(f"  samples {X.shape[0]}, compounds {X.shape[1]} (p > N)")
    print(f"  sites: {sorted(pd.unique(y))}")
    print(f"  scheme: {args.scheme}   target site: {args.target_site}")
    print(f"  seeds: {args.seeds}   trials/fold: {args.trials}   jobs: {args.jobs}")
    if args.target_site not in set(y):
        raise SystemExit(f"target site not found: {args.target_site}")

    seeds = list(range(args.seeds))
    ps, folds, runs = step_multiseed(Xr, y, meta, seeds, args.trials,
                                     args.scheme, args.jobs, args.outdir,
                                     args.inner_splits)
    fig_algorithm_stability(folds, ps, args.figdir)
    step_site_performance(runs, y, args.target_site, args.outdir, args.figdir)
    step_permutation(Xr, y, meta, args.scheme, args.perm, args.perm_mode,
                     0, args.jobs, args.outdir, args.figdir, args.perm_trials)
    if args.skip_shap:
        print("\n  --skip-shap given; steps 4 and 5 not run.")
        gi = None
    else:
      try:
        gi = step_shap_target(Xr, y, meta, compounds, runs, args.target_site,
                              args.scheme, args.jobs, args.outdir, args.figdir,
                              min_stability=args.min_stability,
                              shap_backend=args.shap_backend)
      except Exception:
        import traceback
        print("\n  STEP 4 FAILED - the cross-validation results above are "
              "unaffected and\n  have already been written to disk. Full error:\n")
        traceback.print_exc()
        gi = None

    if gi is not None:
        try:
            vip = (pd.read_csv(args.vip_csv)["compound"].tolist()
                   if args.vip_csv else DEFAULT_VIP_MARKERS)
            step_convergence(gi, vip, compounds, args.outdir, args.figdir)
        except Exception:
            import traceback
            print("\n  STEP 5 FAILED. Full error:\n")
            traceback.print_exc()

    with open(f"{args.outdir}/run_config.json", "w") as f:
        json.dump({**vars(args), "runtime_min": (time.time() - t0) / 60}, f, indent=2)
    banner(f"DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
