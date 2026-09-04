"""
nested_selection.py
===================
Cross-validation engine in which the *choice of algorithm* is made inside the
training partition, alongside the hyperparameters and the feature count.

Why the algorithm choice has to be nested too
---------------------------------------------
A common protocol is to run several algorithms under cross-validation and report
the best one. That best value is the maximum of several correlated estimates, so
it is optimistically biased for the same reason that a hyperparameter search
tuned against the reported metric is biased — the selection simply happens one
level up. On this dataset the six algorithms span 0.30 to 0.575 balanced
accuracy with a mean of 0.44, so reporting the maximum overstates by roughly
0.13 relative to the average candidate.

Here the algorithm is one more categorical decision taken by the inner search on
training data only. Each outer fold independently chooses its own algorithm,
feature count and hyperparameters; the held-out fold is predicted once. The
resulting score therefore estimates the generalisation performance of *the whole
selection procedure*, which is the quantity a future user of the workflow would
actually obtain — not the performance of the luckiest candidate.

A useful by-product is the record of which algorithm each fold selected. When
that record is unstable, the algorithm ranking is not resolvable at the
available sample size, and reporting a single "best algorithm" would be
misleading. That record is returned in `fold_table["algorithm"]`.

Repeated runs
-------------
Leave-one-event-out folds are fixed by the sampling design, so repeating the run
with different seeds does not resample the outer splits. What it does vary is
the inner split assignment, the Optuna sampling sequence and model
stochasticity. Repetition therefore quantifies the stability of the *procedure*,
not split-to-split variability, and should be described that way.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import optuna
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from preprocessing import ChapterPreprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

ALGORITHMS = ["RandomForest", "LightGBM", "XGBoost", "CatBoost",
              "LinearSVC", "LogisticRegression"]
K_GRID = [5, 10, 15, 20, 25, 30]


# --------------------------------------------------------------------------- #
def _predict(pipe, X):
    """CatBoost returns a column vector; flatten so all models behave alike."""
    return np.asarray(pipe.predict(X)).ravel().astype(int)


class LabelCodec:
    """Site names <-> contiguous integer codes (XGBoost requires integers)."""

    def __init__(self, y):
        self.classes_ = np.array(sorted(np.unique(y)))
        self._to = {c: i for i, c in enumerate(self.classes_)}

    def encode(self, y):
        return np.array([self._to[v] for v in y], dtype=int)

    def decode(self, c):
        return self.classes_[np.asarray(c, dtype=int)]


# --------------------------------------------------------------------------- #
def suggest_config(trial, n_train, algorithms=ALGORITHMS, standardize=False):
    """Sample an algorithm together with its hyperparameters and feature count."""
    k_max = max(3, min(30, n_train - 2))
    alg = trial.suggest_categorical("algorithm", algorithms)
    cfg = {
        "algorithm": alg,
        "k": trial.suggest_categorical("k", [k for k in K_GRID if k <= k_max] or [k_max]),
        "standardize": standardize,
    }
    if alg == "RandomForest":
        cfg.update(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 500, step=100),
            max_depth=trial.suggest_categorical("rf_max_depth", [2, 3, 5, 8, None]),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 4),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2", 0.5]),
            class_weight=trial.suggest_categorical("rf_class_weight", [None, "balanced"]),
        )
    elif alg == "LightGBM":
        cfg.update(
            n_estimators=trial.suggest_int("lgb_n_estimators", 100, 400, step=50),
            learning_rate=trial.suggest_float("lgb_learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("lgb_num_leaves", 2, 16),
            min_child_samples=trial.suggest_int("lgb_min_child_samples", 2, 8),
            subsample=trial.suggest_float("lgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("lgb_colsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("lgb_reg_lambda", 1e-3, 10.0, log=True),
        )
    elif alg == "XGBoost":
        cfg.update(
            n_estimators=trial.suggest_int("xgb_n_estimators", 100, 400, step=50),
            learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("xgb_max_depth", 2, 6),
            min_child_weight=trial.suggest_float("xgb_min_child_weight", 0.5, 5.0),
            subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True),
        )
    elif alg == "CatBoost":
        cfg.update(
            iterations=trial.suggest_int("cat_iterations", 100, 400, step=100),
            learning_rate=trial.suggest_float("cat_learning_rate", 0.02, 0.3, log=True),
            depth=trial.suggest_int("cat_depth", 2, 6),
            l2_leaf_reg=trial.suggest_float("cat_l2", 1.0, 20.0, log=True),
        )
    elif alg == "LinearSVC":
        cfg.update(
            C=trial.suggest_float("svc_C", 1e-3, 1e2, log=True),
            class_weight=trial.suggest_categorical("svc_class_weight", [None, "balanced"]),
        )
    elif alg == "LogisticRegression":
        cfg.update(
            C=trial.suggest_float("lr_C", 1e-3, 1e2, log=True),
            penalty=trial.suggest_categorical("lr_penalty", ["l1", "l2"]),
            class_weight=trial.suggest_categorical("lr_class_weight", [None, "balanced"]),
        )
    return cfg


def default_config(algorithm="CatBoost", k=20, standardize=False):
    """Fixed configuration used for the permutation test in `fixed` mode."""
    cfg = {"algorithm": algorithm, "k": k, "standardize": standardize}
    cfg.update({
        "RandomForest": dict(n_estimators=300, max_depth=None, min_samples_leaf=1,
                             max_features="sqrt", class_weight="balanced"),
        "LightGBM": dict(n_estimators=200, learning_rate=0.05, num_leaves=8,
                         min_child_samples=3, subsample=0.9, colsample_bytree=0.8,
                         reg_lambda=1.0),
        "XGBoost": dict(n_estimators=200, learning_rate=0.05, max_depth=3,
                        min_child_weight=1.0, subsample=0.9, colsample_bytree=0.8,
                        reg_lambda=1.0),
        "CatBoost": dict(iterations=200, learning_rate=0.05, depth=4, l2_leaf_reg=3.0),
        "LinearSVC": dict(C=1.0, class_weight="balanced"),
        "LogisticRegression": dict(C=1.0, penalty="l2", class_weight="balanced"),
    }[algorithm])
    return cfg


def build_estimator(cfg, seed=0, n_jobs=1):
    alg = cfg["algorithm"]
    p = {k: v for k, v in cfg.items() if k not in ("algorithm", "k", "standardize")}
    if alg == "RandomForest":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **p)
    if alg == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1, **p)
    if alg == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, n_jobs=n_jobs, tree_method="hist",
                             verbosity=0, **p)
    if alg == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=seed, verbose=0, thread_count=n_jobs,
                                  allow_writing_files=False, **p)
    if alg == "LinearSVC":
        return LinearSVC(random_state=seed, max_iter=50000, dual="auto", **p)
    if alg == "LogisticRegression":
        return LogisticRegression(random_state=seed, max_iter=20000, solver="saga", **p)
    raise ValueError(alg)


def build_pipeline(cfg, seed=0, n_jobs=1, min_detect_frac=0.10):
    """
    Full modelling pipeline. Every step is fitted on training data only:
    the chapter preprocessing (filter, imputation, log, max-scale), the
    univariate feature filter, optional standardisation, then the classifier.
    """
    steps = [
        ("prep", ChapterPreprocessor(min_detect_frac=min_detect_frac)),
        ("select", SelectKBest(f_classif, k=cfg["k"])),
    ]
    if cfg.get("standardize"):
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", build_estimator(cfg, seed, n_jobs)))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
def get_outer_splits(scheme, y, meta):
    """Return (splits, groups, fold_labels). Grouped schemes are deterministic."""
    n = len(y)
    if scheme == "stratified5":
        raise ValueError("stratified5 requires a seed; use get_outer_splits_seeded")
    key = {"lodo_date": "event_date", "loeo_rainday": "rain_day",
           "loeo_episode": "episode"}[scheme]
    groups = meta[key].values
    splits = list(LeaveOneGroupOut().split(np.zeros(n), y, groups))
    return splits, groups, [str(groups[te][0]) for _, te in splits]


def get_outer_splits_seeded(scheme, y, meta, seed):
    if scheme == "stratified5":
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        splits = list(cv.split(np.zeros(len(y)), y))
        return splits, meta["event_date"].values, [f"Fold {i+1}" for i in range(5)]
    return get_outer_splits(scheme, y, meta)


def get_inner_cv(scheme, y_tr, g_tr, seed, n_splits=3):
    if scheme == "stratified5":
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed)
                    .split(np.zeros(len(y_tr)), y_tr))
    ns = min(n_splits, len(np.unique(g_tr)))
    if ns < 2:
        return list(StratifiedKFold(2, shuffle=True, random_state=seed)
                    .split(np.zeros(len(y_tr)), y_tr))
    return list(StratifiedGroupKFold(ns, shuffle=True, random_state=seed)
                .split(np.zeros(len(y_tr)), y_tr, g_tr))


# --------------------------------------------------------------------------- #
def nested_run(X_raw, y, meta, scheme="lodo_date", n_trials=40, seed=0,
               inner_splits=3, n_jobs=1, algorithms=ALGORITHMS,
               standardize=False, min_detect_frac=0.10, verbose=False):
    """
    One complete nested run: algorithm, feature count and hyperparameters are
    all selected inside each training partition.
    """
    codec = LabelCodec(y)
    yc = codec.encode(y)
    splits, groups, fold_labels = get_outer_splits_seeded(scheme, y, meta, seed)

    oof = np.empty(len(y), dtype=int)
    oof_base = np.empty(len(y), dtype=int)
    rows, configs = [], []

    for fi, (tr, te) in enumerate(splits):
        y_tr = yc[tr]
        g_tr = groups[tr]
        inner = get_inner_cv(scheme, y_tr, g_tr, seed, inner_splits)
        Xtr_raw = X_raw[tr]

        def objective(trial):
            cfg = suggest_config(trial, len(tr), algorithms, standardize)
            sc = []
            for itr, ite in inner:
                if len(np.unique(y_tr[itr])) < 2:
                    continue
                pipe = build_pipeline(cfg, seed, n_jobs, min_detect_frac)
                pipe.fit(Xtr_raw[itr], y_tr[itr])
                sc.append(balanced_accuracy_score(y_tr[ite], _predict(pipe, Xtr_raw[ite])))
            return float(np.mean(sc)) if sc else 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed * 977 + fi))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = suggest_config(optuna.trial.FixedTrial(study.best_params),
                              len(tr), algorithms, standardize)

        pipe = build_pipeline(best, seed, n_jobs, min_detect_frac)
        pipe.fit(X_raw[tr], y_tr)
        oof[te] = _predict(pipe, X_raw[te])
        oof_base[te] = np.asarray(
            DummyClassifier(strategy="most_frequent").fit(X_raw[tr], y_tr)
            .predict(X_raw[te])).ravel().astype(int)

        rows.append({
            "seed": seed, "scheme": scheme, "fold": fold_labels[fi],
            "algorithm": best["algorithm"], "k": best["k"],
            "n_train": len(tr), "n_test": len(te),
            "inner_score": study.best_value,
            "fold_balanced_accuracy": balanced_accuracy_score(
                yc[te], oof[te]),
        })
        configs.append(dict(best))
        if verbose:
            print(f"    fold {fold_labels[fi]:12s} -> {best['algorithm']:18s} "
                  f"k={best['k']:2d}  inner={study.best_value:.3f}", flush=True)

    return {
        "seed": seed, "scheme": scheme,
        "oof_pred": codec.decode(oof),
        "oof_baseline": codec.decode(oof_base),
        "pooled_balanced_accuracy": balanced_accuracy_score(y, codec.decode(oof)),
        "fold_table": pd.DataFrame(rows),
        "configs": configs,
        "codec": codec,
    }


# --------------------------------------------------------------------------- #
def permutation_test(X_raw, y, meta, scheme="lodo_date", n_perm=300, seed=0,
                     mode="fixed", cfg=None, n_trials=8, inner_splits=3,
                     n_jobs=1, min_detect_frac=0.10, verbose=False):
    """
    Restricted permutation test: site labels are shuffled only among samples
    collected on the same date, so the rain-event structure is preserved and the
    test asks specifically whether site is predictable *given* the event.

    mode="fixed"  : a single fixed configuration is used for both the observed
                    statistic and every permutation. Fast, and observed and null
                    are produced by an identical procedure — but the null does
                    not include the cost of selecting an algorithm, so if the
                    configuration was chosen after seeing the real result the
                    p-value is mildly anti-conservative.
    mode="nested" : each permutation re-runs the full nested selection. This is
                    the honest version, because the null then reflects the same
                    selection procedure applied to the observed data. It costs
                    roughly n_perm x n_trials x inner_splits x n_folds fits, so
                    use a small n_trials and expect hours rather than minutes.
    """
    rng = np.random.default_rng(seed)
    codec = LabelCodec(y)
    yc = codec.encode(y)
    splits, _, _ = get_outer_splits_seeded(scheme, y, meta, seed)
    dates = meta["event_date"].values
    blocks = [np.where(dates == d)[0] for d in np.unique(dates)]
    cfg = cfg or default_config()

    def run_fixed(yy):
        pred = np.empty(len(yy), dtype=int)
        for tr, te in splits:
            pipe = build_pipeline(cfg, seed, n_jobs, min_detect_frac)
            pipe.fit(X_raw[tr], yy[tr])
            pred[te] = _predict(pipe, X_raw[te])
        return balanced_accuracy_score(yy, pred)

    def run_nested(yy):
        lab = codec.decode(yy)
        r = nested_run(X_raw, lab, meta, scheme, n_trials, seed,
                       inner_splits, n_jobs, min_detect_frac=min_detect_frac)
        return r["pooled_balanced_accuracy"]

    run = run_fixed if mode == "fixed" else run_nested

    observed = run(yc)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = yc.copy()
        for idx in blocks:
            yp[idx] = rng.permutation(yc[idx])
        null[i] = run(yp)
        if verbose and (i + 1) % 25 == 0:
            print(f"    permutation {i+1}/{n_perm}", flush=True)

    return {
        "mode": mode, "observed": float(observed), "null": null,
        "p_value": float((np.sum(null >= observed) + 1) / (n_perm + 1)),
        "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
        "null_q95": float(np.quantile(null, 0.95)), "n_perm": n_perm,
    }
