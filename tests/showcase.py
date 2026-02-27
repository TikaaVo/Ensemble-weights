#!/usr/bin/env python3
"""
Dynamic Ensemble Selection — Showcase
======================================
Benchmarks four DES algorithms against single-model and global-ensemble
baselines on two regression and two classification datasets.

Regression datasets
  California Housing  20K samples, 8 features. Clear coastal/inland price regimes.
  Bike Sharing        17K samples, 8 numeric features. Temporal and weather regimes.

Classification datasets
  Letter Recognition  20K samples, 16 features, 26 classes (A-Z pixel statistics).
  Phoneme             5.4K samples, 5 features, binary (nasal vs oral phoneme).

Models (regression)
  Linear Reg., KNN Regressor, Hist. Gradient Boosting — orthogonal inductive biases.

Models (classification)
  Logistic Regression, KNN Classifier, Hist. Gradient Boosting — same diversity
  principle. All three support predict_proba, enabling the probability metrics.

DES metric choice
  Regression     MAE, mode='min'. Continuous per-sample signal; no probabilities needed.
  Classification log_loss, mode='min', with predict_proba() input. This gives a
                 continuous per-sample signal even for classification. Without it, a
                 0/1 accuracy metric collapses KNORA-U and KNORA-E to near-random
                 behaviour in settings where one model dominates.

Why threshold differs between algorithms and tasks
  knn-dws  threshold=0.5 always. Gates averaged neighbourhood scores after
           normalization; 0.5 means "exclude the bottom half of local range".
  OLA      threshold ignored (argmax needs no gating).
  KNORA    threshold=1.0 for regression (only strictly-best model per neighbour
           earns a vote -- correct oracle analogue for continuous metrics where
           normalization is relative not absolute).
           threshold=0.5 for classification with log_loss (log_loss gives a
           meaningful continuous scale so 0.5 correctly identifies locally
           uncompetitive models without being over-restrictive).

Install:  pip install scikit-learn scipy faiss-cpu
Runtime:  ~5-8 min on a MacBook Air M3
"""

import contextlib
import io
import time
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ensemble_weights.des.knndws   import KNNDWS
from ensemble_weights.des.ola      import OLA
from ensemble_weights.des.knorau   import KNORAU
from ensemble_weights.des.knorae   import KNORAE
from ensemble_weights.des.knoraiu import KNORAIU

_DES_CLASSES = {
    'knn-dws':  KNNDWS,
    'ola':      OLA,
    'knora-u':  KNORAU,
    'knora-e':  KNORAE,
    'knora-iu': KNORAIU,
}

warnings.filterwarnings('ignore')

SEED = 42
W    = 80

# ── Regression settings ───────────────────────────────────────────────
K_REG    = 20
TEMP_REG = 0.1

THRESHOLDS_REG = {
    'knn-dws':  0.5,
    'ola':      0.5,
    'knora-u':  1.0,
    'knora-e':  1.0,
    'knora-iu': 1.0,
}

# ── Classification settings ───────────────────────────────────────────
K_CLF    = 20
TEMP_CLF = 1.0

THRESHOLDS_CLF = {
    'knn-dws':  0.5,
    'ola':      0.5,
    'knora-u':  0.5,
    'knora-e':  0.5,
    'knora-iu': 0.5,
}

DES_METHODS = ['knn-dws', 'ola', 'knora-u', 'knora-e', 'knora-iu']


# ── Display helpers ───────────────────────────────────────────────────

def banner():
    print(f"\n{'━' * W}")
    print("  Dynamic Ensemble Selection -- Showcase  (Regression + Classification)")
    print(f"{'━' * W}")
    print("  Best Single       best val-set model applied everywhere")
    print("  Global Ensemble   fixed weights, Nelder-Mead on val set")
    print("  DES knn-dws       per-sample adaptive blending  <- this library")
    print("  DES OLA           per-sample hard model selection  <- this library")
    print("  DES KNORA-U       per-sample voting (union of competent models)  <- this library")
    print("  DES KNORA-E       per-sample intersection (competent on all neighbours)  <- this library")
    print(f"{'━' * W}")


def dataset_header(name, n_samples, n_features, extra):
    print(f"\n\n{'━' * W}")
    print(f"  Dataset: {name}")
    print(f"  {n_samples:,} samples  .  {n_features} features  .  {extra}")
    print(f"{'━' * W}")


def section(title):
    print(f"\n  {title}")
    print(f"  {'-' * (W - 4)}")


def show_results_reg(rows, best_mae, y_mean):
    best_overall = min(mae for _, mae in rows)
    print(f"\n  {'Method':<44} {'MAE':>8}  {'% of mean':>10}  {'vs Best':>9}")
    print(f"  {'-'*44}  {'-'*8}  {'-'*10}  {'-'*9}")
    for name, mae in rows:
        pct_mean = mae / y_mean * 100
        delta    = (mae - best_mae) / best_mae * 100
        d_str    = "    -    " if mae == best_mae else f"{'+' if delta >= 0 else ''}{delta:.2f}%"
        marker   = "  <" if mae == best_overall else ""
        print(f"  {name:<44}  {mae:>8.4f}  {pct_mean:>9.2f}%  {d_str:>9}{marker}")


def show_results_clf(rows, best_acc):
    best_overall = max(acc for _, acc in rows)
    print(f"\n  {'Method':<44} {'Accuracy':>9}  {'vs Best':>9}")
    print(f"  {'-'*44}  {'-'*9}  {'-'*9}")
    for name, acc in rows:
        delta = (acc - best_acc) / best_acc * 100
        d_str = "    -    " if acc == best_acc else f"{'+' if delta >= 0 else ''}{delta:.2f}%"
        marker = "  <" if acc == best_overall else ""
        print(f"  {name:<44}  {acc*100:>8.2f}%  {d_str:>9}{marker}")


def show_timing(methods, fit_times, predict_times, n_test):
    print(f"\n  DES timing on {n_test:,} test samples:")
    print(f"    {'Method':<12}  {'Fit (ms)':>8}  {'Predict (ms)':>12}  {'ms/sample':>10}")
    print(f"    {'-'*12}  {'-'*8}  {'-'*12}  {'-'*10}")
    for m in methods:
        print(f"    {m:<12}  {fit_times[m]:>8.2f}  {predict_times[m]:>12.2f}"
              f"  {predict_times[m]/n_test:>10.4f}")


def _make_router(task, method, metric, mode, k, preset='balanced'):
    """Instantiate the appropriate DES algorithm class, suppressing the preset print."""
    with contextlib.redirect_stdout(io.StringIO()):
        return _DES_CLASSES[method](
            task=task, metric=metric, mode=mode, k=k, preset=preset,
        )


# ── Regression data and models ────────────────────────────────────────

def load_california():
    print("  Loading California Housing...", end=' ', flush=True)
    X, y = fetch_california_housing(return_X_y=True)
    print("done")
    return X, y, 'California Housing', X.shape[1]


def load_bike():
    print("  Fetching Bike Sharing from OpenML...", end=' ', flush=True)
    d = fetch_openml(data_id=42712, as_frame=True, parser='auto')
    X = d.data.select_dtypes(include=['number']).astype(float)
    X = SimpleImputer(strategy='median').fit_transform(X)
    y = d.target.astype(float).values
    print("done")
    return X, y, 'Bike Sharing (Hourly)', X.shape[1]


def build_regressors(seed=SEED):
    return {
        'Linear Reg.': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc',  StandardScaler()),
            ('m',   LinearRegression()),
        ]),
        'KNN Regressor': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc',  StandardScaler()),
            ('m',   KNeighborsRegressor(n_neighbors=10, n_jobs=-1)),
        ]),
        'Hist. Boosting': HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=seed,
        ),
    }


# ── Classification data and models ────────────────────────────────────

def load_letter():
    print("  Fetching Letter Recognition from OpenML...", end=' ', flush=True)
    d = fetch_openml(data_id=6, as_frame=True, parser='auto')
    X = d.data.astype(float).values
    y = LabelEncoder().fit_transform(d.target)
    print("done")
    return X, y, 'Letter Recognition', X.shape[1], len(np.unique(y))


def load_phoneme():
    print("  Fetching Phoneme from OpenML...", end=' ', flush=True)
    d = fetch_openml(data_id=1489, as_frame=True, parser='auto')
    X = d.data.astype(float).values
    y = LabelEncoder().fit_transform(d.target)
    print("done")
    return X, y, 'Phoneme', X.shape[1], len(np.unique(y))


def build_classifiers(seed=SEED):
    return {
        'Logistic Reg.': Pipeline([
            ('sc', StandardScaler()),
            ('m',  LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        'KNN Classifier': Pipeline([
            ('sc', StandardScaler()),
            ('m',  KNeighborsClassifier(n_neighbors=10, n_jobs=-1)),
        ]),
        'Hist. Boosting': HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=seed,
        ),
    }


# ── Regression ensemble helpers ───────────────────────────────────────

def fit_global_ensemble_reg(val_preds, y_val):
    names   = list(val_preds.keys())
    stacked = np.stack([val_preds[n] for n in names])

    def objective(w):
        w = np.abs(w); w /= w.sum()
        return mean_absolute_error(y_val, w @ stacked)

    res = minimize(objective, np.ones(len(names)) / len(names), method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    w = np.abs(res.x); w /= w.sum()
    return dict(zip(names, w))


def apply_global_weights_reg(preds, weights):
    names = list(weights.keys())
    return np.array([weights[n] for n in names]) @ np.stack([preds[n] for n in names])


def des_predict_reg(router, X_test, test_preds, temperature, threshold):
    names  = list(test_preds.keys())
    result = router.predict(X_test, temperature=temperature, threshold=threshold)
    if isinstance(result, dict):
        result = [result]
    return np.array([
        sum(w[n] * test_preds[n][i] for n in names)
        for i, w in enumerate(result)
    ])


# ── Classification ensemble helpers ──────────────────────────────────

def fit_global_ensemble_clf(val_probas, y_val):
    names   = list(val_probas.keys())
    stacked = np.stack([val_probas[n] for n in names])
    y_int   = y_val.astype(int)

    def objective(w):
        w = np.abs(w); w /= w.sum()
        blended = np.einsum('m,mnc->nc', w, stacked)
        blended = np.clip(blended, 1e-15, 1.0)
        return -np.mean(np.log(blended[np.arange(len(y_int)), y_int]))

    res = minimize(objective, np.ones(len(names)) / len(names), method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    w = np.abs(res.x); w /= w.sum()
    return dict(zip(names, w))


def apply_global_weights_clf(probas, weights):
    names = list(weights.keys())
    w     = np.array([weights[n] for n in names])
    return np.einsum('m,mnc->nc', w, np.stack([probas[n] for n in names]))


def des_predict_clf(router, X_test, test_probas, temperature, threshold):
    names  = list(test_probas.keys())
    result = router.predict(X_test, temperature=temperature, threshold=threshold)
    if isinstance(result, dict):
        result = [result]
    return np.array([
        sum(w[n] * test_probas[n][i] for n in names)
        for i, w in enumerate(result)
    ])


# ── Regression benchmark ──────────────────────────────────────────────

def _method_label_reg(method):
    return {
        'knn-dws':  f'knn-dws  (gate={THRESHOLDS_REG["knn-dws"]}, T={TEMP_REG})',
        'ola':       'OLA',
        'knora-u':  f'KNORA-U   (threshold={THRESHOLDS_REG["knora-u"]})',
        'knora-e':  f'KNORA-E   (threshold={THRESHOLDS_REG["knora-e"]})',
        'knora-iu': f'KNORA-IU  (threshold={THRESHOLDS_REG["knora-iu"]})',
    }[method]


def run_regression(loader, seed=SEED, verbose=True):
    """
    Run a full regression benchmark on one dataset.

    Parameters
    ----------
    loader : callable  -- returns (X, y, name, n_features)
    seed   : int       -- split seed and stochastic model seed
    verbose: bool      -- if False, all printing is suppressed

    Returns
    -------
    dict[str, float]   -- method label -> test MAE (stable keys across seeds)
    """
    _print = print if verbose else lambda *a, **kw: None

    X, y, ds_name, n_features = loader()
    if verbose:
        dataset_header(ds_name, len(X), n_features,
                       f"target mean = {y.mean():.2f}  std = {y.std():.2f}")

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    X_tr, X_val, y_tr, y_val   = train_test_split(X_tv, y_tv, test_size=0.25, random_state=seed)
    _print(f"\n  Split -> {len(X_tr):,} train / {len(X_val):,} val / {len(X_test):,} test")

    if verbose:
        section("Training models  (val-set MAE)")
    models = build_regressors(seed=seed)
    val_preds, test_preds, val_maes = {}, {}, {}
    for mname, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        val_preds[mname]  = model.predict(X_val)
        test_preds[mname] = model.predict(X_test)
        val_maes[mname]   = mean_absolute_error(y_val, val_preds[mname])
        _print(f"    v {mname:<22}  MAE = {val_maes[mname]:.4f}   ({time.time()-t0:.1f}s)")

    best_name = min(val_maes, key=val_maes.get)

    if verbose:
        section("Fitting ensembles  (val set only)")
    t0   = time.time()
    ge_w = fit_global_ensemble_reg(val_preds, y_val)
    _print(f"    v Global Ensemble  (Nelder-Mead, {time.time()-t0:.1f}s)")
    for n, w in ge_w.items():
        _print(f"        {n:<22}  weight = {w:.3f}")

    des_scaler = StandardScaler().fit(X_val)
    X_val_s    = des_scaler.transform(X_val)
    X_test_s   = des_scaler.transform(X_test)

    fit_times, predict_times, des_preds = {}, {}, {}
    for method in DES_METHODS:
        threshold = THRESHOLDS_REG[method]
        label     = _method_label_reg(method)
        router    = _make_router('regression', method, 'mae', 'min', K_REG)

        t0 = time.perf_counter()
        router.fit(X_val_s, y_val, val_preds)
        fit_times[method] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        des_preds[method] = des_predict_reg(
            router, X_test_s, test_preds, TEMP_REG, threshold)
        predict_times[method] = (time.perf_counter() - t0) * 1000

        _print(f"    v DES {label:<30}  fit: {fit_times[method]:6.2f}ms"
               f"  |  predict: {predict_times[method]:6.2f}ms")

    best_mae = mean_absolute_error(y_test, test_preds[best_name])
    ge_mae   = mean_absolute_error(y_test, apply_global_weights_reg(test_preds, ge_w))
    rows = [
        (f"Best Single  ({best_name})",      best_mae),
        ("Global Ensemble  (Nelder-Mead)",   ge_mae),
        *[(f"DES {_method_label_reg(m)}",
           mean_absolute_error(y_test, des_preds[m])) for m in DES_METHODS],
    ]
    if verbose:
        section("Results on held-out test set  (MAE - lower is better)")
        show_results_reg(rows, best_mae, float(y_test.mean()))
        show_timing(DES_METHODS, fit_times, predict_times, len(X_test_s))

    return dict(rows)


# ── Classification benchmark ──────────────────────────────────────────

def _method_label_clf(method):
    return {
        'knn-dws':  f'knn-dws  (gate={THRESHOLDS_CLF["knn-dws"]}, T={TEMP_CLF})',
        'ola':       'OLA',
        'knora-u':  f'KNORA-U   (threshold={THRESHOLDS_CLF["knora-u"]})',
        'knora-e':  f'KNORA-E   (threshold={THRESHOLDS_CLF["knora-e"]})',
        'knora-iu': f'KNORA-IU  (threshold={THRESHOLDS_CLF["knora-iu"]})',
    }[method]


def run_classification(loader, k=K_CLF, seed=SEED, verbose=True):
    """
    Run a full classification benchmark on one dataset.

    Parameters
    ----------
    loader : callable  -- returns (X, y, name, n_features, n_classes)
    k      : int       -- number of DES neighbours
    seed   : int       -- split seed and stochastic model seed
    verbose: bool      -- if False, all printing is suppressed

    Returns
    -------
    dict[str, float]   -- method label -> test accuracy (stable keys across seeds)
    """
    _print = print if verbose else lambda *a, **kw: None

    X, y, ds_name, n_features, n_classes = loader()
    if verbose:
        dataset_header(ds_name, len(X), n_features, f"{n_classes} classes")

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=seed, stratify=y_tv)
    _print(f"\n  Split -> {len(X_tr):,} train / {len(X_val):,} val / {len(X_test):,} test"
           f"  (stratified)")

    if verbose:
        section("Training models  (val-set accuracy)")
    models = build_classifiers(seed=seed)
    val_probas, test_probas, val_accs = {}, {}, {}
    for mname, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        val_probas[mname]  = model.predict_proba(X_val)
        test_probas[mname] = model.predict_proba(X_test)
        val_accs[mname]    = accuracy_score(y_val, model.predict(X_val))
        _print(f"    v {mname:<22}  Acc = {val_accs[mname]*100:.2f}%   ({time.time()-t0:.1f}s)")

    best_name = max(val_accs, key=val_accs.get)

    if verbose:
        section("Fitting ensembles  (val set only - metric: log_loss on probabilities)")
    t0   = time.time()
    ge_w = fit_global_ensemble_clf(val_probas, y_val)
    _print(f"    v Global Ensemble  (Nelder-Mead, {time.time()-t0:.1f}s)")
    for n, w in ge_w.items():
        _print(f"        {n:<22}  weight = {w:.3f}")

    des_scaler = StandardScaler().fit(X_val)
    X_val_s    = des_scaler.transform(X_val)
    X_test_s   = des_scaler.transform(X_test)

    fit_times, predict_times, des_probas = {}, {}, {}
    for method in DES_METHODS:
        threshold = THRESHOLDS_CLF[method]
        label     = _method_label_clf(method)
        router    = _make_router('classification', method, 'log_loss', 'min', k)

        t0 = time.perf_counter()
        router.fit(X_val_s, y_val, val_probas)
        fit_times[method] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        des_probas[method] = des_predict_clf(
            router, X_test_s, test_probas, TEMP_CLF, threshold)
        predict_times[method] = (time.perf_counter() - t0) * 1000

        _print(f"    v DES {label:<30}  fit: {fit_times[method]:6.2f}ms"
               f"  |  predict: {predict_times[method]:6.2f}ms")

    best_single_acc = accuracy_score(y_test, models[best_name].predict(X_test))
    ge_acc          = accuracy_score(
        y_test, apply_global_weights_clf(test_probas, ge_w).argmax(axis=1))
    rows = [
        (f"Best Single  ({best_name})",      best_single_acc),
        ("Global Ensemble  (Nelder-Mead)",   ge_acc),
        *[(f"DES {_method_label_clf(m)}",
           accuracy_score(y_test, des_probas[m].argmax(axis=1))) for m in DES_METHODS],
    ]
    if verbose:
        section("Results on held-out test set  (Accuracy - higher is better)")
        show_results_clf(rows, best_single_acc)
        show_timing(DES_METHODS, fit_times, predict_times, len(X_test_s))

    return dict(rows)


if __name__ == '__main__':
    banner()

    print(f"\n{'━' * W}")
    print("  Regression")
    print(f"{'━' * W}")
    run_regression(load_california)
    run_regression(load_bike)

    print(f"\n\n{'━' * W}")
    print("  Classification  (DES metric: log_loss on predict_proba output)")
    print(f"{'━' * W}")
    run_classification(load_letter, k=20)
    run_classification(load_phoneme, k=10)

    print(f"\n\n{'━' * W}\n  Done.\n{'━' * W}\n")