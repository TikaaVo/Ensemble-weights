#!/usr/bin/env python3
"""
deskit Showcase
==============
Benchmarks deskit against DESlib and DESReg across regression and
classification tasks using a deliberately diverse pool of weaker,
more opinionated models.

Why deskit is framework-agnostic
  deskit receives any numeric predictions — numpy arrays, PyTorch tensors, JAX
  arrays, Keras model outputs — and returns routing weights. It never calls
  fit() or predict() on your models; it only sees their outputs.
  DESlib requires sklearn-compatible estimators (fit/predict/predict_proba API)
  and has no regression support.

Pool design  (both tasks)
  The key difference from the original benchmark is the pool. The original used
  strong globally-competitive models (RF, HGB, Extra Trees, etc.) that stay
  within a few percent of each other everywhere — leaving little for DES routing
  to gain. This showcase uses deliberately diverse, more opinionated models where
  each has a clearly different inductive bias and a clear failure mode. DES has
  genuine signal to exploit.

  Regression pool
    KNN (k=5)         — purely local; wins in dense low-noise regions,
                         fails on sparse or high-dimensional spaces
    Decision Tree     — axis-aligned splits; wins in piecewise-constant regions,
                         fails near smooth or diagonal relationships
    SVR (rbf)         — kernel margins; wins near smooth continuous boundaries,
                         fails on large well-separated clusters
    Ridge             — linear; wins on globally linear relationships,
                         fails on any nonlinear structure
    Bayesian Ridge    — probabilistic linear with automatic regularisation;
                         wins on well-behaved Gaussian data

  Classification pool
    KNN (k=5)         — purely local; wins in dense low-noise regions
    Decision Tree     — axis-aligned rules; wins in piecewise-constant regions
    Gaussian NB       — feature independence assumption; wins when features are
                         roughly Gaussian and separable
    SVM-RBF           — kernel margins; wins near tight decision boundaries
    Logistic Reg      — linear; wins on linearly separable regions

Regression datasets
  California Housing  sklearn built-in,  20,640 samples,  8 features
  Bike Sharing        OpenML 42712,      17,379 samples, 12 features
  Abalone             OpenML 183,         4,177 samples,  9 features
  Diabetes            sklearn built-in,     442 samples, 10 features
  Concrete Strength   OpenML 4353,        1,030 samples,  8 features

Classification datasets   (deskit vs DESlib, direct head-to-head)
  HAR                 OpenML 1478,       10,299 samples, 561 features, 6 classes
  Yeast               OpenML 181,         1,484 samples,   8 features, 10 classes
  Image Segment       OpenML 36,          2,310 samples,  19 features,  7 classes
  Vowel               OpenML 307,           990 samples,  10 features, 11 classes
  Waveform            OpenML 60,          5,000 samples,  40 features,  3 classes

Install:  pip install scikit-learn scipy faiss-cpu deslib DESReg
Runtime:  ~30-50 min on a modern laptop
"""

import contextlib
import io
import time
import warnings

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from deskit.des.knndws  import KNNDWS
from deskit.des.knndwsi import KNNDWSI
from deskit.des.ola     import OLA
from deskit.des.knorau  import KNORAU
from deskit.des.knorae  import KNORAE
from deskit.des.knoraiu import KNORAIU

warnings.filterwarnings('ignore')

# ── Optional DESlib import ─────────────────────────────────────────────────────
try:
    # sklearn >= 1.6 removed BaseEstimator._validate_data;
    # patch it back so DESlib (built against older sklearn) doesn't crash.
    from sklearn.base import BaseEstimator
    if not hasattr(BaseEstimator, '_validate_data'):
        from sklearn.utils.validation import check_array, check_X_y
        def _validate_data_compat(self, X, y=None, reset=True, **kw):
            _safe = {k: v for k, v in kw.items()
                     if k in ('accept_sparse', 'dtype', 'force_all_finite', 'order')}
            if y is not None:
                return check_X_y(X, y, **_safe)
            return check_array(X, **_safe)
        BaseEstimator._validate_data = _validate_data_compat

    from deslib.des.knora_u  import KNORAU  as DL_KNORAU
    from deslib.des.knora_e  import KNORAE  as DL_KNORAE
    from deslib.dcs.ola      import OLA     as DL_OLA
    from deslib.des.meta_des import METADES as DL_METADES
    from deslib.des.knop     import KNOP    as DL_KNOP
    from deslib.des.des_p    import DESP    as DL_DESP
    from deslib.des.des_knn  import DESKNN  as DL_DESKNN
    DESLIB_AVAILABLE = True
except ImportError:
    DESLIB_AVAILABLE = False

# ── Optional DESReg import ─────────────────────────────────────────────────────
# DESReg: Dynamic Ensemble Selection for Regression tasks
# https://github.com/lperezgodoy/DESReg  /  pip install DESReg
#
# Architecture note: DESReg bags the pool internally from unfitted regressors
# and manages the DSEL split itself. This is fundamentally different from deskit,
# which accepts pre-fitted model outputs and never touches or calls your models.
# DESReg is sklearn-only; deskit is framework-agnostic.
try:
    from desReg.des.DESRegression import DESRegression as _DESReg
    DESREG_AVAILABLE = True
except ImportError:
    DESREG_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────────────────

SEED = 42
W    = 84      # print width

K_REG    = 20
K_CLF    = 20
TEMP_REG = 0.1
TEMP_CLF = 1.0

THRESHOLDS_REG = {
    'knn-dws':   0.5,
    'knn-dws-i': 0.5,
    'ola':       0.5,
    'knora-u':   1.0,
    'knora-e':   1.0,
    'knora-iu':  1.0,
}
THRESHOLDS_CLF = {
    'knn-dws':   0.5,
    'knn-dws-i': 0.5,
    'ola':       0.5,
    'knora-u':   0.5,
    'knora-e':   0.5,
    'knora-iu':  0.5,
}

DES_METHODS = ['knn-dws', 'knn-dws-i', 'ola', 'knora-u', 'knora-e', 'knora-iu']

_DES_CLASSES = {
    'knn-dws':   KNNDWS,
    'knn-dws-i': KNNDWSI,
    'ola':       OLA,
    'knora-u':   KNORAU,
    'knora-e':   KNORAE,
    'knora-iu':  KNORAIU,
}

# DESlib algorithms run in the comparison (7 total: 3 basic + 4 advanced)
DL_METHODS = ['KNORA-U', 'KNORA-E', 'OLA', 'META-DES', 'KNOP', 'DESP', 'DESKNN']

# DESReg modes: DES = dynamic ensemble selection, DSR = dynamic regressor selection
DR_METHODS = ['DES', 'DSR']


# ── Display helpers ────────────────────────────────────────────────────────────

def banner():
    dl_status = ('installed — 7 algorithms: KNORA-U/E · OLA · META-DES · KNOP · DESP · DESKNN'
                 if DESLIB_AVAILABLE
                 else 'not installed (pip install deslib) — skipping comparison')
    dr_status = ('installed — head-to-head comparison on regression'
                 if DESREG_AVAILABLE
                 else 'not installed (pip install DESReg) — skipping comparison')
    print(f"\n{'━' * W}")
    print("  deskit Showcase  —  Diverse Pool Benchmark")
    print(f"{'━' * W}")
    print("  Regression pool:      KNN · Decision Tree · SVR · Ridge · Bayesian Ridge")
    print("  Classification pool:  KNN · Decision Tree · Gaussian NB · SVM-RBF · Logistic Reg")
    print()
    print("  Best Single       best val-set model applied to test set everywhere")
    print("  Simple Average    uniform equal-weight blend of all models (no tuning)")
    print("  deskit             KNN-DWS · KNN-DWS-I · OLA · KNORA-U · KNORA-E · KNORA-IU")
    print(f"  DESlib            {dl_status}")
    print(f"  DESReg            {dr_status}")
    print()
    print("  deskit is framework-agnostic: pass predictions from PyTorch, JAX,")
    print("  Keras, or any library. DESlib and DESReg require sklearn estimators.")
    print("  DESlib has no regression support. DESReg has no classification support.")
    print(f"{'━' * W}")


def dataset_header(name, n_samples, n_features, extra):
    print(f"\n\n{'━' * W}")
    print(f"  Dataset: {name}")
    print(f"  {n_samples:,} samples  ·  {n_features} features  ·  {extra}")
    print(f"{'━' * W}")


def section(title):
    print(f"\n  {title}")
    print(f"  {'-' * (W - 4)}")


def show_results_reg(rows, best_mae, y_mean):
    valid        = [(n, m) for n, m in rows if m is not None]
    best_overall = min(m for _, m in valid) if valid else best_mae
    print(f"\n  {'Method':<48} {'MAE':>8}  {'% of mean':>10}  {'vs Best':>9}")
    print(f"  {'-'*48}  {'-'*8}  {'-'*10}  {'-'*9}")
    prev_section = None
    for name, mae in rows:
        cur_section = 'desreg' if name.startswith('DESReg') else 'other'
        if cur_section != prev_section and prev_section is not None:
            print(f"  {'·'*48}  {'·'*8}  {'·'*10}  {'·'*9}")
        prev_section = cur_section
        if mae is None:
            print(f"  {name:<48}  {'N/A':>8}  {'':>10}  {'':>9}")
            continue
        pct_mean = mae / y_mean * 100
        delta    = (mae - best_mae) / best_mae * 100
        d_str    = "    -    " if mae == best_mae else f"{'+' if delta >= 0 else ''}{delta:.2f}%"
        marker   = "  <" if mae == best_overall else ""
        print(f"  {name:<48}  {mae:>8.4f}  {pct_mean:>9.2f}%  {d_str:>9}{marker}")


def show_results_clf(rows, best_acc):
    valid = [(n, a) for n, a in rows if a is not None]
    best_overall = max(a for _, a in valid) if valid else best_acc
    print(f"\n  {'Method':<48} {'Accuracy':>9}  {'vs Best':>9}")
    print(f"  {'-'*48}  {'-'*9}  {'-'*9}")
    prev_section = None
    for name, acc in rows:
        cur_section = 'deslib' if name.startswith('DESlib') else 'deskit'
        if cur_section != prev_section and prev_section is not None:
            print(f"  {'·'*48}  {'·'*9}  {'·'*9}")
        prev_section = cur_section
        if acc is None:
            print(f"  {name:<48}  {'N/A':>9}  {'':>9}")
            continue
        delta  = (acc - best_acc) / best_acc * 100
        d_str  = "    -    " if acc == best_acc else f"{'+' if delta >= 0 else ''}{delta:.2f}%"
        marker = "  <" if acc == best_overall else ""
        print(f"  {name:<48}  {acc*100:>8.2f}%  {d_str:>9}{marker}")


def show_timing(methods, fit_times, predict_times, n_test):
    print(f"\n  deskit timing on {n_test:,} test samples:")
    print(f"    {'Method':<14}  {'Fit (ms)':>8}  {'Predict (ms)':>12}  {'ms/sample':>10}")
    print(f"    {'-'*14}  {'-'*8}  {'-'*12}  {'-'*10}")
    for m in methods:
        print(f"    {m:<14}  {fit_times[m]:>8.2f}  {predict_times[m]:>12.2f}"
              f"  {predict_times[m]/n_test:>10.4f}")


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(X_tr, X_val, X_test):
    """
    Impute and StandardScale all splits using training statistics.

    All models and both routing libraries receive identically scaled features,
    avoiding double-scaling artifacts from Pipeline-internal scalers.
    """
    prep     = Pipeline([('imp', SimpleImputer(strategy='median')),
                         ('sc',  StandardScaler())])
    X_tr_s   = prep.fit_transform(X_tr)
    X_val_s  = prep.transform(X_val)
    X_test_s = prep.transform(X_test)
    return X_tr_s, X_val_s, X_test_s


# ── Model builders ─────────────────────────────────────────────────────────────

def build_regressors(seed=SEED):
    """
    5 diverse regressors — analogues of the classification pool.
    Each has a clearly different inductive bias and failure mode, giving DES
    genuine competence regions to exploit.

      KNN (k=5)       — purely local; wins in dense low-noise regions,
                         fails on sparse or high-dimensional spaces
      Decision Tree   — axis-aligned splits; wins in piecewise-constant regions,
                         fails near smooth or diagonal relationships
      SVR (rbf)       — kernel margins; wins near smooth continuous boundaries,
                         fails on large well-separated clusters
      Ridge           — linear; wins on globally linear relationships,
                         fails on any nonlinear structure
      Bayesian Ridge  — probabilistic linear with automatic regularisation;
                         wins on well-behaved Gaussian data
    """
    return {
        'KNN':            KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        'Decision Tree':  DecisionTreeRegressor(max_depth=8, random_state=seed),
        'SVR':            SVR(kernel='rbf', C=1.0),
        'Ridge':          Ridge(alpha=1.0),
        'Bayesian Ridge': BayesianRidge(),
    }


def build_classifiers(seed=SEED):
    """
    5 diverse classifiers — each with a clearly different inductive bias
    and failure mode, giving DES real competence regions to exploit.

      KNN (k=5)      — local geometry; wins in dense low-noise regions
      Decision Tree  — axis-aligned rules; wins in piecewise-constant regions
      Gaussian NB    — feature independence assumption; wins when features are
                        roughly Gaussian and separable
      SVM-RBF        — kernel margins; wins near tight decision boundaries
      Logistic Reg   — linear; wins on linearly separable regions
    """
    return {
        'KNN':           KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=seed),
        'Gaussian NB':   GaussianNB(),
        'SVM-RBF':       SVC(C=1.0, kernel='rbf', probability=True,
                             random_state=seed),
        'Logistic Reg':  LogisticRegression(max_iter=1000, random_state=seed,
                                            n_jobs=-1),
    }


# ── Dataset loaders ────────────────────────────────────────────────────────────
# Regression

def load_california():
    print("  Loading California Housing...", end=' ', flush=True)
    X, y = fetch_california_housing(return_X_y=True)
    print("done")
    return X, y, 'California Housing', X.shape[1]


def load_bike():
    print("  Fetching Bike Sharing from OpenML...", end=' ', flush=True)
    d = fetch_openml(data_id=42712, as_frame=True, parser='auto')
    X = d.data.select_dtypes(include=['number']).astype(float).values
    y = d.target.astype(float).values
    print("done")
    return X, y, 'Bike Sharing (Hourly)', X.shape[1]


def load_abalone():
    """
    UCI Abalone dataset: predict age (number of rings) from physical measurements.
    Sex (M/F/I) is one-hot encoded. SimpleImputer handles any missing values.
    """
    print("  Fetching Abalone from OpenML...", end=' ', flush=True)
    import pandas as pd
    d = fetch_openml(data_id=183, as_frame=True, parser='auto')
    X = pd.get_dummies(d.data, drop_first=True).astype(float).values
    y = d.target.astype(float).values
    print("done")
    return X, y, 'Abalone (Rings)', X.shape[1]


def load_diabetes_data():
    """
    sklearn Diabetes dataset: 442 samples, 10 features (age, sex, BMI,
    blood pressure, six serum measurements), continuous target is disease
    progression one year after baseline. Small enough that competence regions
    are meaningful — KNN wins on similar patient profiles, SVR wins near smooth
    physiological boundaries, Ridge wins where the relationship is additive.
    """
    print("  Loading Diabetes (sklearn built-in)...", end=' ', flush=True)
    X, y = load_diabetes(return_X_y=True)
    print("done")
    return X, y, 'Diabetes (sklearn)', X.shape[1]


def load_concrete():
    """
    Concrete Compressive Strength (OpenML 4353): 1,030 samples, 8 features
    (cement, blast furnace slag, fly ash, water, superplasticiser, coarse
    aggregate, fine aggregate, age), target is compressive strength in MPa.
    Nonlinear ingredient interactions create genuine competence regions —
    Decision Tree wins on dominant ingredient splits, SVR wins near smooth
    interaction boundaries, Ridge fails on the nonlinear age-cure relationship.
    """
    print("  Fetching Concrete Strength from OpenML...", end=' ', flush=True)
    d = fetch_openml(data_id=4353, as_frame=True, parser='auto')
    # If target is None, assume the last column is the target (strength)
    if d.target is None:
        X = d.data.iloc[:, :-1].astype(float).values
        y = d.data.iloc[:, -1].astype(float).values
    else:
        X = d.data.astype(float).values
        y = d.target.astype(float).values
    print("done")
    return X, y, 'Concrete Strength', X.shape[1]


# Classification

def load_har():
    """
    Human Activity Recognition (OpenML 1478): 10,299 samples of smartphone
    sensor data (accelerometer + gyroscope), 561 features, 6 activity classes.
    Static activities (sitting, standing, laying) occupy completely different
    regions of feature space than dynamic ones (walking, running) — each model
    family handles a different activity type.
    """
    print("  Fetching HAR from OpenML...", end=' ', flush=True)
    ds = fetch_openml(data_id=1478, as_frame=True, parser='auto')
    X  = ds.data.astype(float).values
    y  = LabelEncoder().fit_transform(ds.target)
    print("done")
    return X, y, 'Human Activity Recognition (HAR)', X.shape[1], len(np.unique(y))


def load_yeast():
    """
    Yeast protein localisation (OpenML 181): 1,484 samples, 8 features,
    10 classes. Highly multiclass with class imbalance — some classes have
    fewer than 30 samples. DES literature consistently shows gains on small
    imbalanced multiclass problems where the global best model is pulled
    toward majority classes.
    """
    print("  Fetching Yeast from OpenML...", end=' ', flush=True)
    ds = fetch_openml(data_id=181, as_frame=True, parser='auto')
    X  = ds.data.astype(float).values
    y  = LabelEncoder().fit_transform(ds.target)
    print("done")
    return X, y, 'Yeast (Protein Localisation)', X.shape[1], len(np.unique(y))


def load_segment():
    """
    Image Segment (OpenML 36): 2,310 samples of outdoor image segments,
    19 features (colour, texture, shape statistics), 7 classes (sky, grass,
    cement, window, brick, path, foliage). Different segment types produce
    different feature distributions — natural competence regions for each
    model family.
    """
    print("  Fetching Image Segment from OpenML...", end=' ', flush=True)
    ds = fetch_openml(data_id=36, as_frame=True, parser='auto')
    X  = ds.data.astype(float).values
    y  = LabelEncoder().fit_transform(ds.target)
    print("done")
    return X, y, 'Image Segment', X.shape[1], len(np.unique(y))


def load_vowel():
    """
    Vowel Recognition (OpenML 307): 990 samples, 10 features (LPC-derived
    formant frequencies), 11 vowel classes. A canonical DES benchmark —
    phonetically similar vowels overlap heavily in feature space, creating
    genuine uncertainty regions where routing to the right specialist matters.
    Speaker name column is metadata and dropped (non-numeric).
    """
    print("  Fetching Vowel from OpenML...", end=' ', flush=True)
    ds = fetch_openml(data_id=307, as_frame=True, parser='auto')
    X  = ds.data.select_dtypes(include='number').astype(float).values
    y  = LabelEncoder().fit_transform(ds.target)
    print("done")
    return X, y, 'Vowel Recognition', X.shape[1], len(np.unique(y))


def load_waveform():
    """
    Waveform (OpenML 60 / waveform-5000): 5,000 samples, 40 features,
    3 classes (three types of waveform generated by combining 2 of 3 base
    waves). All features are continuous with Gaussian noise added. The
    three classes overlap heavily in feature space — Bayes-optimal error
    is ~14% — making this a classic benchmark for methods that benefit
    from routing to local specialists rather than relying on a single
    global boundary. SVM-RBF wins near tight decision boundaries, KNN
    wins in dense low-noise pockets, and Logistic Reg wins in the
    linearly-separable regions where one wave dominates.
    """
    print("  Fetching Waveform from OpenML...", end=' ', flush=True)
    ds = fetch_openml(data_id=60, as_frame=True, parser='auto')
    X  = ds.data.astype(float).values
    y  = LabelEncoder().fit_transform(ds.target)
    print("done")
    return X, y, 'Waveform (waveform-5000)', X.shape[1], len(np.unique(y))


# ── Router factory ─────────────────────────────────────────────────────────────

def _make_router(task, method, metric, mode, k, preset='balanced'):
    """Instantiate a deskit algorithm class, suppressing the preset print."""
    with contextlib.redirect_stdout(io.StringIO()):
        return _DES_CLASSES[method](task=task, metric=metric, mode=mode, k=k, preset=preset)


# ── Ensemble helpers ───────────────────────────────────────────────────────────

def fit_global_ensemble_reg(val_preds, y_val):
    """Uniform average — equal weight to every model."""
    names = list(val_preds.keys())
    w = np.ones(len(names)) / len(names)
    return dict(zip(names, w))


def apply_global_weights_reg(preds, weights):
    names = list(weights.keys())
    return np.array([weights[n] for n in names]) @ np.stack([preds[n] for n in names])


def fit_global_ensemble_clf(val_probas, y_val):
    """Uniform average — equal weight to every model."""
    names = list(val_probas.keys())
    w = np.ones(len(names)) / len(names)
    return dict(zip(names, w))


def apply_global_weights_clf(probas, weights):
    names = list(weights.keys())
    w     = np.array([weights[n] for n in names])
    return np.einsum('m,mnc->nc', w, np.stack([probas[n] for n in names]))


def des_predict_reg(router, X_test, test_preds, temperature, threshold):
    names  = list(test_preds.keys())
    result = router.predict(X_test, temperature=temperature, threshold=threshold)
    if isinstance(result, dict):
        result = [result]
    return np.array([
        sum(w[n] * test_preds[n][i] for n in names)
        for i, w in enumerate(result)
    ])


def des_predict_clf(router, X_test, test_probas, temperature, threshold):
    names  = list(test_probas.keys())
    result = router.predict(X_test, temperature=temperature, threshold=threshold)
    if isinstance(result, dict):
        result = [result]
    return np.array([
        sum(w[n] * test_probas[n][i] for n in names)
        for i, w in enumerate(result)
    ])


# ── DESlib comparison ──────────────────────────────────────────────────────────

def run_deslib(fitted_models, X_val_s, y_val, X_test_s, y_test, k=K_CLF):
    """
    Run DESlib's 7 algorithms on the same pool and scaled features.

    Returns
    -------
    results   : dict[str, float | None]  accuracy per algorithm (None if failed)
    fit_ms    : dict[str, float]
    pred_ms   : dict[str, float]
    """
    pool     = list(fitted_models.values())
    results  = {}
    fit_ms   = {}
    pred_ms  = {}

    _dl_registry = [
        ('KNORA-U',  DL_KNORAU),
        ('KNORA-E',  DL_KNORAE),
        ('OLA',      DL_OLA),
        ('META-DES', DL_METADES),
        ('KNOP',     DL_KNOP),
        ('DESP',     DL_DESP),
        ('DESKNN',   DL_DESKNN),
    ]
    for dl_label, cls in _dl_registry:
        try:
            m = cls(pool_classifiers=pool, k=k)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.fit(X_val_s, y_val)
            fit_ms[dl_label] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            y_pred = m.predict(X_test_s)
            pred_ms[dl_label] = (time.perf_counter() - t0) * 1000

            results[dl_label] = accuracy_score(y_test, y_pred)
        except Exception as exc:
            results[dl_label] = None
            fit_ms[dl_label]  = float('nan')
            pred_ms[dl_label] = float('nan')
            print(f"      ✗ DESlib {dl_label} failed: {exc}")

    return results, fit_ms, pred_ms


# ── Label helpers ──────────────────────────────────────────────────────────────

def _label_reg(method):
    return {
        'knn-dws':   f'deskit KNN-DWS    (gate={THRESHOLDS_REG["knn-dws"]}, T={TEMP_REG})',
        'knn-dws-i': f'deskit KNN-DWS-I  (gate={THRESHOLDS_REG["knn-dws-i"]}, T={TEMP_REG})',
        'ola':        'deskit OLA',
        'knora-u':   f'deskit KNORA-U    (th={THRESHOLDS_REG["knora-u"]})',
        'knora-e':   f'deskit KNORA-E    (th={THRESHOLDS_REG["knora-e"]})',
        'knora-iu':  f'deskit KNORA-IU   (th={THRESHOLDS_REG["knora-iu"]})',
    }[method]


def _label_clf(method):
    return {
        'knn-dws':   f'deskit KNN-DWS    (gate={THRESHOLDS_CLF["knn-dws"]}, T={TEMP_CLF})',
        'knn-dws-i': f'deskit KNN-DWS-I  (gate={THRESHOLDS_CLF["knn-dws-i"]}, T={TEMP_CLF})',
        'ola':        'deskit OLA',
        'knora-u':   f'deskit KNORA-U    (th={THRESHOLDS_CLF["knora-u"]})',
        'knora-e':   f'deskit KNORA-E    (th={THRESHOLDS_CLF["knora-e"]})',
        'knora-iu':  f'deskit KNORA-IU   (th={THRESHOLDS_CLF["knora-iu"]})',
    }[method]


# ── DESReg comparison ──────────────────────────────────────────────────────────

def run_desreg(X_tv_s, y_tv, X_test_s, y_test, seed=SEED, k=K_REG, verbose=True):
    """
    Run DESReg on the same data budget as deskit.

    DESReg receives X_tv (train+val combined, pre-scaled) and manages its own
    DSEL split internally via DSEL_perc=0.25 — giving it the same ~20% of total
    data for its competence region that deskit uses as its val set.

    Pool: identical 5-model pool as deskit (no nested ensembles, so bagging is fast).
    n_estimators_bag=2 is the minimum valid value; each instance is a
    BaggingRegressor trained on a bootstrap sample of X_tv.

    Returns
    -------
    results  : dict[str, float | None]  MAE per DESReg mode
    fit_ms   : dict[str, float]
    pred_ms  : dict[str, float]
    """
    _print = print if verbose else lambda *a, **kw: None
    regressors = list(build_regressors(seed=seed).values())

    results = {}
    fit_ms  = {}
    pred_ms = {}

    for mode in DR_METHODS:
        try:
            m = _DESReg(
                regressors_list   = regressors,
                n_estimators_bag  = 2,
                DSEL_perc         = 0.25,
                XTRAIN_full       = True,
                k                 = k,
                ensemble_type     = mode,
            )
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.fit(X_tv_s, y_tv)
            fit_ms[mode] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            y_pred = m.predict(X_test_s)
            pred_ms[mode] = (time.perf_counter() - t0) * 1000

            results[mode] = float(mean_absolute_error(y_test, y_pred))
            _print(f"    v DESReg {mode:<5}  MAE = {results[mode]:.4f}"
                   f"  fit: {fit_ms[mode]:6.2f}ms  |  predict: {pred_ms[mode]:6.2f}ms")
        except Exception as exc:
            results[mode] = None
            fit_ms[mode]  = float('nan')
            pred_ms[mode] = float('nan')
            _print(f"      ✗ DESReg {mode} failed: {exc}")

    return results, fit_ms, pred_ms


# ── Regression benchmark ───────────────────────────────────────────────────────

def run_regression(loader, seed=SEED, verbose=True):
    """
    Full regression benchmark on one dataset.

    Parameters
    ----------
    loader  : callable  → (X, y, name, n_features)
    seed    : int
    verbose : bool

    Returns
    -------
    dict[str, float]   method label → test MAE
    dict               labelled fit times
    dict               labelled predict times
    """
    _print = print if verbose else lambda *a, **kw: None

    X, y, ds_name, n_features = loader()
    if verbose:
        dataset_header(ds_name, len(X), n_features,
                       f"target mean = {y.mean():.2f}  std = {y.std():.2f}")

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    X_tr, X_val,  y_tr, y_val  = train_test_split(X_tv, y_tv, test_size=0.25, random_state=seed)
    X_tr_s, X_val_s, X_test_s  = preprocess(X_tr, X_val, X_test)
    # X_tv_s: train+val on training scaler — passed to DESReg for a fair data budget
    _prep_tv = Pipeline([('imp', SimpleImputer(strategy='median')),
                         ('sc',  StandardScaler())])
    X_tv_s   = _prep_tv.fit_transform(X_tv)
    _print(f"\n  Split -> {len(X_tr):,} train / {len(X_val):,} val / {len(X_test):,} test")
    if DESLIB_AVAILABLE and verbose:
        _print("  Note: DESlib does not support regression — only deskit benchmarked here.")

    if verbose:
        section("Training models  (val-set MAE)")
    models = build_regressors(seed=seed)
    val_preds, test_preds, val_maes = {}, {}, {}
    for mname, model in models.items():
        t0 = time.time()
        model.fit(X_tr_s, y_tr)
        val_preds[mname]  = model.predict(X_val_s)
        test_preds[mname] = model.predict(X_test_s)
        val_maes[mname]   = mean_absolute_error(y_val, val_preds[mname])
        _print(f"    v {mname:<20}  MAE = {val_maes[mname]:.4f}   ({time.time()-t0:.1f}s)")

    best_name = min(val_maes, key=val_maes.get)

    if verbose:
        section("Fitting ensembles  (val set only)")
    ge_w = fit_global_ensemble_reg(val_preds, y_val)
    _print(f"    v Simple Average")

    fit_times, predict_times, des_preds = {}, {}, {}
    for method in DES_METHODS:
        th     = THRESHOLDS_REG[method]
        label  = _label_reg(method)
        router = _make_router('regression', method, 'mae', 'min', K_REG)
        t0 = time.perf_counter()
        router.fit(X_val_s, y_val, val_preds)
        fit_times[method] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        des_preds[method] = des_predict_reg(router, X_test_s, test_preds, TEMP_REG, th)
        predict_times[method] = (time.perf_counter() - t0) * 1000
        _print(f"    v {label:<42}  fit: {fit_times[method]:6.2f}ms"
               f"  |  predict: {predict_times[method]:6.2f}ms")

    dr_results, dr_fit_ms, dr_pred_ms = {}, {}, {}
    if DESREG_AVAILABLE:
        if verbose:
            section("DESReg comparison  (same regressor types, same data budget)")
            _print("  DESReg manages its own bagging and DSEL split internally.")
            _print("  Same 5-model pool as deskit -- comparison is routing quality only.")
            _print("  n_estimators_bag=2, DSEL_perc=0.25 (~20% of total data for DSEL).")
        dr_results, dr_fit_ms, dr_pred_ms = run_desreg(
            X_tv_s, y_tv, X_test_s, y_test, seed=seed, k=K_REG, verbose=verbose)

    best_mae = mean_absolute_error(y_test, test_preds[best_name])
    ge_mae   = mean_absolute_error(y_test, apply_global_weights_reg(test_preds, ge_w))
    rows = [
        (f"Best Single  ({best_name})", best_mae),
        ("Simple Average",  ge_mae),
        *[(_label_reg(m), mean_absolute_error(y_test, des_preds[m])) for m in DES_METHODS],
    ]
    if DESREG_AVAILABLE:
        for dr_mode in DR_METHODS:
            rows.append((f"DESReg {dr_mode}", dr_results.get(dr_mode)))
    if verbose:
        section("Results on held-out test set  (MAE — lower is better)")
        show_results_reg(rows, best_mae, float(y_test.mean()))
        show_timing(DES_METHODS, fit_times, predict_times, len(X_test_s))

    labelled_fit  = {_label_reg(m): fit_times[m]     for m in DES_METHODS}
    labelled_pred = {_label_reg(m): predict_times[m] for m in DES_METHODS}
    return dict(rows), labelled_fit, labelled_pred


# ── Classification benchmark ───────────────────────────────────────────────────

def run_classification(loader, k=K_CLF, seed=SEED, verbose=True, note=None):
    """
    Full classification benchmark on one dataset, with optional DESlib comparison.

    Parameters
    ----------
    loader  : callable  → (X, y, name, n_features, n_classes)
    k       : int
    seed    : int
    verbose : bool
    note    : str or None  — extra context printed after the dataset header

    Returns
    -------
    dict[str, float | None]   method label → test accuracy
    dict                      labelled fit times
    dict                      labelled predict times
    """
    _print = print if verbose else lambda *a, **kw: None

    X, y, ds_name, n_features, n_classes = loader()
    if verbose:
        dataset_header(ds_name, len(X), n_features, f"{n_classes} classes")
        if note:
            print(f"\n  ℹ  {note}")

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=seed, stratify=y_tv)
    X_tr_s, X_val_s, X_test_s = preprocess(X_tr, X_val, X_test)
    _print(f"\n  Split -> {len(X_tr):,} train / {len(X_val):,} val / {len(X_test):,} test"
           f"  (stratified)")

    if verbose:
        section("Training models  (val-set accuracy)")
    models = build_classifiers(seed=seed)
    val_probas, val_preds_hard, test_probas, val_accs = {}, {}, {}, {}
    for mname, model in models.items():
        t0 = time.time()
        model.fit(X_tr_s, y_tr)
        val_probas[mname]     = model.predict_proba(X_val_s)
        val_preds_hard[mname] = model.predict(X_val_s)
        test_probas[mname]    = model.predict_proba(X_test_s)
        val_accs[mname]       = accuracy_score(y_val, model.predict(X_val_s))
        _print(f"    v {mname:<20}  Acc = {val_accs[mname]*100:.2f}%   ({time.time()-t0:.1f}s)")

    best_name = max(val_accs, key=val_accs.get)

    if verbose:
        section("Fitting ensembles  (val set — metric: log_loss on probabilities)")
    ge_w = fit_global_ensemble_clf(val_probas, y_val)
    _print(f"    v Simple Average")

    fit_times, predict_times, des_probas = {}, {}, {}
    # KNORA algorithms use accuracy (0/1) on hard predictions — matching the
    # original algorithm definition and DESlib's behaviour. KNN-DWS and OLA
    # use log_loss on probabilities for richer continuous competence signals.
    _KNORA = {'knora-u', 'knora-e', 'knora-iu'}
    for method in DES_METHODS:
        th    = THRESHOLDS_CLF[method]
        label = _label_clf(method)
        if method in _KNORA:
            router    = _make_router('classification', method, 'accuracy', 'max', k)
            fit_input = val_preds_hard
        else:
            router    = _make_router('classification', method, 'log_loss', 'min', k)
            fit_input = val_probas
        t0 = time.perf_counter()
        router.fit(X_val_s, y_val, fit_input)
        fit_times[method] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        des_probas[method] = des_predict_clf(router, X_test_s, test_probas, TEMP_CLF, th)
        predict_times[method] = (time.perf_counter() - t0) * 1000
        _print(f"    v {label:<42}  fit: {fit_times[method]:6.2f}ms"
               f"  |  predict: {predict_times[method]:6.2f}ms")

    # DESlib comparison — always run so benchmark.py gets results even with verbose=False.
    # Printing is gated on verbose separately.
    dl_results, dl_fit_ms, dl_pred_ms = {}, {}, {}
    if DESLIB_AVAILABLE:
        dl_results, dl_fit_ms, dl_pred_ms = run_deslib(
            models, X_val_s, y_val, X_test_s, y_test, k=k)
        if verbose:
            section("DESlib comparison  (same pool classifiers, same scaled features)")
            for dl_label in DL_METHODS:
                acc = dl_results.get(dl_label)
                ft  = dl_fit_ms.get(dl_label, float('nan'))
                pt  = dl_pred_ms.get(dl_label, float('nan'))
                acc_str = f"{acc*100:.2f}%" if acc is not None else "N/A"
                _print(f"    v DESlib {dl_label:<20}  acc: {acc_str:<8}"
                       f"  fit: {ft:6.2f}ms  |  predict: {pt:6.2f}ms")

    best_acc = accuracy_score(y_test, models[best_name].predict(X_test_s))
    ge_acc   = accuracy_score(
        y_test, apply_global_weights_clf(test_probas, ge_w).argmax(axis=1))
    rows = [
        (f"Best Single  ({best_name})", best_acc),
        ("Simple Average",  ge_acc),
        *[(_label_clf(m),
           accuracy_score(y_test, des_probas[m].argmax(axis=1)))
          for m in DES_METHODS],
    ]
    if DESLIB_AVAILABLE:
        for dl_label in DL_METHODS:
            rows.append((f"DESlib {dl_label}", dl_results.get(dl_label)))

    if verbose:
        section("Results on held-out test set  (Accuracy — higher is better)")
        show_results_clf(rows, best_acc)
        show_timing(DES_METHODS, fit_times, predict_times, len(X_test_s))

    labelled_fit  = {_label_clf(m): fit_times[m]     for m in DES_METHODS}
    labelled_pred = {_label_clf(m): predict_times[m] for m in DES_METHODS}
    if DESLIB_AVAILABLE:
        labelled_fit.update({f'DESlib {dl}':  v for dl, v in dl_fit_ms.items()})
        labelled_pred.update({f'DESlib {dl}': v for dl, v in dl_pred_ms.items()})
    return dict(rows), labelled_fit, labelled_pred


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    banner()

    print(f"\n{'━' * W}")
    print("  Regression  (deskit vs DESReg — DESlib has no regression support)")
    print(f"{'━' * W}")
    run_regression(load_california)
    run_regression(load_bike)
    run_regression(load_abalone)
    run_regression(load_diabetes_data)
    run_regression(load_concrete)

    print(f"\n\n{'━' * W}")
    print("  Classification  (deskit vs DESlib — direct head-to-head)")
    print(f"{'━' * W}")
    run_classification(load_har,      k=20)
    run_classification(load_yeast,    k=10)
    run_classification(load_segment,  k=10)
    run_classification(load_vowel,    k=10)
    run_classification(load_waveform, k=10)

    print(f"\n\n{'━' * W}")
    print("  Done.")
    print(f"{'━' * W}")