#!/usr/bin/env python3
"""
Dynamic Ensemble Selection — Showcase (Regression)
====================================================
Three ensembling strategies, head-to-head on real regression datasets.

  Best Single Model    Find the best individual model on validation data;
                       apply it uniformly to every test sample.

  Global Ensemble      Learn a single fixed weight vector over model predictions
                       minimising validation MAE with gradient-free Nelder-Mead.
                       Weights are the same for every sample.

  DES — knn-dw         For each test point, retrieve its K nearest neighbors in
                       the validation set and weight models by their local MAE
                       there. Weights adapt per sample.  ← this library

  DES — OLA            Same neighborhood lookup, but performs hard selection:
                       assign full weight to the single locally-best model.
                       ← this library

Why regression?
  Classification only gives a 0/1 signal per neighbor, which is too coarse
  for DES's softmax weighting to differentiate models within a small window.
  MAE gives a continuous signal — a model that was off by 10k is meaningfully
  worse than one off by 1k — producing sharper, more useful local weights.

Why these models?
  The five models span completely different inductive biases:
    · Linear Regression  — fast, globally linear, fails on non-linear regions
    · KNN Regressor      — purely local averaging, no global structure at all
    · SVR (RBF)          — smooth kernel boundaries, different failure modes
    · Random Forest      — high-variance tree ensemble
    · Hist. Boosting     — sequential boosting, handles NaN natively
  Because they fail on different samples, DES has genuinely different "experts"
  to route between. The previous tree-heavy pool was too correlated for this.

───────────────────────────────────────────────────────────────────────
  Datasets  California Housing (sklearn)  ·  Ames Housing (OpenML)
  Metric    MAE  (Mean Absolute Error, lower is better)
───────────────────────────────────────────────────────────────────────

  Install:  pip install scikit-learn scipy
  Runtime:  ~2-4 min on a MacBook Air M3
"""

import warnings
import time
import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

from ensemble_weights import DynamicRouter

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
SEED = 42
K    = 20   # neighbors for DES; larger = smoother/more robust local estimates

# ─────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────
W = 66

def banner():
    print(f"\n{'━' * W}")
    print("  Dynamic Ensemble Selection — Showcase  (Regression)")
    print(f"{'━' * W}")
    print("  ◆ Best Single       use the top val-set model everywhere")
    print("  ◆ Global Ensemble   fixed weights, Nelder-Mead on val set")
    print("  ◆ DES knn-dw        per-sample adaptive blending  ← this library")
    print("  ◆ DES OLA           per-sample best-model select  ← this library")
    print(f"  Metric: MAE (lower is better)")
    print(f"{'━' * W}")

def dataset_header(name, n_samples, n_features, y_std):
    print(f"\n\n{'━' * W}")
    print(f"  Dataset: {name}")
    print(f"  {n_samples:,} samples  ·  {n_features} features  ·  target std = {y_std:.3f}")
    print(f"{'━' * W}")

def section(title):
    print(f"\n  {title}")
    print(f"  {'─' * (W - 4)}")

def show_results(rows, best_mae):
    """
    rows: list of (method_name, test_mae)
    best_mae: MAE of the best single model (used as baseline for delta column)
    """
    best_overall = min(mae for _, mae in rows)
    print(f"\n  {'Method':<36} {'Test MAE':>10}  {'vs Best':>9}")
    print(f"  {'─' * 36}  {'─' * 10}  {'─' * 9}")
    for name, mae in rows:
        delta  = mae - best_mae
        d_str  = "    —      " if delta == 0 else f" {'+' if delta >= 0 else ''}{delta:.4f}"
        marker = "  ◀" if mae == best_overall else ""
        print(f"  {name:<36}  {mae:>10.4f}  {d_str:>9}{marker}")

# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────
def load_california():
    """
    California Housing (sklearn)
    Predict median house value (in $100K) from 8 geographic/demographic features.
    20,640 samples. Strong geographic non-linearity: coastal areas command large
    premiums that linear models systematically underestimate; KNN captures these
    local clusters well. This regional variability is exactly what DES exploits —
    it can hand off coastal samples to KNN and inland samples to linear/boosting.
    """
    print("  Loading California Housing…", end=' ', flush=True)
    X, y = fetch_california_housing(return_X_y=True)
    print("done")
    return X, y, 'California Housing', X.shape[1]


def load_ames():
    """
    Ames Housing (OpenML id 42165)
    Predict house sale price from ~80 features (mix of numeric and categorical).
    ~1,460 samples. Highly non-linear with many interaction effects and outliers.
    Linear models are severely penalised by outlier mansions; KNN benefits from
    neighbourhood-level price clustering; boosting handles the skewed distribution.
    More variability than California Housing on a per-model basis.
    """
    print("  Fetching Ames Housing from OpenML…", end=' ', flush=True)
    d   = fetch_openml(data_id=42165, as_frame=True, parser='auto')
    X   = d.data.copy()

    # Encode categoricals ordinally, then impute all missings
    cat_cols = X.select_dtypes(['category', 'object']).columns
    X[cat_cols] = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    ).fit_transform(X[cat_cols])
    X   = SimpleImputer(strategy='median').fit_transform(X.astype(float))

    y   = d.target.astype(float).values
    print("done")
    return X, y, 'Ames Housing', X.shape[1]

# ─────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────
def build_models():
    """
    Four regressors with maximally diverse inductive biases.
    SVR was dropped — its RBF kernel requires dense, low-dimensional data and
    produced catastrophic errors on the 80-feature Ames dataset (3x worse than
    the best model), which corrupted DES local routing estimates everywhere it
    appeared in a neighborhood.

      Linear Reg.    Globally linear, fast. Systematically wrong on non-linear
                     or interaction-driven regions — a useful anchor/floor.

      Ridge Reg.     L2-regularised linear. Handles collinear / many features
                     much better than plain OLS (critical for Ames, 80 features).
                     Different failure profile on sparse regions.

      KNN Regressor  Pure local averaging — no global structure at all.
                     Excels in dense clustered regions (e.g. coastal CA prices);
                     degrades in sparse or high-variance areas.

      Hist. Boosting Sequential boosting — concentrates capacity on hard samples.
                     Different error pattern from KNN/linear on the same inputs.
    """
    return {
        'Linear Reg.': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc',  StandardScaler()),
            ('m',   LinearRegression()),
        ]),
        'Ridge Reg.': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc',  StandardScaler()),
            ('m',   Ridge(alpha=10.0)),
        ]),
        'KNN Regressor': Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc',  StandardScaler()),
            ('m',   KNeighborsRegressor(n_neighbors=10, n_jobs=-1)),
        ]),
        'Hist. Boosting': HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=SEED,
        ),
    }

# ─────────────────────────────────────────────────────────────────────
# GLOBAL ENSEMBLE  (scipy Nelder-Mead)
# ─────────────────────────────────────────────────────────────────────
def fit_global_ensemble(val_preds, y_val):
    """
    Finds w* = argmin_w  MAE( Σ w_i · ŷ_i(x),  y )  over the validation set
    using gradient-free Nelder-Mead (scipy.optimize).

    Weights are constrained to be non-negative and sum to 1 by taking abs()
    and normalising inside the objective, avoiding the need for explicit bounds.
    The result is a single fixed vector applied to every test sample — the
    standard competition ensembling approach.
    """
    names   = list(val_preds.keys())
    stacked = np.stack([val_preds[n] for n in names])   # (M, N_val)

    def objective(w):
        w = np.abs(w)
        w /= w.sum()
        blended = w @ stacked                           # (N_val,)
        return mean_absolute_error(y_val, blended)

    w0  = np.ones(len(names)) / len(names)
    res = minimize(objective, w0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    w   = np.abs(res.x)
    w  /= w.sum()
    return dict(zip(names, w))


def apply_global_weights(preds, weights):
    """Blend scalar regression outputs with a fixed weight vector."""
    names = list(weights.keys())
    w     = np.array([weights[n] for n in names])
    return w @ np.stack([preds[n] for n in names])     # (N_test,)

# ─────────────────────────────────────────────────────────────────────
# DES PREDICTION
# ─────────────────────────────────────────────────────────────────────
def des_predict(router, X_test, test_preds, temperature=0.3):
    """
    Apply DES per-sample weights to scalar regression predictions.

    For knn-dw, weights are a soft blend based on local MAE in the K-nearest
    validation neighbors. For OLA, it's a one-hot selection of the locally
    best model. Both reduce to a weighted sum of scalar predictions.
    """
    names  = list(test_preds.keys())
    result = router.predict(X_test, temperature=temperature)
    if isinstance(result, dict):
        result = [result]

    preds = []
    for i, w in enumerate(result):
        blended = sum(w[n] * test_preds[n][i] for n in names)
        preds.append(blended)
    return np.array(preds)

# ─────────────────────────────────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────────────────────────────────
def run(loader):
    X, y, ds_name, n_features = loader()
    dataset_header(ds_name, len(X), n_features, float(y.std()))

    # ── Split ─────────────────────────────────────────────────────────
    # Train 60% / Val 20% / Test 20% — strict separation throughout.
    # Ensembles only ever see val labels during fitting; test is untouched.
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED)
    X_tr, X_val, y_tr, y_val   = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=SEED)

    print(f"\n  Split  →  {len(X_tr):,} train  /  "
          f"{len(X_val):,} val  /  {len(X_test):,} test")

    # ── Train base models ─────────────────────────────────────────────
    section("Training models  (reporting val-set MAE)")
    models     = build_models()
    val_preds  = {}
    test_preds = {}
    val_maes   = {}

    for mname, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        val_preds[mname]  = model.predict(X_val)
        test_preds[mname] = model.predict(X_test)
        val_maes[mname]   = mean_absolute_error(y_val, val_preds[mname])
        print(f"    ✓ {mname:<22}  MAE = {val_maes[mname]:.4f}   ({time.time()-t0:.1f}s)")

    best_name = min(val_maes, key=val_maes.get)

    # ── Fit ensembles on val set ───────────────────────────────────────
    section("Fitting ensembles  (val set only — test never touched)")

    t0   = time.time()
    ge_w = fit_global_ensemble(val_preds, y_val)
    print(f"    ✓ Global Ensemble  (Nelder-Mead, {time.time()-t0:.1f}s)")
    for n, w in ge_w.items():
        print(f"        {n:<22}  weight = {w:.3f}")

    # Standardize features for DES — critical because the KNN inside DES
    # computes Euclidean distances in feature space. Without scaling, high-
    # magnitude features (e.g. LotArea in Ames) dominate distances and make
    # all neighborhoods essentially random. Scaler is fit on val only so no
    # test information leaks into the neighborhood structure.
    des_scaler  = StandardScaler().fit(X_val)
    X_val_s     = des_scaler.transform(X_val)
    X_test_s    = des_scaler.transform(X_test)

    # DES receives hard predictions (scalar per sample) + MAE metric.
    # MAE is continuous — within a 20-neighbor window, a model off by 50k
    # is meaningfully penalised vs one off by 5k. The score matrix is
    # normalized internally to [0,1] so temperature=1.0 produces meaningful
    # softmax gradations regardless of the metric's absolute scale.
    t0         = time.time()
    router_knn = DynamicRouter(task='regression', dtype='tabular',
                               method='knn-dw', metric='mae', mode='min',
                               k=K, preset='exact')
    router_knn.fit(X_val_s, y_val, val_preds)
    print(f"    ✓ DES — knn-dw  (k={K}, MAE metric, {time.time()-t0:.1f}s)")

    t0         = time.time()
    router_ola = DynamicRouter(task='regression', dtype='tabular',
                               method='ola', metric='mae', mode='min',
                               k=K, preset='exact')
    router_ola.fit(X_val_s, y_val, val_preds)
    print(f"    ✓ DES — OLA     (k={K}, MAE metric, {time.time()-t0:.1f}s)")

    # ── Evaluate on held-out test set ─────────────────────────────────
    section("Results on held-out test set  (MAE — lower is better)")

    best_mae    = mean_absolute_error(y_test, test_preds[best_name])
    ge_mae      = mean_absolute_error(y_test, apply_global_weights(test_preds, ge_w))
    des_knn_mae = mean_absolute_error(y_test, des_predict(router_knn, X_test_s, test_preds))
    des_ola_mae = mean_absolute_error(y_test, des_predict(router_ola, X_test_s, test_preds))

    rows = [
        (f"Best Single  ({best_name})",          best_mae),
        ("Global Ensemble  (Nelder-Mead)",        ge_mae),
        ("DES — knn-dw     (this library)",       des_knn_mae),
        ("DES — OLA        (this library)",       des_ola_mae),
    ]
    show_results(rows, best_mae)

# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    banner()
    run(load_california)
    run(load_ames)
    print(f"\n\n{'━' * W}\n  Done.\n{'━' * W}\n")