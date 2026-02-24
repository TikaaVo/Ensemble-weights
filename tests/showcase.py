#!/usr/bin/env python3
"""
Dynamic Ensemble Selection — Showcase (Regression)
===================================================
Compares four strategies on two real regression datasets.

  Best Single      Best individual model on validation data, applied uniformly.
  Global Ensemble  Single fixed weight vector learned via Nelder-Mead on val set.
  DES knn-dw       Per-sample blending weighted by local MAE in K val neighbors.
  DES OLA          Same neighborhood lookup; hard selection of the local best.

Datasets
  California Housing  20K samples, 8 features. Coastal/inland regime split.
  Bike Sharing        17K samples, 8 numeric features. Temporal/weather regimes.

Models
  Linear Regression, KNN Regressor, and Hist. Gradient Boosting.
  Diverse inductive biases give DES meaningfully different experts to route between.

Why regression?
  MAE gives a continuous per-sample signal. Classification accuracy (0/1) is too
  coarse for softmax weighting to differentiate models within a small neighborhood.

Install:  pip install scikit-learn scipy
Runtime:  ~2-4 min on a MacBook Air M3
"""

import warnings
import time
import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

from ensemble_weights import DynamicRouter

warnings.filterwarnings('ignore')

SEED      = 42
K         = 20     # neighbors for DES
THRESHOLD = 0.5    # competence gate: exclude models below 50% of best local score
TEMP      = 0.1    # softmax temperature; lower = sharper routing

W = 66


def banner():
    print(f"\n{'━' * W}")
    print("  Dynamic Ensemble Selection — Showcase  (Regression)")
    print(f"{'━' * W}")
    print("  Best Single       best val-set model applied everywhere")
    print("  Global Ensemble   fixed weights, Nelder-Mead on val set")
    print("  DES knn-dw        per-sample adaptive blending  ← this library")
    print("  DES OLA           per-sample hard model selection  ← this library")
    print("  Metric: MAE (lower is better)")
    print(f"{'━' * W}")


def dataset_header(name, n_samples, n_features, y_std):
    print(f"\n\n{'━' * W}")
    print(f"  Dataset: {name}")
    print(f"  {n_samples:,} samples  ·  {n_features} features  ·  target std = {y_std:.2f}")
    print(f"{'━' * W}")


def section(title):
    print(f"\n  {title}")
    print(f"  {'-' * (W - 4)}")


def show_results(rows, best_mae):
    """Print results table; marks overall winner and delta vs best single model."""
    best_overall = min(mae for _, mae in rows)
    print(f"\n  {'Method':<36} {'Test MAE':>10}  {'vs Best':>9}")
    print(f"  {'-' * 36}  {'-' * 10}  {'-' * 9}")
    for name, mae in rows:
        delta  = mae - best_mae
        d_str  = "    —      " if delta == 0 else f" {'+' if delta >= 0 else ''}{delta:.4f}"
        marker = "  ◀" if mae == best_overall else ""
        print(f"  {name:<36}  {mae:>10.4f}  {d_str:>9}{marker}")


# ── Data loading ──────────────────────────────────────────────────────

def load_california():
    """California Housing (sklearn builtin). Predict median house value ($100K)."""
    print("  Loading California Housing...", end=' ', flush=True)
    X, y = fetch_california_housing(return_X_y=True)
    print("done")
    return X, y, 'California Housing', X.shape[1]


def load_bike():
    """Bike Sharing (OpenML 42712). Predict hourly rental count."""
    print("  Fetching Bike Sharing from OpenML...", end=' ', flush=True)
    d = fetch_openml(data_id=42712, as_frame=True, parser='auto')
    X = d.data.select_dtypes(include=['number']).astype(float)
    X = SimpleImputer(strategy='median').fit_transform(X)
    y = d.target.astype(float).values
    print("done")
    return X, y, 'Bike Sharing (Hourly)', X.shape[1]


# ── Models ────────────────────────────────────────────────────────────

def build_models():
    """
    Three regressors with orthogonal inductive biases.

    Linear Reg.      Globally linear; correct where the relationship is simple,
                     systematically wrong in non-linear or interaction-driven regions.
    KNN Regressor    Pure local averaging; excels in dense clusters, degrades in sparse areas.
    Hist. Boosting   Sequential boosting; best on hard samples other models miss.
    """
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
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=SEED,
        ),
    }


# ── Global ensemble ───────────────────────────────────────────────────

def fit_global_ensemble(val_preds, y_val):
    """
    Solve w* = argmin_w MAE(Σ w_i·ŷ_i, y) on the val set via Nelder-Mead.
    One fixed weight vector applied to every test sample regardless of local structure.
    """
    names   = list(val_preds.keys())
    stacked = np.stack([val_preds[n] for n in names])   # (M, N_val)

    def objective(w):
        w = np.abs(w); w /= w.sum()
        return mean_absolute_error(y_val, w @ stacked)

    w0  = np.ones(len(names)) / len(names)
    res = minimize(objective, w0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    w = np.abs(res.x); w /= w.sum()
    return dict(zip(names, w))


def apply_global_weights(preds, weights):
    """Blend scalar predictions with a fixed weight vector."""
    names = list(weights.keys())
    return np.array([weights[n] for n in names]) @ np.stack([preds[n] for n in names])


# ── DES prediction ────────────────────────────────────────────────────

def des_predict(router, X_test, test_preds, temperature=1.0):
    """Apply per-sample DES weights to scalar model predictions."""
    names  = list(test_preds.keys())
    result = router.predict(X_test, temperature=temperature)
    if isinstance(result, dict):
        result = [result]
    return np.array([
        sum(w[n] * test_preds[n][i] for n in names)
        for i, w in enumerate(result)
    ])


# ── Benchmark ─────────────────────────────────────────────────────────

def run(loader):
    X, y, ds_name, n_features = loader()
    dataset_header(ds_name, len(X), n_features, float(y.std()))

    # Strict three-way split: train 60% / val 20% / test 20%.
    # Ensembles fit on val only; test is never touched until final evaluation.
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    X_tr, X_val, y_tr, y_val   = train_test_split(X_tv, y_tv, test_size=0.25, random_state=SEED)
    print(f"\n  Split → {len(X_tr):,} train / {len(X_val):,} val / {len(X_test):,} test")

    section("Training models  (val-set MAE)")
    models = build_models()
    val_preds, test_preds, val_maes = {}, {}, {}

    for mname, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        val_preds[mname]  = model.predict(X_val)
        test_preds[mname] = model.predict(X_test)
        val_maes[mname]   = mean_absolute_error(y_val, val_preds[mname])
        print(f"    ✓ {mname:<22}  MAE = {val_maes[mname]:.4f}   ({time.time()-t0:.1f}s)")

    best_name = min(val_maes, key=val_maes.get)

    section("Fitting ensembles  (val set only)")

    t0 = time.time()
    ge_w = fit_global_ensemble(val_preds, y_val)
    print(f"    ✓ Global Ensemble  (Nelder-Mead, {time.time()-t0:.1f}s)")
    for n, w in ge_w.items():
        print(f"        {n:<22}  weight = {w:.3f}")

    # Scaler fit on val only — test information must not influence neighborhood structure.
    des_scaler = StandardScaler().fit(X_val)
    X_val_s    = des_scaler.transform(X_val)
    X_test_s   = des_scaler.transform(X_test)

    t0 = time.time()
    router_knn = DynamicRouter(
        task='regression', dtype='tabular', method='knn-dw',
        metric='mae', mode='min', k=K, preset='exact',
        competence_threshold=THRESHOLD,
    )
    router_knn.fit(X_val_s, y_val, val_preds)
    print(f"    ✓ DES knn-dw  (k={K}, gate={THRESHOLD}, temp={TEMP}, {time.time()-t0:.1f}s)")

    t0 = time.time()
    router_ola = DynamicRouter(
        task='regression', dtype='tabular', method='ola',
        metric='mae', mode='min', k=K, preset='exact',
    )
    router_ola.fit(X_val_s, y_val, val_preds)
    print(f"    ✓ DES OLA     (k={K}, hard select, {time.time()-t0:.1f}s)")

    section("Results on held-out test set  (MAE — lower is better)")

    best_mae    = mean_absolute_error(y_test, test_preds[best_name])
    ge_mae      = mean_absolute_error(y_test, apply_global_weights(test_preds, ge_w))
    des_knn_mae = mean_absolute_error(y_test, des_predict(router_knn, X_test_s, test_preds, temperature=TEMP))
    des_ola_mae = mean_absolute_error(y_test, des_predict(router_ola, X_test_s, test_preds))

    show_results([
        (f"Best Single  ({best_name})",               best_mae),
        ("Global Ensemble  (Nelder-Mead)",             ge_mae),
        (f"DES knn-dw  (gate={THRESHOLD}, T={TEMP})",  des_knn_mae),
        ("DES OLA  (hard select)",                     des_ola_mae),
    ], best_mae)


if __name__ == '__main__':
    banner()
    run(load_california)
    run(load_bike)
    print(f"\n\n{'━' * W}\n  Done.\n{'━' * W}\n")