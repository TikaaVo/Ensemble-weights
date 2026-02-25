#!/usr/bin/env python3
"""
Dynamic Ensemble Selection — Showcase (Regression)
===================================================
Compares seven strategies on two real regression datasets:

  Best Single       best individual model on validation data, applied uniformly.
  Global Ensemble   single fixed weight vector learned via Nelder‑Mead on val set.
  DES knn‑dws       per‑sample blending (softmax) weighted by local MAE in K neighbours.
  DES OLA           per‑sample hard selection of the single locally best model.
  DES KNORA‑U       per‑sample voting: weight = fraction of neighbours on which a model is competent.
  DES KNORA‑E       per‑sample: find largest neighbourhood where at least one model is competent on ALL neighbours.

Datasets
  California Housing  20K samples, 8 features. Coastal/inland regime split.
  Bike Sharing        17K samples, 8 numeric features. Temporal/weather regimes.

Models
  Linear Regression, KNN Regressor, and Hist. Gradient Boosting.
  Diverse inductive biases give DES meaningfully different experts to route between.

Why regression?
  MAE gives a continuous per‑sample signal. Classification accuracy (0/1) is too
  coarse for softmax weighting to differentiate models within a small neighbourhood.

Install:  pip install scikit-learn scipy
Runtime:  ~3‑5 min on a MacBook Air M3
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

SEED = 42
K    = 20    # neighbours for DES
TEMP = 0.1   # softmax temperature for knn-dws; lower = sharper routing

# Threshold interpretation differs by algorithm:
#   knn-dws  — gates averaged neighbourhood scores; 0.5 = exclude bottom half of range
#   ola      — ignored (argmax needs no threshold)
#   knora-u  — per-neighbour binary competence; 1.0 = only the strictly best model
#              votes on each neighbour, matching the original classification definition
#   knora-e  — same per-neighbour criterion as knora-u; 1.0 is the correct analogue
#
# Why 1.0 for KNORA in regression?
#   KNORA was designed for 0/1 classification accuracy, where "competent" means
#   "correct" and normalization yields 1.0 for correct models, 0.0 for wrong ones.
#   With continuous regression error, per-neighbour normalization still produces
#   exactly 1.0 for the best model and 0.0 for the worst, but intermediate models
#   land somewhere in between based purely on relative rank within the range.
#   threshold=0.5 then lets "second-best" models earn votes even when they are
#   substantially worse, which corrupts the ensemble on datasets where one model
#   dominates (e.g. Bike Sharing: Linear MAE=105 vs Boosting MAE=42).
#   threshold=1.0 recovers the binary "oracle" criterion: only the strictly best
#   model on each neighbour votes, which is the correct regression analogue of
#   "correct classification."
THRESHOLDS = {
    'knn-dws':  0.5,
    'ola':      0.5,   # ignored by OLA
    'knora-u':  1.0,
    'knora-e':  1.0,
}

W = 80


def banner():
    print(f"\n{'━' * W}")
    print("  Dynamic Ensemble Selection — Showcase  (Regression)")
    print(f"{'━' * W}")
    print("  Best Single       best val-set model applied everywhere")
    print("  Global Ensemble   fixed weights, Nelder-Mead on val set")
    print("  DES knn-dws       per-sample adaptive blending  ← this library")
    print("  DES OLA           per-sample hard model selection  ← this library")
    print("  DES KNORA-U       per-sample voting (union of competent models)  ← this library")
    print("  DES KNORA-E       per-sample intersection (models correct on all neighbours)  ← this library")
    print("  Metric: MAE (lower is better)")
    print(f"{'━' * W}")
    print("  Note: KNORA-U/E use threshold=1.0 (only strictly-best model per neighbour")
    print("  earns a vote). This matches the original binary 'oracle' criterion. See")
    print("  THRESHOLDS in source for a full explanation of why 0.5 harms regression.")
    print(f"{'━' * W}")


def dataset_header(name, n_samples, n_features, y_mean, y_std):
    print(f"\n\n{'━' * W}")
    print(f"  Dataset: {name}")
    print(f"  {n_samples:,} samples  ·  {n_features} features  ·  "
          f"target mean = {y_mean:.2f}  std = {y_std:.2f}")
    print(f"{'━' * W}")


def section(title):
    print(f"\n  {title}")
    print(f"  {'-' * (W - 4)}")


def show_results(rows, best_mae, y_mean):
    """
    Print results table with three columns:
      MAE        — raw error in target units
      % of mean  — MAE as a fraction of the target mean; intuitive size‑of‑error gauge
      vs Best    — % improvement (negative = better) relative to the best single model
    """
    best_overall = min(mae for _, mae in rows)
    print(f"\n  {'Method':<44} {'MAE':>8}  {'% of mean':>10}  {'vs Best':>9}")
    print(f"  {'-' * 44}  {'-' * 8}  {'-' * 10}  {'-' * 9}")
    for name, mae in rows:
        pct_mean = mae / y_mean * 100
        delta    = (mae - best_mae) / best_mae * 100
        d_str    = "    —    " if mae == best_mae else f"{'+' if delta >= 0 else ''}{delta:.2f}%"
        marker   = "  ◀" if mae == best_overall else ""
        print(f"  {name:<44}  {mae:>8.4f}  {pct_mean:>9.2f}%  {d_str:>9}{marker}")


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
                     systematically wrong in non‑linear or interaction‑driven regions.
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
    Solve w* = argmin_w MAE(Σ w_i·ŷ_i, y) on the val set via Nelder‑Mead.
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

def des_predict(router, X_test, test_preds, temperature=1.0, threshold=0.5):
    """Apply per‑sample DES weights to scalar model predictions."""
    names  = list(test_preds.keys())
    result = router.predict(X_test, temperature=temperature, threshold=threshold)
    if isinstance(result, dict):
        result = [result]
    return np.array([
        sum(w[n] * test_preds[n][i] for n in names)
        for i, w in enumerate(result)
    ])


# ── Benchmark ─────────────────────────────────────────────────────────

def run(loader):
    X, y, ds_name, n_features = loader()
    dataset_header(ds_name, len(X), n_features, float(y.mean()), float(y.std()))

    # Strict three‑way split: train 60% / val 20% / test 20%.
    # Ensembles fit on val only; test is never touched until final evaluation.
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    X_tr, X_val, y_tr, y_val   = train_test_split(X_tv, y_tv, test_size=0.25, random_state=SEED)
    print(f"\n  Split → {len(X_tr):,} train / {len(X_val):,} val / {len(X_test):,} test")

    section("Training models  (val‑set MAE)")
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
    print(f"    ✓ Global Ensemble  (Nelder‑Mead, {time.time()-t0:.1f}s)")
    for n, w in ge_w.items():
        print(f"        {n:<22}  weight = {w:.3f}")

    # Scaler fit on val only — test information must not influence neighbourhood structure.
    des_scaler = StandardScaler().fit(X_val)
    X_val_s    = des_scaler.transform(X_val)
    X_test_s   = des_scaler.transform(X_test)

    # Dictionary to store all DES routers and their predictions
    des_methods = [
        ('knn-dws',  f'knn-dws  (gate={THRESHOLDS["knn-dws"]}, T={TEMP})'),
        ('ola',       'OLA'),
        ('knora-u',  f'KNORA-U  (threshold={THRESHOLDS["knora-u"]})'),
        ('knora-e',  f'KNORA-E  (threshold={THRESHOLDS["knora-e"]})'),
    ]

    fit_times = {}
    predict_times = {}
    des_predictions = {}

    for method, display_name in des_methods:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            router = DynamicRouter(
                task='regression', dtype='tabular', method=method,
                metric='mae', mode='min', k=K, preset='balanced',
            )
        t0 = time.perf_counter()
        router.fit(X_val_s, y_val, val_preds)
        fit_times[method] = (time.perf_counter() - t0) * 1000

        threshold = THRESHOLDS[method]
        t0 = time.perf_counter()
        preds = des_predict(router, X_test_s, test_preds, temperature=TEMP, threshold=threshold)
        predict_times[method] = (time.perf_counter() - t0) * 1000

        des_predictions[method] = preds
        print(f"    ✓ DES {display_name:<32}  fit: {fit_times[method]:6.2f}ms  |  predict: {predict_times[method]:6.2f}ms")

    section("Results on held‑out test set  (MAE — lower is better)")

    best_mae = mean_absolute_error(y_test, test_preds[best_name])
    ge_mae   = mean_absolute_error(y_test, apply_global_weights(test_preds, ge_w))

    show_results([
        (f"Best Single  ({best_name})",                                      best_mae),
        ("Global Ensemble  (Nelder-Mead)",                                    ge_mae),
        (f"DES knn-dws  (gate={THRESHOLDS['knn-dws']}, T={TEMP})",           mean_absolute_error(y_test, des_predictions['knn-dws'])),
        ("DES OLA",                                                            mean_absolute_error(y_test, des_predictions['ola'])),
        (f"DES KNORA-U  (threshold={THRESHOLDS['knora-u']})",                 mean_absolute_error(y_test, des_predictions['knora-u'])),
        (f"DES KNORA-E  (threshold={THRESHOLDS['knora-e']})",                 mean_absolute_error(y_test, des_predictions['knora-e'])),
    ], best_mae, float(y_test.mean()))

    n_test = len(X_test_s)
    print(f"\n  DES timing summary on {n_test:,} test samples (ms):")
    print(f"    {'Method':<12}  {'Fit (ms)':>8}  {'Predict (ms)':>12}  {'ms/sample':>10}")
    print(f"    {'-'*12}  {'-'*8}  {'-'*12}  {'-'*10}")
    for method, _ in des_methods:
        print(f"    {method:<12}  {fit_times[method]:>8.2f}  {predict_times[method]:>12.2f}  {predict_times[method]/n_test:>10.4f}")


if __name__ == '__main__':
    banner()
    run(load_california)
    run(load_bike)
    print(f"\n\n{'━' * W}\n  Done.\n{'━' * W}\n")