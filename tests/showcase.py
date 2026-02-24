#!/usr/bin/env python3
"""
Dynamic Ensemble Selection (DES) — Showcase
============================================
Dataset : Forest CoverType  (581 k samples → 70 k used, 54 features, 7 classes)
Models  : DecisionTree · LogisticRegression · MLP · RandomForest · HistGBM
Compared: Best Single Model  vs  Global Ensemble (Nelder-Mead)  vs  DES

Install requirements:
    pip install scikit-learn scipy faiss-cpu
    pip install -e .   # installs ensemble_weights from this repo

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠ BUG NOTE – read before using the library elsewhere:

  preds_dict must contain 1-D class-label arrays  (clf.predict)
  NOT 2-D probability arrays                       (clf.predict_proba)

  Passing probabilities causes np.vectorize to iterate over every
  element of the N×C matrix individually rather than every row,
  making fit() appear to hang for hours without using CPU.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings
import time
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from scipy.special import softmax

from ensemble_weights import DynamicRouter

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 58)
print("  Dynamic Ensemble Selection — Showcase")
print("=" * 58)
print("\n[1/4] Loading Forest CoverType dataset …")

X_full, y_full = fetch_covtype(return_X_y=True)
y_full -= 1  # labels 1–7 → 0–6

# Subsample so training stays fast while still being a meaningful benchmark
rng = np.random.default_rng(42)
idx = rng.choice(len(X_full), size=70_000, replace=False)
X, y = X_full[idx], y_full[idx]

# 60 % train  ·  20 % val (used by DES and global ensemble)  ·  20 % test
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train)
X_val    = scaler.transform(X_val)
X_test   = scaler.transform(X_test)

print(
    f"    Train : {len(X_train):,} samples\n"
    f"    Val   : {len(X_val):,} samples  (used for ensemble fitting)\n"
    f"    Test  : {len(X_test):,} samples  (held-out evaluation)"
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRAIN DIVERSE MODELS
#     Intentionally varied: shallow to deep, linear to non-linear.
#     They should differ enough for ensembling to matter.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Training base models …")

MODELS = {
    "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "LogisticReg":  LogisticRegression(max_iter=500,  n_jobs=-1, random_state=42),
    "MLP":          MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                                  random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "HistGBM":      HistGradientBoostingClassifier(max_iter=200, random_state=42),
}

val_labels  = {}   # 1-D class predictions on val set  — what DES expects
val_probas  = {}   # 2-D probabilities on val set       — what global ensemble uses
test_labels = {}
test_probas = {}

val_accs    = {}

for name, clf in MODELS.items():
    print(f"    Fitting {name} …", end=" ", flush=True)
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    val_labels[name]  = clf.predict(X_val)          # ← 1-D, correct for DES
    val_probas[name]  = clf.predict_proba(X_val)    # ← 2-D, for global ensemble
    test_labels[name] = clf.predict(X_test)
    test_probas[name] = clf.predict_proba(X_test)

    val_accs[name] = accuracy_score(y_val, val_labels[name])
    print(f"val acc = {val_accs[name]:.4f}  ({elapsed:.1f}s)")

model_names = list(MODELS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# 3.  BASELINE — Best Single Model
#     Selects the model with the highest validation accuracy.
#     Fair because DES and global ensemble also see the validation set.
# ─────────────────────────────────────────────────────────────────────────────
best_name          = max(val_accs, key=val_accs.get)
best_single_acc    = accuracy_score(y_test, test_labels[best_name])

# ─────────────────────────────────────────────────────────────────────────────
# 4.  GLOBAL ENSEMBLE — fixed softmax weights, gradient-free optimisation
#     Uses scipy Nelder-Mead (no gradients needed, just accuracy on val set).
#     This is the most widely used approach when you only have accuracy/loss
#     and not class probabilities from a meta-learner.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Fitting Global Ensemble (Nelder-Mead weight search) …")

val_probas_arr  = np.stack([val_probas[n]  for n in model_names], axis=0)  # (M, N, C)
test_probas_arr = np.stack([test_probas[n] for n in model_names], axis=0)


def neg_val_acc(raw_weights):
    """Objective: minimise –accuracy by adjusting un-normalised weights."""
    w       = softmax(raw_weights)                             # sum to 1, all positive
    blended = np.einsum("m,mnc->nc", w, val_probas_arr)       # weighted average of probs
    preds   = blended.argmax(axis=1)
    return -accuracy_score(y_val, preds)


t0     = time.time()
result = minimize(
    neg_val_acc,
    x0      = np.ones(len(model_names)),
    method  = "Nelder-Mead",
    options = {"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-4},
)
elapsed = time.time() - t0

global_weights   = softmax(result.x)
blended_test     = np.einsum("m,mnc->nc", global_weights, test_probas_arr)
global_test_acc  = accuracy_score(y_test, blended_test.argmax(axis=1))
print(f"    Done in {elapsed:.1f}s  (val acc = {-result.fun:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  DES — DynamicRouter
#     FAISS IVF (preset='balanced') finds approximate nearest neighbours fast.
#     For each test instance it locates its k=15 nearest validation neighbours,
#     looks up each model's historical accuracy on those neighbours, and turns
#     those scores into soft routing weights via softmax.
#
#     KEY: pass class-label predictions (val_labels), not probabilities.
#          See the bug note at the top of this file.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Fitting DynamicRouter (balanced preset — FAISS IVF) …")

router = DynamicRouter(
    task    = "classification",
    dtype   = "tabular",
    method  = "knn-dw",
    metric  = "accuracy",
    mode    = "max",
    preset  = "balanced",   # FAISS IVF, ~98 % recall, much faster than exact KNN
    k       = 15,
)

t0 = time.time()
router.fit(X_val, y_val, val_labels)   # ← val_labels, not val_probas
print(f"    Fitted in {time.time() - t0:.1f}s")

print("    Predicting on test set …", end=" ", flush=True)
t0 = time.time()

# predict() returns a list of weight-dicts for batch_size > 1, single dict otherwise
weights_list = router.predict(X_test)
if isinstance(weights_list, dict):          # safety: wrap single-sample result
    weights_list = [weights_list]

des_preds = []
for j, w_dict in enumerate(weights_list):
    # Blend per-model probability vectors using DES-assigned weights
    blended = sum(w_dict[n] * test_probas[n][j] for n in model_names)
    des_preds.append(blended.argmax())

des_test_acc = accuracy_score(y_test, des_preds)
print(f"done in {time.time() - t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  RESULTS — Forest CoverType (7-class), test set")
print("=" * 58)

print("\nIndividual model validation accuracies:")
for name, acc in sorted(val_accs.items(), key=lambda x: -x[1]):
    tag = "  ← best single" if name == best_name else ""
    print(f"    {name:<20} {acc:.4f}{tag}")

print(f"\n{'Method':<34} {'Test Acc':>8}  {'vs Best':>8}")
print("-" * 52)

entries = [
    (f"Best Single Model ({best_name})", best_single_acc),
    ("Global Ensemble (Nelder-Mead)",     global_test_acc),
    ("DES – DynamicRouter (knn-dw)",      des_test_acc),
]

for label, acc in entries:
    delta = acc - best_single_acc
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<32} {acc:.4f}  {sign}{delta:.4f}")

print("=" * 58)

print("\nGlobal Ensemble — learned per-model weights:")
for name, w in sorted(zip(model_names, global_weights), key=lambda x: -x[1]):
    bar = "█" * int(w * 40)
    print(f"    {name:<20} {w:.4f}  {bar}")

print(
    "\n[Interpretation]\n"
    "  Best Single Model picks one model for every test instance.\n"
    "  Global Ensemble uses a fixed weighted vote across all instances.\n"
    "  DES routes each instance to the models that performed best on\n"
    "  its local neighbourhood — capturing that different models can\n"
    "  be better at different parts of the feature space.\n"
)