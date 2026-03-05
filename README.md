# despy

despy is a flexible, light, and easy-to-use ensembling library that implements
Dynamic Ensemble Selection (DES) algorithms for ensembling multiple ML models
on a singular dataset. 

The library works entirely with data, taking as input a validation dataset 
along with pre-computed predictions and outputting a dictionary of weights 
per model. This means that it can be used with any library or model without 
requiring any wrappers, including custom models, popular ML libraries, and APIs.

despy contains multiple different DES algorithms, and it works with both classification
and regression.

# Dynamic Ensemble Selection

Ensemble learning in machine learning refers to when multiple models trained on a 
single dataset combine their predictions to create a single, more accurate prediction,
usually through weighted voting or picking the best model.

DES refers to techniques where the models or their voting weights are selected dynamically
for every test case. This selection bases on the idea of competence regions, which is the 
concept that there are regions of feature space where certain models perform particularly well,
so every base model can be an expert in a different region.
Only the most competent, or an ensemble of the most competent models is selected for the prediction.

Through empirical studies, DES has been shown to perform best with small-sized, imbalanced, or 
heterogeneous datasets, as well as non-stationary data (concept drift), models that haven't perfected a dataset, 
and when used on an ensemble of models with differing architectures and perspectives.

---

## Installation

```bash
pip install despy

# The library runs with Nearest Neighbors from sklearn for exact KNN
pip install scikit-learn

# Alternatively, ANN can be used for faster runtimes at the cost of
# slightly lower accuracy. The following three are supported;
# Install the one you want to use.
pip install faiss-cpu   # FAISS (good default for most datasets)
pip install annoy       # Annoy (memory-efficient, simple)
pip install hnswlib     # HNSW (best for high-dimensional data)
```

---

## Dependencies

Python (>= 3.9)

NumPy (>= 1.21)

---

## Quick start

Full explanation of the algorithms, syntax, and parameters is available in the documentation.

```python
from despy.des.knorau  import KNORAU

# 1. Train your models
models = {"rf": rf, "xgb": xgb, "mlp": mlp}

# 2. Get predictions on a held-out validation set
#    Regression: scalar arrays
#    Classification: probability arrays OR hard predictions
val_preds = {name: m.predict_proba(X_val) for name, m in models.items()}

# 3. Fit the router
router = KNORAU(task="classification", metric="Accuracy", mode="min", k=20)
router.fit(X_val, y_val, val_preds)

# 4. Route test samples
test_preds = {name: m.predict_proba(X_test) for name, m in models.items()}

for i, x in enumerate(X_test):
    weights = router.predict(x, temperature=0.1)
    # weights example: {"rf": 0.7, "xgb": 0.2, "mlp": 0.1}
    prediction = sum(weights[n] * test_preds[n][i] for n in weights)
```

For classification with probability arrays, blend the output the same way to
get a final probability distribution, then take the argmax.

---

## Why despy?

Most DES libraries are tied to scikit-learn. despy only ever sees a numpy
feature matrix and a dict of prediction arrays, so the models themselves are
never touched after training. This allows for more flexibility and a lighter library.

Furthermore, despy works with both classification and regression, while the majority of DES
libraries and literature is focused only on classification tasks.

```python
# PyTorch example 
with torch.no_grad():
    val_preds  = {name: m(X_val_t).cpu().numpy()  for name, m in models.items()}
    test_preds = {name: m(X_test_t).cpu().numpy() for name, m in models.items()}

router = KNORAU(task="classification", metric="accuracy", mode="max", k=20)
router.fit(X_val, y_val, val_preds)
weights = router.predict(X_test[i])
```

---

## Algorithms

| Method | Best for | Notes |
|---|---|---|
| `KNNDWS` | Regression | Softmax over neighbourhood-averaged scores. Temperature controls sharpness. |
| `KNORAU` | Classification | Vote-count weighting. Each model earns one vote per neighbour it correctly classifies. |
| `KNORAE` | Classification | Intersection-based. Only models correct on all neighbours survive; falls back to smaller neighbourhoods. |
| `KNORAIU` | Classification | Like KNORA-U but votes are inverse-distance weighted. |
| `OLA` | Both | Hard selection: only the single best model in the neighbourhood contributes. |

---

## ANN backends

despy supports three Approximate Nearest Neighbour backends plus exact search:

| Preset | Backend | Install | Notes |
|---|---|---|---|
| `exact` | sklearn KNN |  `scikit-learn` | Exact, no extra deps |
| `balanced` | FAISS IVF | `faiss-cpu` | ~98% recall, good default |
| `fast` | FAISS IVF | `faiss-cpu` | ~95% recall, faster queries |
| `turbo` | FAISS flat | `faiss-cpu` | Exact via FAISS, GPU-friendly |
| `high_dim_balanced` | HNSW | `hnswlib` | Best for >100 features, balanced |
| `high_dim_fast` | HNSW | `hnswlib` | Best for >100 features, faster |

Annoy is also available as a custom backend — memory-efficient and simple,
good for datasets that need to be persisted to disk.

```python
# Exact search (no extra deps)
router = KNORAU(..., preset="exact")

# High-dimensional data
router = KNORAU(..., preset="high_dim_balanced")

# Custom FAISS config
router = KNORAU(..., preset="custom", finder="faiss",
                index_type="ivf", n_probes=50)

# Annoy
router = KNORAU(..., preset="custom", finder="annoy",
                n_trees=100, search_k=-1)
```

---

## Custom metrics

Any callable `(y_true, y_pred) -> float` works:

```python
def pinball(y_true, y_pred, alpha=0.9):
    e = y_true - y_pred
    return alpha * e if e >= 0 else (alpha - 1) * e

router = KNNDWS(task="regression", metric=pinball, mode="min", k=20)
```

Built-in metric strings: `accuracy`, `mae`, `mse`, `rmse`, `log_loss`, `prob_correct`.

---

## Benchmark results

10-seed benchmark (seeds 0–9) on standard sklearn and OpenML datasets. "Best Single" is the best
individual model selected on the validation set. "Simple Average" is uniform
equal-weight blending, included as a baseline.

Pool: KNN, Random Forest, Hist. Gradient Boosting, SVM-RBF (C=2), MLP.

### Regression (MAE, lower is better)

% shown as delta vs Best Single. 10-seed mean.

| Dataset | Best Single | Simple Avg | despy best |
|---|---|---|---|
| California Housing (sklearn) | 0.3370 | +2.0% | **−3.2%** (KNN-DWS) |
| Bike Sharing (OpenML) | 31.02 | +32.8% | **−0.4%** (KNN-DWS) |
| Abalone (OpenML) | 1.5479 | −1.5% | **−1.5%** (KNORA-U) |

despy beats best single and simple averaging on every regression dataset across all 10 seeds.
Simple Average performs poorly on Bike Sharing (+32.8%), because pool models specialise
heavily by time-of-day pattern, so equal-weight blending is actively harmful.

### Classification (Accuracy, higher is better)

% shown as delta vs Best Single. 10-seed mean.

| Dataset | Best Single | Simple Avg | despy best |
|---|---|---|---|
| Waveform (OpenML) | 84.94% | +0.40% | **+0.40%** (KNORA-U/IU) |
| Satimage (OpenML) | 91.34% | +0.15% | **+0.19%** (KNORA-IU) |
| MNIST Digits (sklearn) | 96.83% | +0.66% | **+0.83%** (KNORA-E) |
| Pendigits (OpenML) | 99.02% | +0.25% | **+0.32%** (KNORA-E) |

despy beats best single and simple averaging on every classification dataset across all 10 seeds.

### Speed (mean ms fit + predict, 10 seeds)

| Dataset | despy |
|---|---|
| Waveform | 9–12 ms |
| Satimage | 11–15 ms |
| MNIST Digits | 4–5 ms |
| Pendigits | 19–23 ms |

despy caches all model predictions on the validation set at fit time and reads
from that matrix at inference.

---

## Contributing

Issues and PRs welcome.
