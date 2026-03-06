# despy

[despy](https://TikaaVo.github.io/despy/) is a flexible, light, and easy-to-use ensembling library that implements
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

However, DES is not an automatic improvement. It tends to perform worse when datasets are homogeneous or have low diversity, 
when the validation set isn't a good representation of the test set, when using very high dimensional data or few training samples,
or when a single model dominates a dataset.

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

Full explanation of the algorithms, syntax, and parameters is available in the [documentation](https://TikaaVo.github.io/despy/).

```python
from despy.des.knorau  import KNORAU

# 1. Train your models
models = {"rf": rf, "xgb": xgb, "mlp": mlp}

# 2. Get predictions on a held-out validation set
#    Regression: scalar arrays
#    Classification: probability arrays OR hard predictions
val_preds = {name: m.predict_proba(X_val) for name, m in models.items()}

# 3. Fit the router
router = KNORAU(task="classification", metric="accuracy", mode="max", k=20)
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

20-seed benchmark (seeds 0–19) on standard sklearn and OpenML datasets. "Best Single" is the best
individual model selected on the validation set. "Simple Average" is uniform
equal-weight blending, included as a baseline.

For more detailed benchmark numbers, see the [documentation](https://TikaaVo.github.io/despy/).

Pool: KNN, Decision Tree, SVR, Ridge, Bayesian Ridge.

This pool was selected for having variability in architectures while avoiding a single dominant model.

despy algorithms tested: OLA, KNN-DWS, KNORA-U, KNORA-E, KNORA-IU.

### Regression (MAE, lower is better)

% shown as delta vs Best Single. 10-seed mean.

| Dataset                      | Best Single | Simple Avg | despy best            |
|------------------------------|-----------|---|-----------------------|
| California Housing (sklearn) | 0.3956    | +7.99% | **-2.24%** (KNN-DWS)  |
| Bike Sharing (OpenML)        | 51.6779   | +47.77% | **-5.34%** (KNN-DWS)  |
| Abalone (OpenML)             | **1.4981** | +1.14% | +1.47% (KNORA-U)      |
| Diabetes (sklearn)           | **44.5042** | +3.18% | +1.17% (KNN-DWS)      |
| Conrete Strength (OpenML)    | 5.2686 | +23.66% | **-1.05%** (KNORA-IU) |

despy beats best single and simple averaging on 3/5 regression datasets. This shows how DES can provide a
strong boost if used on the right dataset, but it might be counterproductive if used blindly.

### Classification (Accuracy, higher is better)

% shown as delta vs Best Single. 10-seed mean.

| Dataset                | Best Single | Simple Avg | despy best            |
|------------------------|-------------|--------|-----------------------|
| HAR (OpenML)           | 98.24%      | -0.33% | **+0.14%** (KNN-DWS)  |
| Yeast (OpenML)         | 58.87%      | +0.77% | **+1.66%** (KNORA-IU) |
| Image Segment (OpenML) | 93.70%      | +1.40% | **+2.09%** (KNORA-IU) |
| Waveform (OpenML)      | 89.95%      | -2.05% | **+0.93%** (KNORA-E)  |
| Vowel (OpenML)         | **85.91%**  | -0.98% | -0.40% (KNN-DWS)      |

despy beats or matches best single and simple averaging on 4/5 classification datasets. As seen on regression, DES
can improve or hurt performance, so it must be used wisely, but if used correctly it can show promising results.

### Speed (mean ms fit + predict, 20 seeds, all tested algorithms combined)

Consider that usually it is recommended to only use one algorithm at a time, this benchmark ran five of them at the
same time, so with a single one runtime is expected to be about 5x faster. For this benchmark, `preset='balanced'` was used,
so the backend was an ANN algorithm with FAISS IVF.

| Dataset            | despy    |
|--------------------|----------|
| California Housing | 136.6 ms |
| Bike Sharing       | 115.5 ms |
| Abalone            | 28.5 ms  |
| Diabetes           | 8.1 ms   |
| Conrete Strength   | 9.4 ms   |
| HAR                | 297.5 ms |
| Yeast              | 16.3 ms  |
| Image Segment      | 27.2 ms  |
| Waveform           | 48.9 ms  |
| Vowel              | 16.5 ms  |

despy caches all model predictions on the validation set at fit time and reads
from that matrix at inference.

---

## Contributing

Issues and PRs welcome.
