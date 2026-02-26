# ensemble_weights

**Dynamic Ensemble Selection (DES) for tabular data.**

DES improves on static model ensembles by routing each test sample to the models that are locally most competent — determined by their performance on the nearest validation samples in feature space. When models have distinct regional strengths (one excels on coastal data, another on inland; one on linear boundaries, another near decision boundaries) DES consistently outperforms any single model and often outperforms fixed-weight ensembles.

## Installation

```bash
pip install scikit-learn scipy faiss-cpu
```

For high-dimensional data (>100 features), also install `hnswlib`:

```bash
pip install hnswlib
```

## Quickstart

```python
from ensemble_weights.des.knndws import KNNDWS

# 1. Train your models on training data however you like
model_a.fit(X_train, y_train)
model_b.fit(X_train, y_train)

# 2. Fit the router on a held-out validation set
router = KNNDWS(task='regression', metric='mae', mode='min', k=20)
router.fit(
    X_val,
    y_val,
    {
        'model_a': model_a.predict(X_val),
        'model_b': model_b.predict(X_val),
    }
)

# 3. Get per-sample weights at test time
weights = router.predict(X_test)
# [{'model_a': 0.73, 'model_b': 0.27}, {'model_a': 0.11, 'model_b': 0.89}, ...]

# 4. Apply weights to test predictions
test_preds_a = model_a.predict(X_test)
test_preds_b = model_b.predict(X_test)
final = [
    sum(w[name] * preds[i] for name, preds in {'model_a': test_preds_a, 'model_b': test_preds_b}.items())
    for i, w in enumerate(weights)
]
```

## Core concept

The library does not train models. It routes between them. You bring pre-trained models, generate their predictions on a held-out **validation set**, and pass those predictions to `fit()`. At inference time, `predict()` returns a weight dictionary `{model_name: weight}` for each test sample, which you apply to your own test predictions however suits your task.

**Critical requirement:** `features[i]`, `y[i]`, and every `preds_dict[name][i]` must all refer to the same validation sample. The library validates lengths but cannot detect silent row-ordering mismatches — make sure all arrays come from the same split and in the same order.

---

## Algorithms

Import algorithm classes directly from their modules. All four share the same `fit()` interface; `predict()` accepts `temperature` and `threshold` on every class (irrelevant parameters are silently ignored).

### KNNDWS — K-Nearest Neighbors with Distance-Weighted Softmax

```python
from ensemble_weights.des.knndws import KNNDWS
```

The recommended default. Retrieves the K nearest validation neighbors for each test point, averages each model's local scores, normalizes to [0, 1] within the neighborhood, applies a competence gate, then weights models via softmax.

The only algorithm here that never makes hard binary decisions — every step is soft and continuous. When one model clearly dominates, the softmax concentrates weight on it; when models are similar, weights spread out. This makes it robust to noisy neighbor lookups and degrades gracefully when local signal is weak.

```python
router = KNNDWS(
    task='regression',   # or 'classification'
    metric='mae',        # see Metrics section
    mode='min',          # 'min' if lower scores are better, 'max' if higher
    k=20,                # neighborhood size
    threshold=0.5,       # competence gate: exclude models below this fraction of local best
    temperature=0.1,     # softmax sharpness; lower = sharper routing
    preset='balanced',   # neighbor search backend (see Presets)
)
```

**Temperature guidance:** `0.1` for regression (MAE differences between models are large); `1.0` for classification (log_loss differences are more moderate).

**Threshold guidance:** `0.5` works well in most cases. After per-neighborhood normalization, models below 50% of the local range are excluded before softmax.

---

### OLA — Overall Local Accuracy

```python
from ensemble_weights.des.ola import OLA
```

Hard selection: assigns full weight to the single model with the highest average score across the K nearest neighbors. No blending. Useful as a strong baseline — if OLA and KNNDWS produce similar results, the pool lacks meaningful local diversity.

```python
router = OLA(
    task='regression',
    metric='mae',
    mode='min',
    k=20,
    preset='balanced',
)
```

---

### KNORAU — K-Nearest Oracles (Union)

```python
from ensemble_weights.des.knorau import KNORAU
```

Counts how many of the K nearest neighbors each model is competent on. Weight is proportional to vote count (linear, not softmax). Models with zero votes are excluded.

Works best for **classification with probability metrics** (`log_loss`, `prob_correct`), where per-neighbor normalization creates a continuous competence scale. For regression, use `threshold=1.0` to recover the binary oracle criterion — at lower values, per-neighbor normalization can inflate the apparent competence of weaker models.

```python
router = KNORAU(
    task='classification',
    metric='log_loss',
    mode='min',
    k=20,
    threshold=0.5,   # use 1.0 for regression
    preset='balanced',
)
```

---

### KNORAE — K-Nearest Oracles (Eliminate)

```python
from ensemble_weights.des.knorae import KNORAE
```

Finds the largest neighborhood in which at least one model is competent on **every** neighbor (the intersection). If no model passes at K, shrinks to K-1 and retries — down to K=1, which always resolves. Surviving models share equal weight.

More aggressive than KNORAU. Tends to concentrate all weight on a single model. Performs well when one model genuinely dominates a tight local region; underperforms in noisy settings or when no model has clear regional dominance.

```python
router = KNORAE(
    task='classification',
    metric='log_loss',
    mode='min',
    k=20,
    threshold=0.5,   # use 1.0 for regression
    preset='balanced',
)
```

---

## Metrics

Pass a metric name as a string, or import the function directly.

```python
# String (resolved automatically)
router = KNNDWS(task='regression', metric='mae', mode='min')

# Function (import directly)
from ensemble_weights.metrics import mae
router = KNNDWS(task='regression', metric=mae, mode='min')

# Custom callable
router = KNNDWS(task='regression', metric=lambda y_true, y_pred: abs(y_true - y_pred) ** 0.5, mode='min')
```

### Scalar metrics — pass `predict()` output

| Name | Formula | `mode` |
|---|---|---|
| `mae` | `abs(y_true - y_pred)` | `'min'` |
| `mse` | `(y_true - y_pred) ** 2` | `'min'` |
| `rmse` | `sqrt((y_true - y_pred) ** 2)` | `'min'` |
| `accuracy` | `1.0 if correct else 0.0` | `'max'` |

### Probability metrics — pass `predict_proba()` output

| Name | Formula | `mode` | Notes |
|---|---|---|---|
| `log_loss` | `-log(p[y_true])` | `'min'` | Recommended for classification DES. Continuous signal — a model assigning 0.9 to the correct class scores much better than one assigning 0.51. |
| `prob_correct` | `p[y_true]` | `'max'` | Simpler alternative; linear rather than log-scaled. |

For probability metrics, pass 2D arrays of shape `(n_samples, n_classes)` in `preds_dict`. The library validates that the array dimensionality matches the metric at `fit()` time.

**Why `log_loss` for classification?** Without continuous per-sample signal, KNORAU and KNORAE collapse toward near-random behavior in settings where one model dominates. `log_loss` provides the gradient they need.

---

## Classification example

```python
from ensemble_weights.des.knndws import KNNDWS

# Models must support predict_proba
lr.fit(X_train, y_train)
knn.fit(X_train, y_train)
hgb.fit(X_train, y_train)

router = KNNDWS(task='classification', metric='log_loss', mode='min', k=20)
router.fit(
    X_val,
    y_val,
    {
        'lr':  lr.predict_proba(X_val),    # shape (n_val, n_classes)
        'knn': knn.predict_proba(X_val),
        'hgb': hgb.predict_proba(X_val),
    }
)

# Blend probability arrays, then argmax for hard predictions
weights = router.predict(X_test)
test_probas = {
    'lr':  lr.predict_proba(X_test),
    'knn': knn.predict_proba(X_test),
    'hgb': hgb.predict_proba(X_test),
}

import numpy as np
blended = np.array([
    sum(w[name] * test_probas[name][i] for name in w)
    for i, w in enumerate(weights)
])
predictions = blended.argmax(axis=1)
```

---

## Presets

Presets configure the neighbor search backend. The right choice depends on validation set size and dimensionality.

| Preset | Backend | Use when |
|---|---|---|
| `'exact'` | sklearn KNN | Val set < 10K samples, or < 20 features |
| `'balanced'` | FAISS IVF | 10K–100K samples, moderate dimensionality *(default)* |
| `'fast'` | FAISS IVF | Same range, willing to trade ~3% recall for speed |
| `'turbo'` | FAISS Flat | Large datasets where exact results still needed |
| `'high_dim_balanced'` | HNSW | > 100 features |
| `'high_dim_fast'` | HNSW | > 100 features, prioritise speed |

```python
# Print all presets with full parameters
from ensemble_weights import list_presets
list_presets()
```

### Custom preset

```python
router = KNNDWS(
    task='regression', metric='mae', mode='min', k=20,
    preset='custom', finder='faiss', index_type='ivf', n_probes=80,
)
```

### Auto-select based on data size

```python
from ensemble_weights import DynamicRouter

router = DynamicRouter.from_data_size(
    n_samples=50_000,
    n_features=12,
    task='regression',
    method='knn-dws',
    metric='mae',
    mode='min',
    k=20,
    n_queries=4_000,   # optional: test set size, used to weigh ANN fit cost
)
```

---

## Benchmarking across multiple seeds

For benchmarks, use `DynamicRouter` to select algorithms via string in a loop:

```python
from ensemble_weights import DynamicRouter

for method in ['knn-dws', 'ola', 'knora-u', 'knora-e']:
    router = DynamicRouter(
        task='classification',
        method=method,
        metric='log_loss',
        mode='min',
        k=20,
    )
    router.fit(X_val, y_val, val_probas)
    weights = router.predict(X_test)
```

See `tests/showcase.py` for a full benchmark across four datasets (California Housing, Bike Sharing, Letter Recognition, Phoneme) and `tests/multi_run.py` for 30-seed averaged results.

---

## Empirical results (30 seeds)

| Dataset | Best Single | Global Ensemble | KNNDWS | KNORAU |
|---|---|---|---|---|
| California Housing (MAE ↓) | 0.3452 | 0.3449 | **0.3414** | 0.3672 |
| Bike Sharing (MAE ↓) | 42.83 | 42.83 | **42.65** | 51.45 |
| Letter Recognition (Acc ↑) | 93.79% | 94.52% | 94.92% | **95.15%** |
| Phoneme (Acc ↑) | 86.64% | **87.27%** | 86.64% | 86.73% |

Key takeaways: **KNNDWS is the robust default** — it never performs worse than the best single model and consistently improves on it where local structure exists. **KNORAU is the specialist** — on multi-class classification with diverse models it can outperform KNNDWS. **Phoneme** illustrates the DES failure mode: when one model dominates globally and the feature space has no coherent regions of model diversity, a fixed global ensemble beats all routing algorithms.

---

## Package structure

```
ensemble_weights/
├── __init__.py          # Top-level imports: KNNDWS, OLA, KNORAU, KNORAE, DynamicRouter
├── metrics.py           # Named metric functions: mae, log_loss, etc.
├── neighbors.py         # Neighbor finder backends: KNN, FAISS, Annoy, HNSW
├── router.py            # DynamicRouter: string-based factory for benchmark loops
├── utils.py             # to_numpy, add_batch_dim
├── _config.py           # Internal: SPEED_PRESETS, make_finder, prep_fit_inputs
├── base/
│   ├── base.py          # BaseRouter abstract class
│   └── knnbase.py       # KNNBase: score matrix construction, shared fit()
└── des/
    ├── knndws.py        # KNNDWS
    ├── ola.py           # OLA
    ├── knorau.py        # KNORAU
    └── knorae.py        # KNORAE
```

Files with a leading underscore (`_config.py`) are internal implementation details and are not part of the public API.

---

## Algorithm selection guide

```
Do your models have distinct regional strengths?
├── No  → Use a fixed global ensemble (Nelder-Mead on val set). DES won't help.
└── Yes → What task?
    ├── Regression
    │   └── Use KNNDWS (threshold=0.5, temperature=0.1)
    │       KNORAU/KNORAE are not well-suited to regression — see note below.
    └── Classification
        ├── Start with KNNDWS (metric='log_loss', threshold=0.5, temperature=1.0)
        ├── Try KNORAU if you have many models with clear per-region dominance
        └── Avoid KNORAE unless neighborhoods are very clean — it's too aggressive
```

**Why KNORAU/KNORAE underperform on regression:** These algorithms apply a binary competence criterion per neighbor (above/below threshold). After per-neighbor normalization, the best model always maps to 1.0 and the worst to 0.0 — regardless of the actual gap between them. A model that is 70% worse than the best can score 0.52 after normalization and earn equal votes. Setting `threshold=1.0` partially recovers the oracle criterion, but the fundamental mismatch between binary voting and continuous error remains.

---

## Extending the library

### Custom metric

Any callable with signature `(y_true, y_pred) -> float` works:

```python
import numpy as np

def huber(y_true, y_pred, delta=1.0):
    r = abs(y_true - y_pred)
    return r if r <= delta else delta * r - 0.5 * delta ** 2

router = KNNDWS(task='regression', metric=huber, mode='min', k=20)
```

### Custom neighbor finder

Subclass `NeighborFinder` from `neighbors.py` and implement `fit(X)` and `kneighbors(X, k=None)`, then pass it via `preset='custom'`:

```python
router = KNNDWS(
    task='regression', metric='mae', mode='min', k=20,
    preset='custom', finder='knn',   # or swap in your own finder via _config.make_finder
)
```

### New algorithm

Subclass `KNNBase` from `base/knnbase.py`, implement `fit()` (call `prep_fit_inputs` then `super().fit()`) and `predict()`. The score matrix `self.matrix` is always shape `(n_val, n_models)` with higher-is-better scores. See `des/knndws.py` as the reference implementation.