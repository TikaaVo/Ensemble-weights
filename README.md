# ensemble-weights

A Python library for **Dynamic Ensemble Selection (DES)** — per-sample adaptive routing across a pool of pre-trained models, using a held-out validation set to determine which model (or blend of models) performs best in the local neighborhood of each test point.

Unlike fixed ensembles that apply the same weights everywhere, DES assigns different weights to different inputs based on where each model tends to be accurate. A linear model might dominate in low-variance regions while a boosting model wins on complex interactions — DES finds and exploits this automatically.

---

## Installation

```bash
pip install ensemble-weights
```

**Optional dependencies** (install only what you need):

```bash
pip install faiss-cpu   # for 'balanced', 'fast', and 'turbo' presets
pip install hnswlib     # for 'high_dim_balanced' and 'high_dim_fast' presets
```

The `exact` preset requires only scikit-learn, which is always installed.

---

## Quick Start

```python
from ensemble_weights import DynamicRouter

# 1. Train your models

# 2. Get validation predictions from each model and organize them into a dictionary
# This allows it to be compatible with any ML library, as long as you can organize
# the validation predictions as follows:
val_preds = {
    'linear':  linear_model.predict(X_val),
    'knn':     knn_model.predict(X_val),
    'boosting': boost_model.predict(X_val),
}

# 3. Fit the router on the validation set
router = DynamicRouter(
    task='regression',
    dtype='tabular',
    method='knn-dw',   # soft per-sample blending
    metric='mae',
    mode='min',
    k=20,
    preset='exact',
)
router.fit(X_val, y_val, val_preds)

# 4. Get per-sample weights at inference time
test_preds = {
    'linear':  linear_model.predict(X_test),
    'knn':     knn_model.predict(X_test),
    'boosting': boost_model.predict(X_test),
}

weights = router.predict(X_test)  # list of {model_name: weight} dicts

# 5. Blend predictions using the returned weights
import numpy as np

results = []
for i, w in enumerate(weights):
    blended = sum(w[name] * test_preds[name][i] for name in w)
    results.append(blended)
predictions = np.array(results)
```

---

## How It Works

At fit time, the router builds a neighbor index over the validation set features and records each model's per-sample score in a score matrix `(n_val, n_models)`.

At predict time, for each test point:

1. Retrieve its K nearest neighbors from the validation set.
2. Average each model's scores across those K neighbors.
3. Normalize scores within the neighborhood so the best model = 1.0, worst = 0.0.
4. Apply a **competence gate** — zero out any model below a threshold of the local best.
5. Run softmax with a temperature parameter to produce final blend weights.

This means that in regions where one model clearly dominates, it receives most or all of the weight. In genuinely ambiguous regions, weights are spread more evenly.

---

## Methods

### `knn-dw` — Distance-Weighted Blending

Soft blending using per-neighborhood softmax weights. Recommended for most use cases. The competence gate and temperature parameter control how aggressively it routes toward the single best local model.

### `ola` — Overall Local Accuracy

Hard selection: always assigns weight 1.0 to the single model with the highest average score in the local neighborhood. Equivalent to `knn-dw` with `temperature → 0` and `competence_threshold=1.0`. Use when you want pure model selection with no blending.

---

## DynamicRouter API

```python
DynamicRouter(
    task,                    # 'regression' or 'classification'
    dtype,                   # 'tabular' or 'image'
    method='knn-dw',         # 'knn-dw' or 'ola'
    metric='accuracy',       # see Metrics section below
    mode='max',              # 'max' if higher score = better, 'min' if lower
    preset='balanced',       # see Presets section below
    k=10,                    # number of neighbors per query
    competence_threshold=0.5,# knn-dw only; see Tuning section below
    feature_extractor=None,  # optional callable applied before neighbor search
    finder=None,             # required only with preset='custom'
    **kwargs,                # forwarded to the neighbor finder
)
```

### `.fit(features, y, preds_dict)`

Fits the routing model on validation data.

| Parameter | Description |
|---|---|
| `features` | `(n_val, n_features)` — validation set features |
| `y` | `(n_val,)` — validation ground-truth labels or values |
| `preds_dict` | `dict[str, array]` — validation predictions keyed by model name |

**Important:** `features` must be from a held-out validation set that was not used to train the base models. Using training data here will cause the router to overfit to in-sample performance.

### `.predict(x, temperature=None)`

Returns per-sample model weights.

| Parameter | Description |
|---|---|
| `x` | `(n_features,)` or `(batch_size, n_features)` |
| `temperature` | Softmax sharpness for `knn-dw`. `None` uses auto-default (0.1 for `mode='min'`, 1.0 for `mode='max'`) |

**Returns:** A single `{model_name: weight}` dict for one sample, or a list of such dicts for a batch.

---

## Metrics

Pass a string name or any callable with signature `(y_true, y_pred) -> float`.

| Name | Formula | Use with `mode` |
|---|---|---|
| `'accuracy'` | `1 if y_true == y_pred else 0` | `'max'` |
| `'mae'` | `abs(y_true - y_pred)` | `'min'` |
| `'mse'` | `(y_true - y_pred) ** 2` | `'min'` |
| `'rmse'` | `sqrt((y_true - y_pred) ** 2)` | `'min'` |
| Custom | Any `(y_true, y_pred) -> float` | depends |

---

## Presets

Presets configure the neighbor search backend. Higher-numbered presets are faster but may require additional dependencies.

| Preset | Backend | Notes |
|---|---|---|
| `'exact'` | sklearn KNN | 100% accurate, no extra dependencies. Best for small datasets or low-dimensional data. |
| `'balanced'` | FAISS IVF | ~98% recall. Good default for medium datasets (10K–100K val samples). Requires `faiss-cpu`. |
| `'fast'` | FAISS IVF | ~95% recall. Faster queries than `'balanced'`, slightly lower recall. Requires `faiss-cpu`. |
| `'turbo'` | FAISS flat | Exact results with C++/SIMD speed. Best for large datasets when accuracy matters. Requires `faiss-cpu`. |
| `'high_dim_balanced'` | HNSW (hnswlib) | Best for >100D feature spaces, balanced. Requires `hnswlib`. |
| `'high_dim_fast'` | HNSW (hnswlib) | Best for >100D feature spaces, faster. Requires `hnswlib`. |
| `'custom'` | Your choice | Specify `finder='knn'/'faiss'/'annoy'/'hnsw'` and pass finder kwargs directly. |

> **Note on Annoy:** `AnnoyNeighborFinder` is available via `preset='custom', finder='annoy'` but has a known bug on Apple Silicon (M1/M2/M3) where it returns only 1 neighbor regardless of settings. Use FAISS presets on macOS ARM64.

### Auto-selection

```python
router = DynamicRouter.from_data_size(
    n_samples=50_000,   # validation set size
    n_features=15,
    n_queries=1_000,    # optional: expected test set size; influences fit/predict tradeoff
    task='regression',
    dtype='tabular',
)
```

`from_data_size` selects a preset based on dataset dimensions and optionally the expected query volume — ANN methods have higher fit cost but lower per-query cost, so if `n_queries` is small relative to `n_samples`, exact search may be faster overall.

```python
DynamicRouter.list_presets()  # print all presets with descriptions
```

---

## Tuning

### `competence_threshold` (knn-dw only)

After per-neighborhood normalization, any model scoring below this fraction of the local best is excluded from the softmax blend entirely.

| Value | Behavior |
|---|---|
| `0.0` | No gate — all models always contribute |
| `0.5` | Only models within 50% of the local best contribute (default) |
| `1.0` | Equivalent to OLA — only the single best model contributes |

A value of `0.5` works well in practice. Lower values allow more blending; higher values produce harder routing.

### `temperature`

Controls softmax sharpness in `knn-dw`. Passed to `.predict()`, not the constructor.

| Value | Behavior |
|---|---|
| `0.1` | Near-hard routing; soft only when models are genuinely tied. **Recommended for regression.** |
| `1.0` | Moderate blending. **Recommended for classification.** |
| `> 1.0` | Increasingly uniform weights regardless of local performance |

When `temperature=None` (default), it is set automatically based on `mode`: `0.1` for `'min'` metrics (regression), `1.0` for `'max'` metrics (classification).

### `k`

Number of validation neighbors per query. More neighbors = smoother, more stable local estimates, but less sensitivity to sharp regional boundaries. `k=20` is a reasonable default; consider `k=10–15` for smaller validation sets (<2K samples).

---

## Feature Scaling

The neighbor search operates in raw feature space. If your features have different scales, standardize them before passing to the router — otherwise high-magnitude features will dominate distance calculations and neighborhoods will be meaningless.

```python
from sklearn.preprocessing import StandardScaler

# Fit scaler on validation set only — not train or test
scaler = StandardScaler().fit(X_val)
X_val_scaled  = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

router.fit(X_val_scaled, y_val, val_preds)
weights = router.predict(X_test_scaled)
```

This does not apply to the base models themselves, only to the features passed to the router.

---

## When DES Works Well

DES provides the most benefit when:

- **Models have orthogonal failure modes** — e.g. a linear model dominates in smooth regions while KNN dominates in dense clusters. If all models fail on the same samples, there is nothing to route between.
- **The validation set is large enough** — local estimates from K neighbors are noisy. At least ~1,000 val samples is recommended; very small val sets (< 300) produce unreliable local scores.
- **Input space has regional structure** — distinct regimes where different models genuinely perform differently (geographic clusters, temporal patterns, distinct subpopulations).

DES will match or slightly underperform a fixed global ensemble when one model dominates everywhere, or when val set size is too small to estimate local performance reliably.

---

## Custom Neighbor Finders

Use `preset='custom'` to configure a finder directly:

```python
# Custom FAISS IVF with specific parameters
router = DynamicRouter(
    task='regression', dtype='tabular', method='knn-dw',
    metric='mae', mode='min', k=20,
    preset='custom', finder='faiss',
    index_type='ivf', n_cells=64, n_probes=20,
)

# Custom HNSW with nmslib backend
router = DynamicRouter(
    task='regression', dtype='tabular', method='knn-dw',
    metric='mae', mode='min', k=20,
    preset='custom', finder='hnsw',
    backend='nmslib', M=32, ef_construction=400, ef_search=200,
)
```

Or instantiate a finder directly:

```python
from ensemble_weights.models.neighbors import FaissNeighborFinder
from ensemble_weights.models.knn import KNNModel

finder = FaissNeighborFinder(k=20, index_type='ivf', n_probes=50)
model = KNNModel(metric=lambda y, p: abs(y - p), mode='min',
                 neighbor_finder=finder, competence_threshold=0.5)
model.fit(X_val, y_val, val_preds)
```

---

## Image Tasks

For image inputs, pass a `feature_extractor` to map raw images to a meaningful embedding space before neighbor search:

```python
import torchvision.models as models
import torch

resnet = models.resnet50(pretrained=True)
extractor = lambda x: resnet(torch.tensor(x)).detach().numpy()

router = DynamicRouter(
    task='classification', dtype='image', method='knn-dw',
    metric='accuracy', mode='max', k=10,
    preset='high_dim_balanced',
    feature_extractor=extractor,
)
```

---

## License

MIT