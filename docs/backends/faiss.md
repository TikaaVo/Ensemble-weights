# FAISS

This backend uses Facebook's [FAISS](https://faiss.ai) library for approximate nearest neighbour search.
It is the recommended default for most datasets and ships with three presets.

---

## When to use

- Use as the default choice for most low-to-medium dimensional datasets
- It performs best when datasets are large and an accuracy loss is worth the reduced computation
- It performs worst on very high-dimensional data

---

## How it works

At fit time, FAISS builds an index over the validation features. The index type determines
the tradeoff between speed and recall. `flat` performs exact search via FAISS's optimised
L2 scan. `ivf` (Inverted File Index) partitions the dataset into cells using k-means, then
at query time searches only a subset of cells determined by `n_probes`, where more probes means
higher recall but slower queries. `n_cells` is set automatically to `sqrt(n_samples)` if
not specified, capped at 4096.

---

## Presets

| Preset | index_type | n_probes | Recall |
|---|---|---|---|
| `balanced` | `ivf` | 50 | ~98% |
| `fast` | `ivf` | 30 | ~95% |
| `turbo` | `flat` | — | 100% |

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | int | 10 | Number of neighbours |
| `index_type` | str | `"flat"` | Index type: `"flat"`, `"ivf"`, or `"hnsw"` |
| `n_cells` | int | `sqrt(n_samples)` | Number of IVF cells. Only used when `index_type="ivf"` |
| `n_probes` | int | 50 | Number of cells searched per query. Only used when `index_type="ivf"` |
| `hnsw_M` | int | 32 | HNSW graph connections per node. Only used when `index_type="hnsw"` |
| `hnsw_efConstruction` | int | 400 | HNSW build-time search width. Only used when `index_type="hnsw"` |
| `hnsw_efSearch` | int | 200 | HNSW query-time search width. Only used when `index_type="hnsw"` |

---

## Dependencies

`faiss-cpu>=1.7`

---

## Example

```python
from deskit.des.knorau import KNORAU

# Using a preset
router = KNORAU(task="classification", metric="accuracy", mode="max", k=20, preset="balanced")
router.fit(X_val, y_val, val_preds)
weights = router.predict(x)

# Custom IVF configuration
router = KNORAU(task="classification", metric="accuracy", mode="max", k=20,
                preset="custom", finder="faiss", index_type="ivf", n_probes=80)
router.fit(X_val, y_val, val_preds)
weights = router.predict(x)
```

---

## Notes

FAISS IVF requires a minimum of 40 samples per cell to train. If the validation set is too
small, the library will automatically reduce `n_cells` and emit a warning.

FAISS Flat may have floating-point precision issues on data with 2 or fewer feature; a
warning will be emitted in that case.