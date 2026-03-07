# HNSW

This backend uses [Hierarchical Navigable Small World](https://github.com/nmslib/hnswlib ) graphs for approximate nearest neighbour
search. It is the recommended choice for high-dimensional data.

---

## When to use

- The data has a large number of features, as HNSW handles high-dimensional spaces
better than FAISS IVF, which degrades as dimensionality increases
- It performs best when a good balance of build time, query speed, and recall on large datasets is needed
- It performs worst on small datasets where the graph overhead is not justified and
`preset="exact"` would be simpler

---

## How it works

At fit time, HNSW builds a multi-layer graph where each node is a validation sample.
`M` controls how many edges each node has — higher values produce a denser graph with
better recall at the cost of more memory and longer build times. `ef_construction`
controls the search width during graph construction, where higher values produce a higher
quality graph. At query time, `ef_search` controls the search width, where higher values
improve recall at the cost of query speed.

Two backends are supported: `hnswlib` (default) and `nmslib`.

---

## Presets

| Preset | M | ef_construction | ef_search |
|---|---|---|---|
| `high_dim_balanced` | 32 | 400 | 200 |
| `high_dim_fast` | 16 | 200 | 100 |

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | int | 10 | Number of neighbours |
| `space` | str | `"l2"` | Distance metric: `"l2"`, `"cosine"`, or `"ip"` |
| `M` | int | 32 | Number of graph connections per node |
| `ef_construction` | int | 400 | Search width during graph construction |
| `ef_search` | int | 200 | Search width at query time |
| `backend` | str | `"hnswlib"` | Underlying library: `"hnswlib"` or `"nmslib"` |

---

## Dependencies

`hnswlib>=0.7`

---

## Example

```python
from deskit.des.knorau import KNORAU

# Using a preset
router = KNORAU(task="classification", metric="accuracy", mode="max", k=20,
                preset="high_dim_balanced")
router.fit(X_val, y_val, val_preds)
weights = router.predict(x)

# Custom configuration
router = KNORAU(task="classification", metric="accuracy", mode="max", k=20,
                preset="custom", finder="hnsw", M=48, ef_construction=600, ef_search=300)
router.fit(X_val, y_val, val_preds)
weights = router.predict(x)
```

---

## Notes

For datasets with a large number of samples, `ef_construction` below 300 may produce a
low-quality graph — a warning will be emitted in that case.

`nmslib` is an alternative backend to `hnswlib` that supports additional distance spaces.
It requires a separate install: `pip install nmslib`.