# Annoy

This backend uses Spotify's [Annoy](https://github.com/spotify/annoy) library for approximate nearest neighbour search.
It is memory-efficient and supports persisting the index to disk, making it useful
when the index needs to be saved and reloaded across sessions.

---

## When to use

- Memory efficiency is a priority
- It performs best when you need to persist the index to disk and reload it without rebuilding
- It performs worst on very low-dimensional data where the
tree structure can degenerate

---

## How it works

At fit time, Annoy builds a forest of `n_trees` random projection trees over the
validation features. More trees produce better recall at the cost of more memory and
longer build times. At query time, `search_k` controls how many nodes are visited
across all trees — higher values improve recall at the cost of slower queries. By
default, `search_k` is set to `n_trees * k`.

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | int | 10 | Number of neighbours |
| `n_trees` | int | 100 | Number of trees in the forest. Higher = better recall, more memory |
| `metric` | str | `"euclidean"` | Distance metric: `"euclidean"`, `"angular"`, `"manhattan"`, `"hamming"`, `"dot"` |
| `search_k` | int | `n_trees * k` | Nodes visited per query. Higher = better recall, slower queries |

---

## Dependencies

`annoy>=1.17`

---

## Example

```python
from deskit.des.knorau import KNORAU

router = KNORAU(task="classification", metric="accuracy", mode="max", k=20,
                preset="custom", finder="annoy", n_trees=100)
router.fit(X_val, y_val, val_preds)
weights = router.predict(x)
```

---

## Notes

Annoy has a known bug on Apple Silicon (M1/M2/M3) where the index silently returns
only 1 neighbour regardless of `k`. deskit detects this at fit time and raises a
`RuntimeError` with instructions to switch to `preset="fast"` or `preset="exact"` instead.

Annoy does not have a built-in preset, and it is only available via `preset="custom"` with
`finder="annoy"`.