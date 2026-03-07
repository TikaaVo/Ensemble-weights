# Exact KNN

This backend uses sklearn's [NearestNeighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) to perform exact nearest neighbour search.
It is the simplest backend and requires scikit-learn.

---

## When to use

- The validation set is small and speed is not a concern
- It performs best when exact results must be guaranteed and cannot afford any recall loss
- It performs worst on large datasets where the linear scan becomes a bottleneck

---

## How it works

At fit time, sklearn's `NearestNeighbors` is fitted on the validation features.
At query time, it performs a brute-force linear scan over all validation samples to find
the exact k nearest neighbours.

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | int | 10 | Number of neighbours |

---

## Dependencies

`scikit-learn>=1.0`

---

## Example

```python
from deskit.des.knorau import KNORAU

router = KNORAU(task="classification", metric="accuracy", mode="max", k=20, preset="exact")
router.fit(X_val, y_val, val_preds)
weights = router.predict(x)
```

---