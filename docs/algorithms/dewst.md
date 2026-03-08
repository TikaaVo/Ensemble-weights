# DEWS-T

This algorithm is designed to be consistent and flexible for both classification and
regression tasks and is a variation of DEWS-I that fits a weighted trend line over
each model's scores across the K neighbours, extrapolating to estimate competence at
the test point itself rather than averaging.
It uses soft blending between the top experts in a certain competence region to compute a set of weights for the models.

---

## When to use

- DEWS-T is currently the general recommendation when you want the best single deskit algorithm across both tasks.
It works best with soft metrics, so it works for regression classification with confidence scores, but not as well with 
hard predictions
- It performs best when competence regions have a smooth, directional structure — i.e. model quality
  changes monotonically with distance from the test point
- It performs worst for homogeneous datasets, noisy neighbourhoods, and classification with hard predictions
- When no significant trend is detected (weighted R² below threshold), it falls back to DEWS-I automatically,
  so it never performs worse than DEWS-I in expectation

---

## How it works

When `fit` is called, DEWS-T fits a KNN algorithm on the validation data and builds a criterion score matrix.
For MAE and MSE, signed residuals are stored instead of raw metric values so that
directional information is preserved across neighbors.

When `predict` is called, it finds the K nearest neighbors from the test point. For each model, it fits
a weighted least squares line over the K neighbors, using inverse-distance weights so closer neighbors
pull the fit more strongly. The line is extrapolated to distance = 0 to estimate the model's competence
at the test point.

The quality of each trend line is evaluated using weighted R². If R² is below `r2_threshold`, that model
falls back to DEWS-I. This fallback happens per model per sample, 
so some models may use the trend while others fall back on the same test point.

The resulting scores are normalised using min-max normalisation and models below `threshold` are removed.
Finally, the remaining models are blended using softmax with temperature.

---

## Parameters

| Parameter       | Type            | Default                               | Description                                                                                                                                                     |
|-----------------|-----------------|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `task`          | str             | —                                     | `"classification"` or `"regression"`                                                                                                                            |
| `metric`        | str or callable | —                                     | Scoring function per sample. Built-ins: `accuracy`, `mae`, `mse`, `rmse`, `log_loss`, `prob_correct`. Custom callables `(y_true, y_pred) -> float` are accepted |
| `mode`          | str             | —                                     | `"max"` if higher is better, `"min"` if lower                                                                                                                   |
| `k`             | int             | 10                                    | Number of neighbours                                                                                                                                            |
| `threshold`     | float           | 0.5                                   | Competence cutoff                                                                                                                                               |
| `temperature`   | float           | 0.1/1.0 for regression/classification | Defines how smooth the model blend is                                                                                                                           |
| `r2_threshold`  | float           | 0.7                                   | Minimum weighted R² for the trend line to be trusted                                                      |
| `preset`        | str             | `"balanced"`                          | ANN backend preset                                                                                                                                              |
| `finder`        | str             | —, optional                           | Only if the preset is `"custom"`; Options: `"knn"`, `"faiss"`, `"annoy"`, `"hnsw"`                                                                              |

---

## Example
```python
# Regression
from deskit.des.dewst import DEWST

router = DEWST(task="regression", metric="mae", mode="min", k=20)
router.fit(X_val, y_val, val_preds)
weights = router.predict(x)
```

```python
# Classification
from deskit.des.dewst import DEWST

router = DEWST(task="classification", metric="log_loss", mode="min", k=20)
router.fit(X_val, y_val, val_preds)
weights = router.predict(x)
```

---

## Notes

A lower temperature is recommended for regression because regression metrics tend to produce scores on a
continuous scale where differences can be large, so a low temperature sharpens the softmax to reflect that.
In contrast, classification metrics tend to produce scores that are closer together, so a higher temperature
keeps the blend soft.

The `r2_threshold` controls how often the trend line is trusted over the DEWS-I fallback. A higher value
means the algorithm is more conservative and only extrapolates when the evidence is strong, converging toward
DEWS-I behavior on noisy datasets. A value of 0.7 is recommended as it filters out weak trends while
still exploiting genuine directional structure when present.