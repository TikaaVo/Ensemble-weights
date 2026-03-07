# deskit

deskit is a flexible, light, and easy-to-use ensembling library that implements
Dynamic Ensemble Selection (DES) algorithms for ensembling multiple ML models
on a singular dataset.

The library works entirely with data, taking as input a validation dataset
along with pre-computed predictions and outputting a dictionary of weights
per model. This means that it can be used with any library or model without
requiring any wrappers, including custom models, popular ML libraries, and APIs.

deskit contains multiple different DES algorithms, and it works with both classification
and regression.

---

## Dynamic Ensemble Selection

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

## Algorithms

deskit implements five DES algorithms. Each algorithm page covers how it works, when to use it,
and all accepted parameters.

- [KNN-DWS](algorithms/knndws.md) — soft blending via distance-weighted softmax. General recommendation for regression.
- [KNORA-U](algorithms/knorau.md) — vote-count weighting. Safe default for classification.
- [KNORA-E](algorithms/knorae.md) — intersection-based. Best when models have clear regional dominance.
- [KNORA-IU](algorithms/knoraiu.md) — like KNORA-U with inverse-distance weighted votes.
- [OLA](algorithms/ola.md) — single model selection. Useful as a diagnostic tool.

---

## ANN Backends

deskit supports four nearest neighbour backends. Each backend page covers when to use it,
available presets, and all configuration parameters.

- [Exact KNN](backends/knn.md) — exact search via sklearn. No extra dependencies.
- [FAISS](backends/faiss.md) — recommended default for most datasets.
- [HNSW](backends/hnsw.md) — best for high-dimensional data.
- [Annoy](backends/annoy.md) — memory-efficient, supports persisting the index to disk.

---

## Benchmark

[Benchmark results](benchmark.md) across seven datasets comparing deskit algorithms against
simple averaging and the best single model in the pool.