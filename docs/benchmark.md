# Benchmark

All results are from a 100-seed benchmark (seeds 0–99) on standard datasets. Each seed
produces a different random train/validation/test split, and results are averaged across
all 20 seeds to reduce variance from any single split.

---

## Methodology

**Split:** Each dataset is split into train, validation, and test sets. Models are trained
on the train set, the DES router is fitted on the validation set, and it is evaluated on the test set.

**Best Single** is the best individual model from the pool, selected by validation
set performance. It represents the baseline without any ensembling.

**Simple Average** is a uniform blend of all five models with no fitting or
tuning. It represents the simplest possible ensemble baseline.

All deskit algorithms use `preset="balanced"` (FAISS IVF) and `k=20`.

---

## Pool

The same five-model pool is used across all datasets:

| Model |
|---|
| K-Nearest Neighbors |
| Decision Tree |
| SVR / SVM-RBF |
| Ridge / Gaussian NB |
| Bayesian Ridge / Logistic Regression |

These five were chosen for having different inductive biases and architectures, which is the kind of scenario that
DES would be used in.

---

## Datasets

### California Housing
**Source:** sklearn built-in. **Size:** 20,640 samples, 8 features.

Predict median house value from census block features.

### Bike Sharing
**Source:** [OpenML 42712](https://www.openml.org/d/42712). **Size:** 17,379 samples, 12 features.

Predict hourly bike rental counts from weather and time features.

### Abalone
**Source:** [OpenML 183](https://www.openml.org/d/183). **Size:** 4,177 samples, 8 features.

Predict abalone age from physical measurements.

### Diabetes
**Source:** sklearn built-in. **Size:** 442 samples, 10 features.

Predict disease progression one year after baseline from physiological measurements.

### Concrete Strength
**Source:** [OpenML 4353](https://www.openml.org/d/4353). **Size:** 1,030 samples, 8 features.

Predict concrete compressive strength from ingredient and curing age features.

### HAR
**Source:** [OpenML 1478](https://www.openml.org/d/1478). **Size:** 10,299 samples, 561 features.

Six-class classification of human activities from smartphone accelerometer and gyroscope data.

### Yeast
**Source:** [OpenML 181](https://www.openml.org/d/181). **Size:** 1,484 samples, 8 features.

Ten-class protein localisation classification with class imbalance.

### Image Segment
**Source:** [OpenML 36](https://www.openml.org/d/36). **Size:** 2,310 samples, 19 features.

Seven-class classification of outdoor image segments from colour and texture statistics.

### Vowel
**Source:** [OpenML 307](https://www.openml.org/d/307). **Size:** 990 samples, 10 features.

Eleven-class vowel recognition from LPC-derived formant frequencies.

### Waveform
**Source:** [OpenML 60](https://www.openml.org/d/60). **Size:** 5,000 samples, 40 features.

Three-class classification of artificially constructed waveforms with deliberate class
overlap.

---

## Regression results

MAE, lower is better. % shown as delta vs Best Single. 100-seed mean ± std.

| Dataset | Best Single        | Simple Avg | KNN-DWS | KNN-DWS-I | OLA | KNORA-U | KNORA-E | KNORA-IU |
|---|--------------------|---|---|---|---|---|---|---|
| California Housing | 0.3955 ± 0.008     | +7.93% | −2.41% | **−2.68%** | −0.31% | −0.81% | +7.22% | −1.03% |
| Bike Sharing | 51.604 ± 1.291     | +48.39% | −4.90% | **−6.25%** | −2.55% | +6.67% | +15.16% | +5.50% |
| Abalone | **1.4923 ± 0.054** | +1.29% | +3.00% | +3.12% | +4.02% | +1.63% | +7.60% | +1.61% |
| Diabetes | **44.986 ± 3.370** | +2.98% | +0.96% | +0.88% | +3.16% | +5.13% | +14.36% | +5.02% |
| Concrete Strength | 5.3934 ± 0.400     | +21.30% | +1.40% | −1.55% | +3.94% | +0.46% | +11.29% | **−2.85%** |

---

KNORA variants are designed for classification, which explains the poor performance
on regression datasets; However, some exception can occur in certain datasets, either where
feature space is has hard clusters (like in Concrete Strength) or when the target is discrete
and classification-like (like in Abalone).

## Classification results

Accuracy, higher is better. % shown as delta vs Best Single. 100-seed mean ± std.
Classification datasets include a comparison against [DESlib](https://github.com/scikit-learn-contrib/DESlib),
a mature sklearn-compatible DES library.

### HAR

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 98.24% | 0.25% | — |
| Simple Average | 97.93% | 0.33% | −0.32% |
| deskit KNN-DWS | 98.37% | 0.27% | +0.13% |
| deskit KNN-DWS-I | **98.37%** | 0.27% | **+0.14%** |
| deskit OLA | 97.99% | 0.31% | −0.25% |
| deskit KNORA-U | 98.18% | 0.29% | −0.05% |
| deskit KNORA-E | 97.99% | 0.31% | −0.25% |
| deskit KNORA-IU | 98.19% | 0.29% | −0.04% |
| DESlib KNORA-U | 98.04% | 0.32% | −0.20% |
| DESlib KNORA-E | 97.83% | 0.34% | −0.41% |
| DESlib OLA | 97.05% | 0.44% | −1.20% |
| DESlib META-DES | **98.37%** | 0.31% | **+0.14%** |
| DESlib KNOP | 98.32% | 0.30% | +0.08% |
| DESlib DESP | 97.98% | 0.33% | −0.26% |
| DESlib DESKNN | 97.81% | 0.33% | −0.43% |

deskit achieves a best mean score of 98.37%; DESlib achieves a best mean score of 98.37%.

### Yeast

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 59.19% | 2.70% | — |
| Simple Average | 59.46% | 2.44% | +0.46% |
| deskit KNN-DWS | 59.89% | 2.42% | +1.18% |
| deskit KNN-DWS-I | 59.91% | 2.51% | +1.23% |
| deskit OLA | 58.93% | 2.37% | −0.44% |
| deskit KNORA-U | 59.89% | 2.53% | +1.18% |
| deskit KNORA-E | 57.05% | 2.63% | −3.61% |
| deskit KNORA-IU | **60.06%** | 2.53% | **+1.48%** |
| DESlib KNORA-U | 59.91% | 2.32% | +1.22% |
| DESlib KNORA-E | 57.64% | 2.49% | −2.61% |
| DESlib OLA | 57.46% | 2.44% | −2.91% |
| DESlib META-DES | 58.28% | 2.61% | −1.52% |
| DESlib KNOP | 59.88% | 2.40% | +1.17% |
| DESlib DESP | 59.48% | 2.23% | +0.50% |
| DESlib DESKNN | 58.19% | 2.11% | −1.69% |

deskit achieves a best mean score of 60.06%; DESlib achieves a best mean score of 59.91%.

### Image Segment

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 93.65% | 1.11% | — |
| Simple Average | 95.24% | 1.04% | +1.70% |
| deskit KNN-DWS | 95.56% | 0.94% | +2.04% |
| deskit KNN-DWS-I | 95.71% | 0.96% | +2.20% |
| deskit OLA | 94.96% | 0.89% | +1.39% |
| deskit KNORA-U | 95.60% | 1.02% | +2.08% |
| deskit KNORA-E | 95.66% | 0.95% | +2.14% |
| deskit KNORA-IU | **95.84%** | 0.98% | **+2.33%** |
| DESlib KNORA-U | 95.10% | 1.05% | +1.54% |
| DESlib KNORA-E | 95.45% | 0.89% | +1.91% |
| DESlib OLA | 94.96% | 0.95% | +1.40% |
| DESlib META-DES | 95.61% | 0.91% | +2.09% |
| DESlib KNOP | 95.34% | 0.96% | +1.80% |
| DESlib DESP | 94.89% | 1.05% | +1.32% |
| DESlib DESKNN | 94.82% | 0.98% | +1.25% |

deskit achieves a best mean score of 95.84%; DESlib achieves a best mean score of 95.61%.

### Vowel

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 90.54% | 2.17% | — |
| Simple Average | 88.90% | 2.40% | −1.81% |
| deskit KNN-DWS | 90.13% | 2.27% | −0.46% |
| deskit KNN-DWS-I | 90.48% | 2.26% | −0.07% |
| deskit OLA | 90.36% | 2.32% | −0.20% |
| deskit KNORA-U | 90.76% | 2.16% | +0.25% |
| deskit KNORA-E | 90.92% | 2.12% | +0.42% |
| deskit KNORA-IU | **91.38%** | 2.05% | **+0.93%** |
| DESlib KNORA-U | 88.76% | 2.31% | −1.96% |
| DESlib KNORA-E | 89.69% | 2.11% | −0.94% |
| DESlib OLA | 88.30% | 2.71% | −2.48% |
| DESlib META-DES | 90.09% | 2.16% | −0.50% |
| DESlib KNOP | 89.27% | 2.30% | −1.40% |
| DESlib DESP | 86.13% | 2.52% | −4.88% |
| DESlib DESKNN | 85.37% | 2.94% | −5.71% |

deskit achieves a best mean score of 91.38%; DESlib achieves a best mean score of 90.09%.

### Waveform

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | **86.28%** | 1.10% | — |
| Simple Average | 85.38% | 1.02% | −1.04% |
| deskit KNN-DWS | 85.80% | 1.04% | −0.56% |
| deskit KNN-DWS-I | 85.80% | 1.02% | −0.55% |
| deskit OLA | 84.03% | 1.10% | −2.61% |
| deskit KNORA-U | 85.60% | 1.02% | −0.78% |
| deskit KNORA-E | 82.84% | 1.13% | −3.99% |
| deskit KNORA-IU | 85.62% | 1.02% | −0.77% |
| DESlib KNORA-U | 85.83% | 1.03% | −0.53% |
| DESlib KNORA-E | 83.19% | 1.14% | −3.57% |
| DESlib OLA | 81.19% | 1.29% | −5.90% |
| DESlib META-DES | 85.29% | 1.11% | −1.15% |
| DESlib KNOP | **86.10%** | 1.08% | **−0.21%** |
| DESlib DESP | 85.78% | 1.03% | −0.57% |
| DESlib DESKNN | 84.61% | 1.18% | −1.93% |

deskit achieves a best mean score of 85.80%; DESlib achieves a best mean score of 86.10%.

---

## Timing

Mean fit + predict time in milliseconds, averaged across 100 seeds. Fit is measured once
per dataset per seed; predict is measured over the full test set.

deskit caches all model predictions on the validation set at fit time and reads from that
matrix at inference, so no model is called at predict time. This is the primary reason for
the speed advantage over DESlib, which calls each model live per neighbour at inference.

deskit used `preset='balanced'`, which uses FAISS IVF instead of KNN, but the difference
in performance isn't very pronounced in datasets of the size used.

### deskit

| Dataset | KNN-DWS | KNN-DWS-I | OLA | KNORA-U | KNORA-E | KNORA-IU |
|---|---|---|---|---|---|---|
| California Housing | 24.4 ms | 23.9 ms | 22.9 ms | 26.2 ms | 34.5 ms | 27.9 ms |
| Bike Sharing | 19.8 ms | 19.3 ms | 18.5 ms | 21.4 ms | 28.4 ms | 22.9 ms |
| Abalone | 5.1 ms | 5.0 ms | 4.7 ms | 5.3 ms | 7.1 ms | 5.7 ms |
| Diabetes | 1.4 ms | 1.4 ms | 1.2 ms | 1.3 ms | 1.6 ms | 1.3 ms |
| Concrete Strength | 1.7 ms | 1.7 ms | 1.6 ms | 1.8 ms | 2.2 ms | 1.8 ms |
| HAR | 66.9 ms | 55.5 ms | 55.0 ms | 55.9 ms | 60.8 ms | 57.9 ms |
| Yeast | 3.5 ms | 3.2 ms | 3.0 ms | 2.8 ms | 3.1 ms | 3.0 ms |
| Image Segment | 5.8 ms | 5.5 ms | 5.3 ms | 5.1 ms | 5.4 ms | 5.3 ms |
| Vowel | 3.5 ms | 3.4 ms | 3.2 ms | 3.1 ms | 3.3 ms | 3.1 ms |
| Waveform | 10.5 ms | 9.8 ms | 9.6 ms | 9.1 ms | 9.9 ms | 9.8 ms |

### DESlib (classification datasets only)

| Dataset | KNORA-U | KNORA-E | OLA | META-DES | KNOP | DESP | DESKNN |
|---|---|---|---|---|---|---|---|
| HAR | 1853.2 ms | 1862.5 ms | 1871.0 ms | 2857.3 ms | 2823.9 ms | 1886.4 ms | 1912.5 ms |
| Yeast | 60.7 ms | 62.0 ms | 63.3 ms | 108.2 ms | 84.3 ms | 61.0 ms | 70.1 ms |
| Image Segment | 20.9 ms | 21.2 ms | 20.8 ms | 36.9 ms | 32.2 ms | 21.3 ms | 25.0 ms |
| Vowel | 53.5 ms | 53.7 ms | 54.7 ms | 96.3 ms | 72.8 ms | 53.6 ms | 58.0 ms |
| Waveform | 186.2 ms | 191.4 ms | 193.1 ms | 333.2 ms | 312.2 ms | 197.7 ms | 211.4 ms |