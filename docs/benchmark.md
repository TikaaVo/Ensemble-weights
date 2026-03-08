# Benchmark

All results are from a 20-seed benchmark (seeds 0–19) on standard datasets. Each seed
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

MAE, lower is better. % shown as delta vs Best Single. 20-seed mean ± std.

| Dataset | Best Single | Simple Avg | DEWS-U | DEWS-I | DEWS-T | DEWS-V | DEWS-IV | LWSE-U | LWSE-I | OLA | KNORA-U | KNORA-E | KNORA-IU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| California Housing | 0.3956 ± 0.008 | +7.99% | −2.24% | **−2.54%** | −2.52% | −1.17% | −1.44% | −1.67% | −1.85% | −0.03% | −0.79% | +7.46% | −0.99% |
| Bike Sharing | 51.678 ± 0.860 | +47.77% | −5.34% | **−6.86%** | −6.85% | −3.25% | −4.63% | −3.85% | −4.59% | −2.97% | +6.57% | +14.79% | +5.39% |
| Abalone | **1.4981 ± 0.044** | +1.14% | +2.68% | +2.82% | +2.80% | +3.22% | +3.20% | +3.33% | +3.38% | +3.67% | +1.47% | +7.18% | +1.47% |
| Diabetes | **44.504 ± 2.645** | +3.18% | +1.17% | +1.09% | +1.09% | +1.09% | **+0.86%** | +3.36% | +3.25% | +3.56% | +5.86% | +15.34% | +5.74% |
| Concrete Strength | 5.2686 ± 0.336 | +23.66% | +1.68% | −1.20% | −1.01% | +3.36% | +0.46% | −3.46% | **−5.41%** | +3.54% | +2.49% | +11.84% | −1.05% |

---

KNORA variants are designed for classification, which explains the poor performance
on regression datasets; However, some exception can occur in certain datasets, either where
feature space has hard clusters (like in Concrete Strength) or when the target is discrete
and classification-like (like in Abalone).

LWSE-I is the clear winner on Concrete Strength (−5.41%), where strong local competence heterogeneity
allows the per-sample NNLS solver to find genuine local blends. DEWS-IV edges all other algorithms on
Diabetes, the only dataset where every ensembling method loses to the best single model.

## Classification results

Accuracy, higher is better. % shown as delta vs Best Single. 20-seed mean ± std.
Classification datasets include a comparison against [DESlib](https://github.com/scikit-learn-contrib/DESlib),
a mature sklearn-compatible DES library.

### HAR

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 98.24% | 0.23% | — |
| Simple Average | 97.92% | 0.26% | −0.33% |
| deskit DEWS-U | 98.38% | 0.27% | +0.14% |
| deskit DEWS-I | 98.39% | 0.27% | +0.15% |
| deskit DEWS-T | **98.40%** | 0.28% | **+0.16%** |
| deskit DEWS-V | 98.38% | 0.27% | +0.14% |
| deskit DEWS-IV | 98.38% | 0.27% | +0.14% |
| deskit LWSE-U | 98.17% | 0.34% | −0.07% |
| deskit LWSE-I | 98.20% | 0.34% | −0.04% |
| deskit OLA | 98.00% | 0.41% | −0.25% |
| deskit KNORA-U | 98.18% | 0.29% | −0.06% |
| deskit KNORA-E | 98.02% | 0.27% | −0.22% |
| deskit KNORA-IU | 98.19% | 0.29% | −0.05% |
| DESlib KNORA-U | 98.00% | 0.26% | −0.25% |
| DESlib KNORA-E | 97.82% | 0.34% | −0.43% |
| DESlib OLA | 97.09% | 0.48% | −1.17% |
| DESlib LCA | 91.29% | 1.27% | −7.08% |
| DESlib MCB | 97.03% | 0.36% | −1.24% |
| DESlib META-DES | 98.35% | 0.30% | +0.11% |
| DESlib KNOP | 98.33% | 0.29% | +0.09% |
| DESlib DESP | 97.97% | 0.27% | −0.28% |
| DESlib DESKNN | 97.81% | 0.33% | −0.44% |
| DESlib DES-MI | 95.56% | 1.31% | −2.73% |

deskit achieves a best mean score of 98.40%; DESlib achieves a best mean score of 98.35%.

### Yeast

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 58.87% | 2.20% | — |
| Simple Average | 59.33% | 1.96% | +0.77% |
| deskit DEWS-U | 59.46% | 1.94% | +1.00% |
| deskit DEWS-I | 59.58% | 1.99% | +1.20% |
| deskit DEWS-T | 59.71% | 2.07% | +1.43% |
| deskit DEWS-V | 59.51% | 2.05% | +1.09% |
| deskit DEWS-IV | 59.56% | 2.19% | +1.17% |
| deskit LWSE-U | 58.40% | 1.97% | −0.80% |
| deskit LWSE-I | 58.42% | 2.10% | −0.77% |
| deskit OLA | 58.16% | 1.94% | −1.20% |
| deskit KNORA-U | 59.68% | 1.91% | +1.37% |
| deskit KNORA-E | 56.82% | 2.37% | −3.49% |
| deskit KNORA-IU | **59.85%** | 1.90% | **+1.66%** |
| DESlib KNORA-U | 59.48% | 1.81% | +1.03% |
| DESlib KNORA-E | 56.97% | 1.91% | −3.23% |
| DESlib OLA | 56.84% | 2.13% | −3.46% |
| DESlib LCA | 55.71% | 1.62% | −5.38% |
| DESlib MCB | 57.22% | 2.34% | −2.80% |
| DESlib META-DES | 57.46% | 2.35% | −2.40% |
| DESlib KNOP | 59.12% | 1.81% | +0.43% |
| DESlib DESP | 58.77% | 1.72% | −0.17% |
| DESlib DESKNN | 57.93% | 1.74% | −1.60% |
| DESlib DES-MI | 56.90% | 2.25% | −3.35% |

deskit achieves a best mean score of 59.85%; DESlib achieves a best mean score of 59.48%.

### Image Segment

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 93.70% | 0.98% | — |
| Simple Average | 95.01% | 1.03% | +1.40% |
| deskit DEWS-U | 95.58% | 0.86% | +2.01% |
| deskit DEWS-I | 95.79% | 0.95% | +2.23% |
| deskit DEWS-T | **95.81%** | 0.98% | **+2.25%** |
| deskit DEWS-V | 95.55% | 0.93% | +1.98% |
| deskit DEWS-IV | **95.81%** | 0.91% | **+2.25%** |
| deskit LWSE-U | 95.58% | 0.87% | +2.01% |
| deskit LWSE-I | 95.79% | 0.82% | +2.23% |
| deskit OLA | 94.98% | 0.92% | +1.36% |
| deskit KNORA-U | 95.37% | 0.92% | +1.78% |
| deskit KNORA-E | 95.41% | 1.01% | +1.82% |
| deskit KNORA-IU | 95.66% | 0.89% | +2.09% |
| DESlib KNORA-U | 94.95% | 0.97% | +1.33% |
| DESlib KNORA-E | 95.25% | 0.89% | +1.65% |
| DESlib OLA | 94.65% | 0.89% | +1.02% |
| DESlib LCA | 92.35% | 1.00% | −1.44% |
| DESlib MCB | 94.62% | 1.08% | +0.98% |
| DESlib META-DES | 95.48% | 0.81% | +1.89% |
| DESlib KNOP | 95.19% | 1.00% | +1.59% |
| DESlib DESP | 94.68% | 0.91% | +1.04% |
| DESlib DESKNN | 94.76% | 1.00% | +1.13% |
| DESlib DES-MI | 94.76% | 1.00% | +1.13% |

deskit achieves a best mean score of 95.81%; DESlib achieves a best mean score of 95.48%.

### Vowel

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 89.95% | 2.94% | — |
| Simple Average | 88.11% | 2.98% | −2.05% |
| deskit DEWS-U | 89.87% | 2.51% | −0.08% |
| deskit DEWS-I | 90.25% | 2.52% | +0.34% |
| deskit DEWS-T | 90.38% | 2.56% | +0.48% |
| deskit DEWS-V | 89.80% | 2.57% | −0.17% |
| deskit DEWS-IV | 90.10% | 2.52% | +0.17% |
| deskit LWSE-U | 91.84% | 1.97% | +2.11% |
| deskit LWSE-I | **92.60%** | 1.81% | **+2.95%** |
| deskit OLA | 90.23% | 2.45% | +0.31% |
| deskit KNORA-U | 90.15% | 2.49% | +0.22% |
| deskit KNORA-E | 90.61% | 2.29% | +0.73% |
| deskit KNORA-IU | 90.78% | 2.14% | +0.93% |
| DESlib KNORA-U | 88.18% | 2.58% | −1.97% |
| DESlib KNORA-E | 89.47% | 2.45% | −0.53% |
| DESlib OLA | 88.38% | 2.96% | −1.74% |
| DESlib LCA | 78.61% | 3.96% | −12.61% |
| DESlib MCB | 86.36% | 2.95% | −3.99% |
| DESlib META-DES | 89.70% | 2.27% | −0.28% |
| DESlib KNOP | 88.61% | 2.58% | −1.49% |
| DESlib DESP | 85.56% | 2.99% | −4.88% |
| DESlib DESKNN | 85.23% | 3.60% | −5.25% |
| DESlib DES-MI | 85.23% | 3.60% | −5.25% |

deskit achieves a best mean score of 92.60%; DESlib achieves a best mean score of 89.70%.

### Waveform

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | **85.91%** | 0.76% | — |
| Simple Average | 85.07% | 0.76% | −0.98% |
| deskit DEWS-U | 85.57% | 0.79% | −0.40% |
| deskit DEWS-I | 85.54% | 0.79% | −0.43% |
| deskit DEWS-T | 85.57% | 0.75% | −0.39% |
| deskit DEWS-V | 85.43% | 0.80% | −0.55% |
| deskit DEWS-IV | 85.43% | 0.81% | −0.56% |
| deskit LWSE-U | 83.87% | 0.96% | −2.37% |
| deskit LWSE-I | 83.92% | 0.94% | −2.32% |
| deskit OLA | 84.15% | 0.89% | −2.04% |
| deskit KNORA-U | 85.41% | 0.80% | −0.59% |
| deskit KNORA-E | 82.91% | 1.12% | −3.50% |
| deskit KNORA-IU | 85.42% | 0.78% | −0.58% |
| DESlib KNORA-U | 85.61% | 0.82% | −0.35% |
| DESlib KNORA-E | 83.19% | 1.02% | −3.17% |
| DESlib OLA | 81.14% | 1.15% | −5.55% |
| DESlib LCA | 77.10% | 1.64% | −10.25% |
| DESlib MCB | 82.16% | 1.18% | −4.36% |
| DESlib META-DES | 85.19% | 0.91% | −0.84% |
| DESlib KNOP | **85.97%** | 0.97% | **+0.07%** |
| DESlib DESP | 85.50% | 0.82% | −0.47% |
| DESlib DESKNN | 84.39% | 0.95% | −1.78% |
| DESlib DES-MI | 84.07% | 1.02% | −2.14% |

deskit achieves a best mean score of 85.57%; DESlib achieves a best mean score of 85.97%.

---

## Timing

Mean fit + predict time in milliseconds, averaged across 20 seeds. Fit is measured once
per dataset per seed; predict is measured over the full test set.

deskit caches all model predictions on the validation set at fit time and reads from that
matrix at inference, so no model is called at predict time. This is the primary reason for
the speed advantage over DESlib, which calls each model live per neighbour at inference.

deskit used `preset='balanced'`, which uses FAISS IVF instead of KNN, but the difference
in performance isn't very pronounced in datasets of the size used.

### deskit

| Dataset | DEWS-U | DEWS-I | DEWS-T | DEWS-V | DEWS-IV | LWSE-U | LWSE-I | OLA | KNORA-U | KNORA-E | KNORA-IU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| California Housing | 25.2 ms | 23.5 ms | 31.0 ms | 27.8 ms | 29.1 ms | 44.1 ms | 59.3 ms | 22.4 ms | 26.3 ms | 34.7 ms | 27.6 ms |
| Bike Sharing | 19.5 ms | 19.1 ms | 25.2 ms | 22.7 ms | 23.5 ms | 35.2 ms | 47.6 ms | 18.8 ms | 21.0 ms | 28.3 ms | 22.6 ms |
| Abalone | 5.2 ms | 5.2 ms | 6.8 ms | 6.0 ms | 6.5 ms | 8.8 ms | 11.9 ms | 4.5 ms | 5.4 ms | 7.3 ms | 5.3 ms |
| Diabetes | 1.4 ms | 1.3 ms | 1.4 ms | 1.4 ms | 1.3 ms | 1.5 ms | 1.8 ms | 1.0 ms | 1.3 ms | 1.4 ms | 1.2 ms |
| Concrete Strength | 1.8 ms | 1.8 ms | 2.4 ms | 2.0 ms | 2.0 ms | 2.4 ms | 3.2 ms | 1.4 ms | 1.6 ms | 2.2 ms | 1.7 ms |
| HAR | 65.8 ms | 56.1 ms | 60.1 ms | 57.6 ms | 58.3 ms | 74.1 ms | 83.4 ms | 56.6 ms | 57.8 ms | 63.2 ms | 60.1 ms |
| Yeast | 4.5 ms | 3.9 ms | 4.5 ms | 3.7 ms | 3.6 ms | 5.8 ms | 7.0 ms | 2.8 ms | 2.9 ms | 3.1 ms | 2.9 ms |
| Image Segment | 6.0 ms | 5.4 ms | 6.3 ms | 5.4 ms | 5.6 ms | 8.5 ms | 11.2 ms | 5.3 ms | 5.3 ms | 5.4 ms | 5.5 ms |
| Vowel | 3.4 ms | 3.3 ms | 3.5 ms | 3.2 ms | 3.1 ms | 4.5 ms | 5.7 ms | 3.1 ms | 3.0 ms | 3.1 ms | 3.1 ms |
| Waveform | 10.0 ms | 10.0 ms | 10.5 ms | 10.2 ms | 10.4 ms | 15.2 ms | 19.8 ms | 9.4 ms | 9.3 ms | 9.9 ms | 9.8 ms |

### DESlib (classification datasets only)

| Dataset | KNORA-U | KNORA-E | OLA | LCA | MCB | META-DES | KNOP | DESP | DESKNN | DES-MI |
|---|---|---|---|---|---|---|---|---|---|---|
| HAR | 1886.2 ms | 1884.9 ms | 1884.5 ms | 1901.0 ms | 1900.5 ms | 2905.5 ms | 2841.3 ms | 1917.8 ms | 1946.7 ms | 1919.4 ms |
| Yeast | 56.7 ms | 57.3 ms | 58.4 ms | 59.4 ms | 63.3 ms | 99.8 ms | 79.5 ms | 56.8 ms | 66.1 ms | 57.1 ms |
| Image Segment | 22.6 ms | 22.1 ms | 21.8 ms | 22.6 ms | 22.8 ms | 37.7 ms | 33.2 ms | 21.3 ms | 25.6 ms | 21.5 ms |
| Vowel | 50.3 ms | 49.1 ms | 50.2 ms | 52.1 ms | 55.2 ms | 89.6 ms | 68.2 ms | 49.4 ms | 53.4 ms | 49.9 ms |
| Waveform | 185.3 ms | 192.1 ms | 194.2 ms | 195.0 ms | 201.6 ms | 332.9 ms | 312.7 ms | 204.3 ms | 220.9 ms | 202.1 ms |