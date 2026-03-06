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

All despy algorithms use `preset="balanced"` (FAISS IVF) and `k=20`.

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

| Dataset | Best Single | Simple Avg | KNN-DWS | OLA | KNORA-U | KNORA-E | KNORA-IU |
|---|---|---|---|---|---|---|---|
| California Housing | 0.3956 ± 0.008 | +7.99% | **−2.24%** | −0.03% | −0.79% | +7.46% | −0.99% |
| Bike Sharing | 51.678 ± 0.860 | +47.77% | **−5.34%** | −2.97% | +6.57% | +14.79% | +5.39% |
| Abalone | 1.4981 ± 0.044 | +1.14% | +2.68% | +3.67% | +1.47% | +7.18% | +1.47% |
| Diabetes | 44.504 ± 2.645 | +3.18% | +1.17% | +3.56% | +5.86% | +15.34% | +5.74% |
| Concrete Strength | 5.2686 ± 0.336 | +23.66% | +1.68% | +3.54% | +2.49% | +11.84% | **−1.05%** |

---

KNORA variants are designed for classification, which explains the poor performance
on regression datasets; An exception occurs in Abalone, because ring amounts are whole numbers, usually 0-29.

## Classification results

Accuracy, higher is better. % shown as delta vs Best Single. 20-seed mean ± std.
Classification datasets include a comparison against [DESlib](https://github.com/scikit-learn-contrib/DESlib),
a mature sklearn-compatible DES library.

### HAR

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 98.24% | 0.23% | — |
| Simple Average | 97.92% | 0.26% | −0.33% |
| despy KNN-DWS | **98.38%** | 0.27% | **+0.14%** |
| despy OLA | 98.00% | 0.41% | −0.25% |
| despy KNORA-U | 98.18% | 0.29% | −0.06% |
| despy KNORA-E | 98.02% | 0.27% | −0.22% |
| despy KNORA-IU | 98.19% | 0.29% | −0.05% |
| DESlib KNORA-U | 98.00% | 0.26% | −0.25% |
| DESlib KNORA-E | 97.82% | 0.34% | −0.43% |
| DESlib OLA | 97.09% | 0.48% | −1.17% |
| DESlib META-DES | 98.35% | 0.30% | +0.11% |
| DESlib KNOP | 98.33% | 0.29% | +0.09% |
| DESlib DESP | 97.97% | 0.27% | −0.28% |
| DESlib DESKNN | 97.81% | 0.33% | −0.44% |

DESlib achieves a best mean score of 98.35%; despy achieves a best mean score of 98.38%.

### Yeast

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 58.87% | 2.20% | — |
| Simple Average | 59.33% | 1.96% | +0.77% |
| despy KNN-DWS | 59.46% | 1.94% | +1.00% |
| despy OLA | 58.16% | 1.94% | −1.20% |
| despy KNORA-U | 59.68% | 1.91% | +1.37% |
| despy KNORA-E | 56.82% | 2.37% | −3.49% |
| despy KNORA-IU | **59.85%** | 1.90% | **+1.66%** |
| DESlib KNORA-U | 59.48% | 1.81% | +1.03% |
| DESlib KNORA-E | 56.97% | 1.91% | −3.23% |
| DESlib OLA | 56.84% | 2.13% | −3.46% |
| DESlib META-DES | 57.46% | 2.35% | −2.40% |
| DESlib KNOP | 59.12% | 1.81% | +0.43% |
| DESlib DESP | 58.77% | 1.72% | −0.17% |
| DESlib DESKNN | 57.93% | 1.74% | −1.60% |

DESlib achieves a best mean score of 59.48%; despy achieves a best mean score of 59.85%.

### Image Segment

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 93.70% | 0.98% | — |
| Simple Average | 95.01% | 1.03% | +1.40% |
| despy KNN-DWS | 95.58% | 0.86% | +2.01% |
| despy OLA | 94.98% | 0.92% | +1.36% |
| despy KNORA-U | 95.37% | 0.92% | +1.78% |
| despy KNORA-E | 95.41% | 1.01% | +1.82% |
| despy KNORA-IU | **95.66%** | 0.89% | **+2.09%** |
| DESlib KNORA-U | 94.95% | 0.97% | +1.33% |
| DESlib KNORA-E | 95.25% | 0.89% | +1.65% |
| DESlib OLA | 94.65% | 0.89% | +1.02% |
| DESlib META-DES | 95.48% | 0.81% | +1.89% |
| DESlib KNOP | 95.19% | 1.00% | +1.59% |
| DESlib DESP | 94.68% | 0.91% | +1.04% |
| DESlib DESKNN | 94.76% | 1.00% | +1.13% |

DESlib achieves a best mean score of 95.48%; despy achieves a best mean score of 95.66%.

### Vowel

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 89.95% | 2.94% | — |
| Simple Average | 88.11% | 2.98% | −2.05% |
| despy KNN-DWS | 89.87% | 2.51% | −0.08% |
| despy OLA | 90.23% | 2.45% | +0.31% |
| despy KNORA-U | 90.15% | 2.49% | +0.22% |
| despy KNORA-E | 90.61% | 2.29% | +0.73% |
| despy KNORA-IU | **90.78%** | 2.14% | **+0.93%** |
| DESlib KNORA-U | 88.18% | 2.58% | −1.97% |
| DESlib KNORA-E | 89.47% | 2.45% | −0.53% |
| DESlib OLA | 88.38% | 2.96% | −1.74% |
| DESlib META-DES | 89.70% | 2.27% | −0.28% |
| DESlib KNOP | 88.61% | 2.58% | −1.49% |
| DESlib DESP | 85.56% | 2.99% | −4.88% |
| DESlib DESKNN | 85.23% | 3.60% | −5.25% |

DESlib achieves a best mean score of 89.70%; despy achieves a best mean score of 90.78%.

### Waveform

| Method | Mean | Std | vs Best Single |
|---|---|---|---|
| Best Single | 85.91% | 0.76% | — |
| Simple Average | 85.07% | 0.76% | −0.98% |
| despy KNN-DWS | 85.57% | 0.79% | −0.40% |
| despy OLA | 84.15% | 0.89% | −2.04% |
| despy KNORA-U | 85.41% | 0.80% | −0.59% |
| despy KNORA-E | 82.91% | 1.12% | −3.50% |
| despy KNORA-IU | 85.42% | 0.78% | −0.58% |
| DESlib KNORA-U | 85.61% | 0.82% | −0.35% |
| DESlib KNORA-E | 83.19% | 1.02% | −3.17% |
| DESlib OLA | 81.14% | 1.15% | −5.55% |
| DESlib META-DES | 85.19% | 0.91% | −0.84% |
| DESlib KNOP | **85.97%** | 0.97% | **+0.07%** |
| DESlib DESP | 85.50% | 0.82% | −0.47% |
| DESlib DESKNN | 84.39% | 0.95% | −1.78% |

DESlib achieves a best mean score of 85.97%; despy achieves a best mean score of 85.57%.

---

## Timing

Mean fit + predict time in milliseconds, averaged across 20 seeds. Fit is measured once
per dataset per seed; predict is measured over the full test set.

despy caches all model predictions on the validation set at fit time and reads from that
matrix at inference, so no model is called at predict time. This is the primary reason for
the speed advantage over DESlib, which calls each model live per neighbour at inference.

despy used `preset='balanced'`, which uses FAISS IVF instead of KNN, but the difference
in performance isn't very pronounced in datasets of the size used.

### despy

| Dataset | KNN-DWS | OLA | KNORA-U | KNORA-E | KNORA-IU |
|---|---|---|---|---|---|
| California Housing | 25.2 ms | 22.5 ms | 26.2 ms | 34.8 ms | 27.9 ms |
| Bike Sharing | 20.7 ms | 19.5 ms | 22.5 ms | 29.7 ms | 23.1 ms |
| Abalone | 5.3 ms | 4.8 ms | 5.5 ms | 7.3 ms | 5.6 ms |
| Diabetes | 1.4 ms | 1.3 ms | 1.3 ms | 1.5 ms | 1.3 ms |
| Concrete Strength | 1.7 ms | 1.6 ms | 1.9 ms | 2.4 ms | 1.8 ms |
| HAR | 67.9 ms | 55.0 ms | 56.3 ms | 60.6 ms | 57.7 ms |
| Yeast | 4.2 ms | 3.2 ms | 2.9 ms | 3.1 ms | 2.9 ms |
| Image Segment | 5.8 ms | 5.4 ms | 5.1 ms | 5.4 ms | 5.5 ms |
| Vowel | 3.5 ms | 3.3 ms | 3.3 ms | 3.3 ms | 3.1 ms |
| Waveform | 10.3 ms | 9.5 ms | 9.1 ms | 10.1 ms | 9.9 ms |

### DESlib (classification datasets only)

| Dataset | KNORA-U | KNORA-E | OLA | META-DES | KNOP | DESP | DESKNN |
|---|---|---|---|---|---|---|---|
| HAR | 1859.8 ms | 1855.4 ms | 1870.5 ms | 2843.1 ms | 2819.2 ms | 1868.3 ms | 1911.7 ms |
| Yeast | 57.5 ms | 58.2 ms | 60.0 ms | 104.5 ms | 80.0 ms | 57.7 ms | 67.3 ms |
| Image Segment | 21.2 ms | 20.7 ms | 19.9 ms | 36.0 ms | 31.5 ms | 21.1 ms | 25.1 ms |
| Vowel | 51.4 ms | 50.8 ms | 50.9 ms | 90.2 ms | 68.9 ms | 50.0 ms | 54.9 ms |
| Waveform | 182.2 ms | 187.0 ms | 187.0 ms | 323.0 ms | 304.2 ms | 194.3 ms | 206.7 ms |