from ensemble_weights.base import BaseRouter
import numpy as np


class KNNModel(BaseRouter):
    def __init__(self, metric, mode="max", neighbor_finder=None):
        self.metric = metric
        self.mode = mode
        self.model = neighbor_finder
        self.matrix = None
        self.models = None
        self.features = None

    def _compute_scores(self, y, preds):
        """
        Compute per-sample scores, always returning a 1D array of length n_samples.

        Vectorizes unconditionally rather than using a try/except fallback.
        This avoids the silent slow path and is consistent across all metrics.
        """
        v_metric = np.vectorize(self.metric)
        return v_metric(y, preds)

    def fit(self, features, y, preds_dict):
        self.features = features
        self.models = list(preds_dict.keys())
        n_val = len(y)
        n_models = len(self.models)
        self.matrix = np.zeros((n_val, n_models))

        for j, name in enumerate(self.models):
            preds = preds_dict[name]
            scores = self._compute_scores(y, preds)

            # Negate for minimization metrics so argmax/softmax logic stays uniform
            self.matrix[:, j] = scores if self.mode == "max" else -scores

        # Normalize the full matrix to [0, 1] globally.
        # Without this, metrics with large absolute values (e.g. MAE on house prices
        # in the tens of thousands) cause softmax to collapse to one-hot for every
        # sample, making knn-dw indistinguishable from OLA regardless of temperature.
        # Global normalization (vs per-column) preserves cross-model comparability:
        # a model that is bad everywhere stays worse than one that is good everywhere.
        mat_min = self.matrix.min()
        mat_max = self.matrix.max()
        if mat_max > mat_min:
            self.matrix = (self.matrix - mat_min) / (mat_max - mat_min)

        self.model.fit(features)

    def predict(self, x, temperature=1.0):
        x = np.atleast_2d(x)
        batch_size = x.shape[0]

        distances, indices = self.model.kneighbors(x)

        # indices shape: (batch_size, k)
        # matrix shape: (n_val, n_models)
        # neighbor_scores shape: (batch_size, k, n_models)
        neighbor_scores = self.matrix[indices]
        avg_scores = neighbor_scores.mean(axis=1)  # (batch_size, n_models)

        # Numerically stable softmax with temperature scaling
        max_scores = avg_scores.max(axis=1, keepdims=True)
        exp_scores = np.exp((avg_scores - max_scores) / temperature)
        weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        if batch_size == 1:
            return dict(zip(self.models, weights[0]))

        return [dict(zip(self.models, w)) for w in weights]