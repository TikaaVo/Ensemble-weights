from ensemble_weights.base import BaseRouter
import numpy as np


class OLAModel(BaseRouter):
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

            # Negate for minimization metrics so argmax logic stays uniform
            self.matrix[:, j] = scores if self.mode == "max" else -scores

        # Normalize to [0, 1] globally for consistency with KNNModel.
        # OLA uses argmax so normalization doesn't change which model is selected,
        # but it keeps the stored matrix interpretable and future-proof.
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

        # OLA: hard assignment to single best model per sample
        best_indices = np.argmax(avg_scores, axis=1)  # (batch_size,)
        weights_array = np.zeros((batch_size, len(self.models)))
        weights_array[np.arange(batch_size), best_indices] = 1.0

        if batch_size == 1:
            return dict(zip(self.models, weights_array[0]))

        return [dict(zip(self.models, w)) for w in weights_array]