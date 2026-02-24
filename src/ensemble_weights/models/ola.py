from ensemble_weights.base import BaseRouter
import numpy as np


class OLAModel(BaseRouter):
    """
    OLA: Overall Local Accuracy.

    Assigns full weight to the single model with the highest average score
    in the K nearest validation neighbors of each test point. No blending.
    """

    def __init__(self, metric, mode='max', neighbor_finder=None):
        self.metric = metric
        self.mode = mode
        self.model = neighbor_finder
        self.matrix = None
        self.models = None
        self.features = None

    def _compute_scores(self, y, preds):
        """Return a 1D array of per-sample metric scores."""
        return np.vectorize(self.metric)(y, preds)

    def fit(self, features, y, preds_dict):
        self.features = features
        self.models = list(preds_dict.keys())
        n_val, n_models = len(y), len(self.models)
        self.matrix = np.zeros((n_val, n_models))

        for j, name in enumerate(self.models):
            scores = self._compute_scores(y, preds_dict[name])
            # Negate minimization metrics so the matrix is always higher-is-better.
            self.matrix[:, j] = scores if self.mode == 'max' else -scores

        # Global normalization to [0, 1]. Argmax is invariant to this, but it
        # keeps the stored matrix consistent with KNNModel for inspection.
        mat_min, mat_max = self.matrix.min(), self.matrix.max()
        if mat_max > mat_min:
            self.matrix = (self.matrix - mat_min) / (mat_max - mat_min)

        self.model.fit(features)

    def predict(self, x, temperature=1.0):
        """
        temperature is accepted for API parity with KNNModel but ignored;
        OLA always assigns weight 1.0 to the single locally-best model.
        """
        x = np.atleast_2d(x)
        batch_size = x.shape[0]

        _, indices = self.model.kneighbors(x)

        avg_scores  = self.matrix[indices].mean(axis=1)  # (batch_size, n_models)
        best_indices = np.argmax(avg_scores, axis=1)

        weights = np.zeros((batch_size, len(self.models)))
        weights[np.arange(batch_size), best_indices] = 1.0

        if batch_size == 1:
            return dict(zip(self.models, weights[0]))
        return [dict(zip(self.models, w)) for w in weights]