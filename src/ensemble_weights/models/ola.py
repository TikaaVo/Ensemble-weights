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

    def fit(self, features, y, preds_dict):
        self.features = features
        self.models = list(preds_dict.keys())
        n_val = len(y)
        n_models = len(self.models)

        self.matrix = np.zeros((n_val, n_models))

        for j, name in enumerate(self.models):
            preds = preds_dict[name]

            try:
                scores = self.metric(y, preds)
            except (ValueError, TypeError):
                v_metric = np.vectorize(self.metric)
                scores = v_metric(y, preds)

            if self.mode == "max":
                self.matrix[:, j] = scores
            else:
                self.matrix[:, j] = -scores

        self.model.fit(features)

    def predict(self, x, temperature=1.0):
        x = np.atleast_2d(x)
        batch_size = x.shape[0]

        distances, indices = self.model.kneighbors(x)
        neighbor_scores = self.matrix[indices]
        avg_scores = neighbor_scores.mean(axis=1)
        best_indices = np.argmax(avg_scores, axis=1)

        weights_array = np.zeros((batch_size, len(self.models)))
        weights_array[np.arange(batch_size), best_indices] = 1.0

        if batch_size == 1:
            return dict(zip(self.models, weights_array[0]))

        return [dict(zip(self.models, w)) for w in weights_array]