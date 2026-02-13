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
            for i in range(n_val):
                score = self.metric(y[i], preds[i])
                if self.mode == "max":
                    self.matrix[i,j] = score
                else:
                    self.matrix[i,j] = -score

        self.model.fit(features)

    def predict(self, x, temperature=1.0):
        distances, indices = self.model.kneighbors(x.reshape(1, -1))
        neighbor_scores = self.matrix[indices]
        avg_scores = neighbor_scores.mean(axis=0)
        best_idx = np.argmax(avg_scores)
        weights = {name: 0.0 for name in self.models}
        weights[self.models[best_idx]] = 1.0
        return weights