from ensemble_weights.base import BaseRouter
import numpy as np
from sklearn.neighbors import NearestNeighbors

class KNNModel(BaseRouter):
    def __init__(self, metric, mode="max", k=10):
        self.k = k
        self.metric = metric
        self.mode = mode
        self.model = NearestNeighbors(n_neighbors=k)
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
        distances, indices = self.model.kneighbors(x.reshape(1,-1))
        scores = self.matrix[indices[0]]
        avg_scores = scores.mean(axis=0)
        exp_scores = np.exp((avg_scores - avg_scores.max()) / temperature)
        weights = exp_scores / exp_scores.sum()
        return dict(zip(self.models, weights))