import numpy as np
from ensemble_weights.utils import to_numpy, add_batch_dim

metrics = {
    'accuracy': lambda y_true, y_pred: 1 if y_true == y_pred else 0,
    'mse': lambda y_true, y_pred: (y_true - y_pred) ** 2,
    'mae': lambda y_true, y_pred: abs(y_true - y_pred),
    'rmse': lambda y_true, y_pred: ((y_true - y_pred) ** 2) ** 0.5
}

class DynamicRouter:
    def __init__(self, task, dtype, method='knn', metric="accuracy", mode="max", feature_extractor=None, **kwargs):
        self.task = task
        self.dtype = dtype
        self.method = method
        if isinstance(metric, str):
            if metric in metrics:
                metric = metrics[metric]
            else:
                raise ValueError(f"Unknown metric name '{metric}'. Available: {list(metrics.keys())}")
        self.metric = metric
        self.kwargs = kwargs
        self.mode = mode
        self.feature_extractor = feature_extractor
        self.model, model_name = self.create_model()

    def create_model(self):
        if self.method == 'knn':
            if self.dtype == 'tabular' or self.dtype == "image":
                from ensemble_weights.knn import KNNModel
                return KNNModel(metric=self.metric, mode=self.mode, **self.kwargs), 'KNN'
        raise ValueError(f"Unsupported combination: method={self.method}, dtype={self.dtype}")

    def fit(self, features, y, preds_dict):
        if self.feature_extractor is not None:
            features = self.feature_extractor(features)

        features = to_numpy(features)
        y = to_numpy(y)
        preds_dict = {name: to_numpy(preds) for name, preds in preds_dict.items()}
        self.model.fit(features, y, preds_dict)

    def predict(self, x, temperature=1.0):
        if self.feature_extractor is not None:
            x = add_batch_dim(x)
            x = self.feature_extractor(x)[0]
        x = to_numpy(x)

        return self.model.predict(x, temperature)