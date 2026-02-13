from ensemble_weights.utils import to_numpy, add_batch_dim

metrics = {
    'accuracy': lambda y_true, y_pred: 1 if y_true == y_pred else 0,
    'mse': lambda y_true, y_pred: (y_true - y_pred) ** 2,
    'mae': lambda y_true, y_pred: abs(y_true - y_pred),
    'rmse': lambda y_true, y_pred: ((y_true - y_pred) ** 2) ** 0.5
}

class DynamicRouter:
    def __init__(self, task, dtype, method='knn', metric="accuracy", mode="max", feature_extractor=None, finder="knn", **kwargs):
        self.task = task.lower()
        self.dtype = dtype.lower()
        self.method = method.lower()
        if isinstance(metric, str):
            if metric in metrics:
                metric = metrics[metric.lower()]
            else:
                raise ValueError(f"Unknown metric name '{metric}'. Available: {list(metrics.keys())}")
        self.metric = metric
        self.kwargs = kwargs
        self.mode = mode.lower()
        self.feature_extractor = feature_extractor
        self.finder = finder.lower()
        self.model, model_name = self.create_model()

    def create_model(self):
        if self.finder == "knn":
            from ensemble_weights.neighbors import KNNNeighborFinder
            finder = KNNNeighborFinder(**self.kwargs)
        elif self.finder == "faiss":
            from ensemble_weights.neighbors import FaissNeighborFinder
            finder = FaissNeighborFinder(**self.kwargs)
        elif self.finder == "hnsw":
            from ensemble_weights.neighbors import HNSWNeighborFinder
            finder = HNSWNeighborFinder(**self.kwargs)
        else:
            raise ValueError(f"Unknown nn_backend: {self.finder}")

        if self.method == 'knn-dw':
            if self.dtype == 'tabular' or self.dtype == "image":
                from ensemble_weights.models.knn import KNNModel
                return KNNModel(metric=self.metric, mode=self.mode, neighbor_finder=finder), 'KNN-DW'
        elif self.method == 'ola':
            if self.dtype == 'tabular' or self.dtype == "image":
                from ensemble_weights.models.ola import OLAModel
                return OLAModel(metric=self.metric, mode=self.mode, neighbor_finder=finder), 'OLA'
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