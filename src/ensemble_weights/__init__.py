metrics = {
    'accuracy': lambda y_true, y_pred: 1 if y_true == y_pred else 0,
    'mse': lambda y_true, y_pred: (y_true - y_pred) ** 2,          # mean squared error
    'mae': lambda y_true, y_pred: abs(y_true - y_pred),            # mean absolute error
    'rmse': lambda y_true, y_pred: ((y_true - y_pred) ** 2) ** 0.5
}

class DynamicRouter:
    def __init__(self, task, dtype, method='knn', metric="accuracy", mode="max", **kwargs):
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
        self.model, model_name = self.create_model()

    def create_model(self):
        if self.method == 'knn':
            if self.dtype == 'tabular':
                from ensemble_weights.knn import KNNModel
                return KNNModel(metric=self.metric, mode=self.mode, **self.kwargs), 'KNN'
        raise ValueError(f"Unsupported combination: method={self.method}, dtype={self.dtype}")

    def fit(self, features, y, preds_dict):
        self.model.fit(features, y, preds_dict)

    def predict(self, x, temperature=1.0):
        return self.model.predict(x, temperature)