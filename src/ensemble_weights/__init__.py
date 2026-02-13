def accuracy(x, y):
    return 0 if x != y else 1

class DynamicRouter:
    def __init__(self, task, dtype, method='knn', metric="accuracy", **kwargs):
        self.task = task
        self.dtype = dtype
        self.method = method
        if metric == "accuracy":
            metric = accuracy
        self.metric = metric
        self.kwargs = kwargs
        self.model, model_name = self.create_model()

    def create_model(self):
        if self.method == 'knn':
            if self.dtype == 'tabular':
                from ensemble_weights.knn import KNNModel
                return KNNModel(metric=self.metric, **self.kwargs), 'KNN'
        raise ValueError(f"Unsupported combination: method={self.method}, dtype={self.dtype}")

    def fit(self, features, y, preds_dict):
        self.model.fit(features, y, preds_dict)

    def predict(self, x, temperature=1.0):
        return self.model.predict(x, temperature)