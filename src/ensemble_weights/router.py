"""
DynamicRouter — string-based factory for programmatic algorithm selection.

Prefer importing algorithm classes directly for new code:

    from ensemble_weights.des.knndws import KNNDWS
    from ensemble_weights.des.ola    import OLA
    from ensemble_weights.des.knorau import KNORAU
    from ensemble_weights.des.knorae import KNORAE

Use DynamicRouter when you need to choose an algorithm via a string at runtime,
for example in benchmark loops:

    from ensemble_weights import DynamicRouter

    for method in ['knn-dws', 'ola', 'knora-u', 'knora-e']:
        router = DynamicRouter(task='classification', method=method, ...)
"""
from ensemble_weights.des.knndws import KNNDWS
from ensemble_weights.des.ola    import OLA
from ensemble_weights.des.knorau import KNORAU
from ensemble_weights.des.knorae import KNORAE
from ensemble_weights._config    import SPEED_PRESETS, list_presets
from ensemble_weights.utils      import to_numpy, add_batch_dim

_METHOD_CLASSES = {
    'knn-dws': KNNDWS,
    'ola':     OLA,
    'knora-u': KNORAU,
    'knora-e': KNORAE,
}


class DynamicRouter:
    """
    String-based factory for Dynamic Ensemble Selection.

    Creates and delegates to one of the four algorithm classes based on the
    `method` string. Prefer importing algorithm classes directly (see module
    docstring) unless you genuinely need runtime method selection.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    method : str
        'knn-dws', 'ola', 'knora-u', or 'knora-e'.
    metric : str or callable
        Per-sample scoring function. Built-in names: 'accuracy', 'mae', 'mse',
        'rmse', 'log_loss', 'prob_correct'. Or any callable (y_true, y_pred) -> float.
    mode : str
        'max' if higher scores are better, 'min' if lower.
    k : int
        Number of neighbors per query.
    threshold : float
        Competence gate applied after per-neighborhood normalization.
    temperature : float, optional
        Softmax sharpness for knn-dws. Ignored by other algorithms.
    preset : str
        Speed/accuracy preset. Call list_presets() for options.
    feature_extractor : callable, optional
        Applied to inputs before neighbor search.
    finder : str, optional
        Required when preset='custom'. One of 'knn', 'faiss', 'annoy', 'hnsw'.
    **kwargs
        Forwarded to the neighbor finder constructor.
    """

    def __init__(self, task, method='knn-dws', metric='accuracy', mode='max',
                 k=10, threshold=0.5, temperature=None, preset='balanced',
                 feature_extractor=None, finder=None, **kwargs):

        method = method.lower()
        if method not in _METHOD_CLASSES:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available: {sorted(_METHOD_CLASSES)}."
            )

        self._feature_extractor = feature_extractor
        self._method            = method
        self._temperature       = temperature

        cls = _METHOD_CLASSES[method]

        # Pass finder through as a kwarg when using preset='custom'.
        extra = {'finder': finder} if finder is not None else {}

        # KNNDWS accepts temperature; the others don't.
        if method == 'knn-dws':
            self._des = cls(
                task=task, metric=metric, mode=mode, k=k,
                threshold=threshold, temperature=temperature,
                preset=preset, **extra, **kwargs
            )
        else:
            self._des = cls(
                task=task, metric=metric, mode=mode, k=k,
                threshold=threshold, preset=preset, **extra, **kwargs
            )

    # ── Delegate fit / predict ────────────────────────────────────────

    def fit(self, features, y, preds_dict):
        """
        Fit the routing model on validation data.

        Parameters
        ----------
        features : array-like, shape (n_val, n_features)
        y : array-like, shape (n_val,)
        preds_dict : dict[str, array-like]
        """
        if self._feature_extractor is not None:
            features = self._feature_extractor(features)
        self._des.fit(features, y, preds_dict)

    def predict(self, x, temperature=None, threshold=None):
        """
        Return per-sample model weights.

        Parameters
        ----------
        x : array-like, shape (n_features,) or (n_samples, n_features)
        temperature : float, optional
            knn-dws only. Overrides the instance temperature for this call.
        threshold : float, optional
            Overrides the instance threshold for this call.

        Returns
        -------
        dict or list of dict
        """
        if self._feature_extractor is not None:
            x = add_batch_dim(to_numpy(x))
            x = self._feature_extractor(x)[0]

        if self._method == 'knn-dws':
            return self._des.predict(x, temperature=temperature, threshold=threshold)
        elif self._method == 'ola':
            return self._des.predict(x)
        else:
            return self._des.predict(x, threshold=threshold)

    # ── Class methods ─────────────────────────────────────────────────

    @classmethod
    def from_data_size(cls, n_samples, n_features, task, method='knn-dws',
                       metric='accuracy', mode='max', k=10, threshold=0.5,
                       n_queries=None, **extra_kwargs):
        """
        Recommend and instantiate a preset based on dataset dimensions.

        Parameters
        ----------
        n_queries : int, optional
            Expected test set size. Used to decide whether ANN fit overhead
            is worthwhile relative to the number of prediction calls.
        """
        if n_samples < 10000:
            preset, reason = 'exact', "Small dataset (<10K) — exact search is fast enough"
        elif n_features < 20:
            preset, reason = 'exact', "Low-dimensional (<20D) — ANN overhead not worthwhile"
        elif n_samples < 100000:
            preset, reason = 'balanced', "Medium dataset (10K-100K) — balanced speed/accuracy"
        elif n_features > 100:
            preset = 'high_dim_fast' if n_samples > 1_000_000 else 'high_dim_balanced'
            reason = "High-dimensional (>100D) — HNSW recommended"
        elif n_samples > 1_000_000:
            preset, reason = 'turbo', "Very large dataset (>1M) — prioritise speed"
        else:
            preset, reason = 'fast', "Large dataset (100K-1M) — fast with good accuracy"

        if n_queries is not None and preset != 'exact':
            if n_queries < n_samples * 0.05:
                preset = 'exact'
                reason = (
                    f"Low query volume ({n_queries:,} queries vs {n_samples:,} val samples) "
                    f"— ANN fit overhead not recouped; exact search is faster overall"
                )

        print(f"Auto-selected preset: '{preset}'\nReason: {reason}")
        print(f"Data: {n_samples:,} samples, {n_features} features"
              + (f", {n_queries:,} queries" if n_queries is not None else ""))

        return cls(
            task=task, method=method, metric=metric, mode=mode,
            preset=preset, k=k, threshold=threshold, **extra_kwargs
        )

    @classmethod
    def list_presets(cls):
        """Print all available presets with descriptions and parameters."""
        list_presets()

    def get_config_info(self):
        """Return a dict summarising the current configuration."""
        des = self._des
        return {
            'method':    self._method,
            'metric':    getattr(des, '_metric_name', '<custom callable>'),
            'mode':      des.mode,
            'threshold': getattr(des, 'threshold', None),
        }