"""
Dynamic Ensemble Selection library.

Main entry point: DynamicRouter.
Supports knn-dws (soft per-sample weighting), knora-u (vote-based selection),
and OLA (hard per-sample selection),
with pluggable neighbor finders for exact and approximate search.
"""
import math
from ensemble_weights.utils import to_numpy, add_batch_dim

# Built-in per-sample metrics. All take scalar (y_true, y_pred) and return float.
# Pass any callable with the same signature to use a custom metric.
#
# Scalar metrics — y_pred is a single number:
#   accuracy, mae, mse, rmse
#
# Probability metrics — y_pred is a 1D array of class probabilities.
# Pass predict_proba() output in preds_dict when using these.
# KNNBase._compute_scores detects 2D input and dispatches row-by-row automatically.
#   log_loss      -log(p[y_true])          mode='min'   continuous per-sample NLL
#   prob_correct  p[y_true]                mode='max'   probability on the correct class
metrics = {
    'accuracy':     lambda y_true, y_pred: 1 if y_true == y_pred else 0,
    'mse':          lambda y_true, y_pred: (y_true - y_pred) ** 2,
    'mae':          lambda y_true, y_pred: abs(y_true - y_pred),
    'rmse':         lambda y_true, y_pred: ((y_true - y_pred) ** 2) ** 0.5,
    'log_loss':     lambda y_true, y_pred: -math.log(max(float(y_pred[int(y_true)]), 1e-15)),
    'prob_correct': lambda y_true, y_pred: float(y_pred[int(y_true)]),
}

SPEED_PRESETS = {
    'exact': {
        'description': 'Exact nearest neighbors — slowest but 100% accurate',
        'finder': 'knn',
        'kwargs': {}
    },
    'balanced': {
        'description': 'Good balance of speed and accuracy (~98% recall)',
        'finder': 'faiss',
        'kwargs': {'index_type': 'ivf', 'n_probes': 50}
    },
    'fast': {
        'description': 'Fast queries with good accuracy (~95% recall)',
        'finder': 'faiss',
        'kwargs': {'index_type': 'ivf', 'n_probes': 30}
    },
    'turbo': {
        'description': 'Maximum speed, exact results — FAISS flat index',
        'finder': 'faiss',
        'kwargs': {'index_type': 'flat'}
    },
    'high_dim_balanced': {
        'description': 'High-dimensional data (>100D), balanced',
        'finder': 'hnsw',
        'kwargs': {'backend': 'hnswlib', 'M': 32, 'ef_construction': 400, 'ef_search': 200}
    },
    'high_dim_fast': {
        'description': 'High-dimensional data (>100D), fast',
        'finder': 'hnsw',
        'kwargs': {'backend': 'hnswlib', 'M': 16, 'ef_construction': 200, 'ef_search': 100}
    },
    'custom': {
        'description': 'Custom configuration — specify finder and parameters manually',
        'finder': None,
        'kwargs': {}
    }
}


class DynamicRouter:
    """
    Main entry point for Dynamic Ensemble Selection.

    Fits a routing model on a held-out validation set and assigns per-sample
    weights to a pool of pre-trained des at inference time.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    dtype : str
        Input data type: 'tabular' or 'image'.
    method : str
        'knn-dws' for soft blending, 'knora-u' for vote-based selection,
        'knora-e' for intersection-based selection, or 'ola' for hard selection.
    metric : str or callable
        Per-sample scoring function. Built-in names: 'accuracy', 'mae', 'mse',
        'rmse'. Or any callable with signature (y_true, y_pred) -> float.
    mode : str
        'max' if higher scores are better, 'min' if lower.
    feature_extractor : callable, optional
        Applied to inputs before neighbor search. Useful when raw inputs are
        not a meaningful feature space (e.g. images passed as pixels).
    preset : str
        Speed/accuracy preset. See SPEED_PRESETS or call list_presets().
    finder : str, optional
        Required when preset='custom'. One of: 'knn', 'faiss', 'annoy', 'hnsw'.
    k : int
        Number of neighbors per query.
    threshold : float
        knn-dw only. Models scoring below this fraction of the local best
        (after per-neighborhood normalization) are excluded from softmax.
        0.0 disables; 1.0 is equivalent to OLA. Default: 0.5.
    **kwargs
        Forwarded to the neighbor finder.
    """

    def __init__(self, task, dtype, method='knn', metric='accuracy', mode='max',
                 feature_extractor=None, preset='balanced', finder=None, k=10,
                 threshold=0.5, **kwargs):
        self.task = task.lower()
        self.dtype = dtype.lower()
        self.method = method.lower()
        self.threshold = threshold

        if isinstance(metric, str):
            if metric in metrics:
                metric = metrics[metric.lower()]
            else:
                raise ValueError(f"Unknown metric '{metric}'. Available: {list(metrics.keys())}")
        self.metric = metric
        self.mode = mode.lower()
        self.feature_extractor = feature_extractor

        if preset and preset != 'custom':
            if preset not in SPEED_PRESETS:
                raise ValueError(f"Unknown preset '{preset}'. Available: {list(SPEED_PRESETS.keys())}")
            preset_config = SPEED_PRESETS[preset]
            self.preset = preset
            self.finder = preset_config['finder']
            self.kwargs = {**preset_config['kwargs'], 'k': k, **kwargs}
            print(f"Using preset '{preset}': {preset_config['description']}")
        else:
            self.preset = 'custom'
            if finder is None:
                raise ValueError("Must specify 'finder' when using preset='custom'")
            self.finder = finder.lower()
            self.kwargs = {'k': k, **kwargs}

        self.model, model_name = self.create_model()

    @classmethod
    def from_data_size(cls, n_samples, n_features, task, dtype, method='knn-dw',
                       metric='accuracy', mode='max', k=10, threshold=0.5,
                       n_queries=None, **extra_kwargs):
        """
        Recommend and instantiate a preset based on dataset dimensions.

        Uses exact search for small or low-dimensional data, HNSW for high-
        dimensional spaces, and FAISS/Annoy for large flat datasets.

        ANN methods have higher fit cost but lower predict cost than exact KNN.
        If n_queries is provided, the recommendation accounts for this tradeoff:
        if the query volume is too low to recoup the ANN fit overhead, exact
        search is preferred even for larger datasets.

        Parameters
        ----------
        n_queries : int, optional
            Expected number of prediction calls (i.e. test set size). If None,
            the recommendation is based on n_samples alone.
        """
        if n_samples < 10000:
            preset, reason = 'exact', "Small dataset (<10K) — exact search is fast enough"
        elif n_features < 20:
            preset, reason = 'exact', "Low-dimensional (<20D) — ANN overhead not worthwhile"
        elif n_samples < 100000:
            preset, reason = 'balanced', "Medium dataset (10K–100K) — balanced speed/accuracy"
        elif n_features > 100:
            preset = 'high_dim_fast' if n_samples > 1_000_000 else 'high_dim_balanced'
            reason = "High-dimensional (>100D) — HNSW recommended"
        elif n_samples > 1_000_000:
            preset, reason = 'turbo', "Very large dataset (>1M) — prioritise speed"
        else:
            preset, reason = 'fast', "Large dataset (100K–1M) — fast with good accuracy"

        # ANN fit cost grows with n_samples; predict cost savings only materialise
        # over many queries. If n_queries is small relative to n_samples, the fit
        # overhead never pays off and exact search is faster overall.
        # Empirical crossover: ANN is worthwhile when n_queries > n_samples * 0.05.
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
            task=task, dtype=dtype, method=method, metric=metric, mode=mode,
            preset=preset, k=k, threshold=threshold,
            **extra_kwargs
        )

    @classmethod
    def list_presets(cls):
        """Print all available presets with descriptions and parameters."""
        print("\nAvailable Speed/Accuracy Presets:")
        print("=" * 70)
        for name, config in SPEED_PRESETS.items():
            if name == 'custom':
                continue
            print(f"\n{name.upper()}\n  {config['description']}\n  Finder: {config['finder']}")
            if config['kwargs']:
                print(f"  Parameters: {config['kwargs']}")
        print("\n" + "=" * 70)

    def get_config_info(self):
        """Return a dict summarising the current configuration."""
        return {
            'preset': self.preset,
            'finder': self.finder,
            'method': self.method,
            'parameters': self.kwargs,
            'threshold': self.threshold,
        }

    def create_model(self):
        """Instantiate the neighbor finder and routing model from current config."""
        if self.finder == 'knn':
            from ensemble_weights.models.neighbors import KNNNeighborFinder
            finder = KNNNeighborFinder(**self.kwargs)
        elif self.finder == 'faiss':
            from ensemble_weights.models.neighbors import FaissNeighborFinder
            finder = FaissNeighborFinder(**self.kwargs)
        elif self.finder == 'annoy':
            from ensemble_weights.models.neighbors import AnnoyNeighborFinder
            finder = AnnoyNeighborFinder(**self.kwargs)
        elif self.finder == 'hnsw':
            from ensemble_weights.models.neighbors import HNSWNeighborFinder
            finder = HNSWNeighborFinder(**self.kwargs)
        else:
            raise ValueError(f"Unknown finder: {self.finder}")

        if self.method == 'knn-dws' and self.dtype in ('tabular', 'image'):
            from ensemble_weights.models.knndws import KNNDWSModel
            return KNNDWSModel(
                metric=self.metric, mode=self.mode, neighbor_finder=finder
            ), 'KNN-DWS'

        if self.method == 'knora-u' and self.dtype in ('tabular', 'image'):
            from ensemble_weights.models.knorau import KNORAUModel
            return KNORAUModel(
                metric=self.metric, mode=self.mode, neighbor_finder=finder,
            ), 'KNORA-U'

        if self.method == 'knora-e' and self.dtype in ('tabular', 'image'):
            from ensemble_weights.models.knorae import KNORAEModel
            return KNORAEModel(
                metric=self.metric, mode=self.mode, neighbor_finder=finder,
            ), 'KNORA-E'

        if self.method == 'ola' and self.dtype in ('tabular', 'image'):
            from ensemble_weights.models.ola import OLAModel
            return OLAModel(metric=self.metric, mode=self.mode, neighbor_finder=finder), 'OLA'

        raise ValueError(f"Unsupported combination: method={self.method}, dtype={self.dtype}")

    def fit(self, features, y, preds_dict):
        """
        Fit the routing model on validation data.

        Parameters
        ----------
        features : array-like (n_val, n_features)
            Validation features. Must not overlap with train or test data.
        y : array-like (n_val,)
            Validation ground-truth labels or values.
        preds_dict : dict[str, array-like]
            Validation predictions keyed by model name.
        """
        if self.feature_extractor is not None:
            features = self.feature_extractor(features)
        features   = to_numpy(features)
        y          = to_numpy(y)
        preds_dict = {name: to_numpy(preds) for name, preds in preds_dict.items()}
        self.model.fit(features, y, preds_dict)

    def predict(self, x, temperature, threshold=0.5):
        """
        Return per-sample model weights for one or more test inputs.

        Parameters
        ----------
        x : array-like (n_features,) or (batch_size, n_features)
        temperature : float
            knn-dw only. Lower = sharper routing toward local best model;
            higher = softer blending. Recommended: 0.1 for regression, 1.0
            for classification. Ignored by OLA.

        Returns
        -------
        dict or list of dict
            Single sample: {model_name: weight}. Batch: list of such dicts.
        """
        if temperature is None:
            temperature = 0.1 if self.mode == 'min' else 1.0
        if self.feature_extractor is not None:
            x = add_batch_dim(x)
            x = self.feature_extractor(x)[0]
        x = to_numpy(x)
        return self.model.predict(x, temperature, threshold)