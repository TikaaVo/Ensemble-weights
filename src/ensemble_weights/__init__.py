"""
Dynamic Ensemble Selection library.

Main entry point: DynamicRouter.
Supports knn-dw (soft per-sample weighting) and OLA (hard per-sample selection),
with pluggable neighbor finders for exact and approximate search.
"""
from ensemble_weights.utils import to_numpy, add_batch_dim
import numpy as np

# Built-in per-sample metrics. All take scalar (y_true, y_pred) and return float.
# Pass any callable with the same signature to use a custom metric.
metrics = {
    'accuracy': lambda y_true, y_pred: 1 if y_true == y_pred else 0,
    'mse':      lambda y_true, y_pred: (y_true - y_pred) ** 2,
    'mae':      lambda y_true, y_pred: abs(y_true - y_pred),
    'rmse':     lambda y_true, y_pred: ((y_true - y_pred) ** 2) ** 0.5,
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
        'description': 'Maximum speed, decent accuracy (~90% recall)',
        'finder': 'annoy',
        'kwargs': {'n_trees': 100, 'search_k': -1}
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
    weights to a pool of pre-trained models at inference time.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    dtype : str
        Input data type: 'tabular' or 'image'.
    method : str
        'knn-dw' for soft blending or 'ola' for hard selection.
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
    competence_threshold : float
        knn-dw only. Models scoring below this fraction of the local best
        (after per-neighborhood normalization) are excluded from softmax.
        0.0 disables; 1.0 is equivalent to OLA. Default: 0.5.
    **kwargs
        Forwarded to the neighbor finder.
    """

    def __init__(self, task, dtype, method='knn', metric='accuracy', mode='max',
                 feature_extractor=None, preset='balanced', finder=None, k=10,
                 competence_threshold=0.5, **kwargs):
        self.task = task.lower()
        self.dtype = dtype.lower()
        self.method = method.lower()
        self.competence_threshold = competence_threshold

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
                       metric='accuracy', mode='max', k=10, competence_threshold=0.5,
                       **extra_kwargs):
        """
        Recommend and instantiate a preset based on dataset dimensions.

        Uses exact search for small or low-dimensional data, HNSW for high-
        dimensional spaces, and FAISS/Annoy for large flat datasets.
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

        print(f"Auto-selected preset: '{preset}'\nReason: {reason}")
        print(f"Data: {n_samples:,} samples, {n_features} features")

        return cls(
            task=task, dtype=dtype, method=method, metric=metric, mode=mode,
            preset=preset, k=k, competence_threshold=competence_threshold,
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
            'competence_threshold': self.competence_threshold,
        }

    def create_model(self):
        """Instantiate the neighbor finder and routing model from current config."""
        if self.finder == 'knn':
            from ensemble_weights.neighbors import KNNNeighborFinder
            finder = KNNNeighborFinder(**self.kwargs)
        elif self.finder == 'faiss':
            from ensemble_weights.neighbors import FaissNeighborFinder
            finder = FaissNeighborFinder(**self.kwargs)
        elif self.finder == 'annoy':
            from ensemble_weights.neighbors import AnnoyNeighborFinder
            finder = AnnoyNeighborFinder(**self.kwargs)
        elif self.finder == 'hnsw':
            from ensemble_weights.neighbors import HNSWNeighborFinder
            finder = HNSWNeighborFinder(**self.kwargs)
        else:
            raise ValueError(f"Unknown finder: {self.finder}")

        if self.method == 'knn-dw' and self.dtype in ('tabular', 'image'):
            from ensemble_weights.models.knn import KNNModel
            return KNNModel(
                metric=self.metric, mode=self.mode, neighbor_finder=finder,
                competence_threshold=self.competence_threshold
            ), 'KNN-DW'

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

    def predict(self, x, temperature):
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
        return self.model.predict(x, temperature)