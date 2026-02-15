"""
Enhanced DynamicRouter with configurable speed/accuracy tradeoff presets.
Updated with bug fixes for better recall across all methods.
"""
from ensemble_weights.utils import to_numpy, add_batch_dim
import numpy as np

metrics = {
    'accuracy': lambda y_true, y_pred: 1 if y_true == y_pred else 0,
    'mse': lambda y_true, y_pred: (y_true - y_pred) ** 2,
    'mae': lambda y_true, y_pred: abs(y_true - y_pred),
    'rmse': lambda y_true, y_pred: ((y_true - y_pred) ** 2) ** 0.5
}

SPEED_PRESETS = {
    'exact': {
        'description': 'Exact nearest neighbors - slowest but 100% accurate',
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
        'description': 'Best for high-dimensional data (>100D), balanced',
        'finder': 'hnsw',
        'kwargs': {
            'backend': 'hnswlib',
            'M': 32,
            'ef_construction': 400,
            'ef_search': 200
        }
    },
    'high_dim_fast': {
        'description': 'Best for high-dimensional data (>100D), fast',
        'finder': 'hnsw',
        'kwargs': {
            'backend': 'hnswlib',
            'M': 16,
            'ef_construction': 200,
            'ef_search': 100
        }
    },
    'custom': {
        'description': 'Custom configuration - specify your own parameters',
        'finder': None,
        'kwargs': {}
    }
}


class DynamicRouter:
    def __init__(self, task, dtype, method='knn', metric="accuracy", mode="max",
                 feature_extractor=None, preset='balanced', finder=None, k=10, **kwargs):
        self.task = task.lower()
        self.dtype = dtype.lower()
        self.method = method.lower()

        # Handle metric
        if isinstance(metric, str):
            if metric in metrics:
                metric = metrics[metric.lower()]
            else:
                raise ValueError(f"Unknown metric name '{metric}'. Available: {list(metrics.keys())}")
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
            # Manual configuration
            self.preset = 'custom'
            if finder is None:
                raise ValueError("Must specify 'finder' when using preset='custom'")
            self.finder = finder.lower()
            self.kwargs = {'k': k, **kwargs}

        self.model, model_name = self.create_model()

    @classmethod
    def from_data_size(cls, n_samples, n_features, task, dtype, method='knn-dw',
                       metric='accuracy', mode='max', k=10, **extra_kwargs):
        """
        Automatically recommend configuration based on data characteristics.

        Parameters
        ----------
        n_samples : int
            Number of samples in training data
        n_features : int
            Number of features/dimensions
        task, dtype, method, metric, mode, k : same as __init__

        Returns
        -------
        DynamicRouter configured with recommended settings
        """
        if n_samples < 10000:
            preset = 'exact'
            reason = "Small dataset (<10K samples) - exact search is fast enough"
        elif n_features < 20:
            preset = 'exact'
            reason = "Low dimensional (<20D) - ANN methods have high overhead and potential precision issues"
        elif n_samples < 100000:
            preset = 'balanced'
            reason = "Medium dataset (10K-100K samples) - balanced speed/accuracy"
        elif n_features > 100:
            if n_samples > 1000000:
                preset = 'high_dim_fast'
                reason = "Very large high-dimensional dataset - maximum speed"
            else:
                preset = 'high_dim_balanced'
                reason = "High-dimensional data (>100D) - HNSW is best"
        elif n_samples > 1000000:
            preset = 'turbo'
            reason = "Very large dataset (>1M samples) - prioritize speed"
        else:
            preset = 'fast'
            reason = "Large dataset (100K-1M samples) - fast with good accuracy"

        print(f"Auto-selected preset: '{preset}'")
        print(f"Reason: {reason}")
        print(f"Data: {n_samples:,} samples, {n_features} features")

        return cls(
            task=task,
            dtype=dtype,
            method=method,
            metric=metric,
            mode=mode,
            preset=preset,
            k=k,
            **extra_kwargs
        )

    @classmethod
    def list_presets(cls):
        """Print all available presets with descriptions."""
        print("\nAvailable Speed/Accuracy Presets:")
        print("=" * 70)
        for name, config in SPEED_PRESETS.items():
            if name == 'custom':
                continue
            print(f"\n{name.upper()}")
            print(f"  {config['description']}")
            print(f"  Finder: {config['finder']}")
            if config['kwargs']:
                print(f"  Parameters: {config['kwargs']}")
        print("\n" + "=" * 70)

    def get_config_info(self):
        """Get information about current configuration."""
        return {
            'preset': self.preset,
            'finder': self.finder,
            'method': self.method,
            'parameters': self.kwargs
        }

    def create_model(self):
        # Create neighbor finder based on configuration
        if self.finder == "knn":
            from ensemble_weights.neighbors import KNNNeighborFinder
            finder = KNNNeighborFinder(**self.kwargs)
        elif self.finder == "faiss":
            from ensemble_weights.neighbors import FaissNeighborFinder
            finder = FaissNeighborFinder(**self.kwargs)
        elif self.finder == "annoy":
            from ensemble_weights.neighbors import AnnoyNeighborFinder
            finder = AnnoyNeighborFinder(**self.kwargs)
        elif self.finder == "hnsw":
            from ensemble_weights.neighbors import HNSWNeighborFinder
            finder = HNSWNeighborFinder(**self.kwargs)
        else:
            raise ValueError(f"Unknown finder: {self.finder}")

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