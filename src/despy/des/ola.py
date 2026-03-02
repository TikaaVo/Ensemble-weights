"""
OLA: Overall Local Accuracy.
"""
from despy.base.knnbase import KNNBase
from despy._config import make_finder, resolve_metric, prep_fit_inputs
from despy.utils import to_numpy
import numpy as np


class OLA(KNNBase):
    """
    OLA: Overall Local Accuracy.
    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    metric : str or callable
        Scoring function.
    mode : str
        'max' if higher scores are better, 'min' if lower.
    k : int
        Neighborhood size. Default: 10.
    preset : str
        Neighbor search preset. Default: 'balanced'. See list_presets().
    """

    def __init__(self, task, metric='mae', mode='min', k=10,
                 preset='balanced', threshold=None, **kwargs):
        metric_name, metric_fn = resolve_metric(metric)
        finder = make_finder(preset, k, **kwargs)
        super().__init__(metric=metric_fn, mode=mode, neighbor_finder=finder)
        self.task         = task
        self._metric_name = metric_name

    def fit(self, features, y, preds_dict):
        """
        Fit the routing model on validation data.

        Parameters
        ----------
        features : array-like, shape (n_val, n_features)
        y : array-like, shape (n_val,)
        preds_dict : dict[str, array-like]
        """
        features, y, preds_dict = prep_fit_inputs(
            features, y, preds_dict, self._metric_name
        )
        super().fit(features, y, preds_dict)
        # Global normalization to [0, 1]
        mat_min, mat_max = self.matrix.min(), self.matrix.max()
        if mat_max > mat_min:
            self.matrix = (self.matrix - mat_min) / (mat_max - mat_min)

    def predict(self, x, temperature=None, threshold=None):
        """
        Return per-sample model weights.

        Parameters
        ----------
        x : array-like, shape (n_features,) or (n_samples, n_features)
        temperature : ignored
            Accepted for API compatibility with other algorithms; OLA always
            selects a single model and does not use softmax.
        threshold : ignored
            Accepted for API compatibility; OLA uses argmax, not a gate.

        Returns
        -------
        dict or list of dict
            Single sample: {model_name: weight}. Batch: list of such dicts.
            The selected model always gets weight 1.0; all others get 0.0.
        """
        x          = np.atleast_2d(to_numpy(x))
        batch_size = x.shape[0]

        _, indices   = self.model.kneighbors(x)
        avg_scores   = self.matrix[indices].mean(axis=1)
        best_indices = np.argmax(avg_scores, axis=1)

        weights = np.zeros((batch_size, len(self.models)))
        weights[np.arange(batch_size), best_indices] = 1.0

        if batch_size == 1:
            return dict(zip(self.models, weights[0]))
        return [dict(zip(self.models, w)) for w in weights]