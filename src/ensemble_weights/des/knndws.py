"""
KNN-DWS: K-Nearest Neighbors with Distance-Weighted Softmax.
"""
from ensemble_weights.base.knnbase import KNNBase
from ensemble_weights._config import make_finder, resolve_metric, prep_fit_inputs
from ensemble_weights.utils import to_numpy
import numpy as np


class KNNDWS(KNNBase):
    """
    KNN-DWS: K-Nearest Neighbors with Distance-Weighted Softmax.

    For each test point, retrieves its K nearest validation neighbors, averages
    each model's local scores, normalizes to [0, 1] within the neighborhood,
    applies an optional competence gate, then weights models via softmax.

    This is the only algorithm here that never makes hard binary decisions —
    every step is soft and continuous. When one model clearly dominates a
    neighborhood, the softmax sharpens toward it. When models are close, weights
    spread out. This gives it natural behavior across both regimes without
    needing to choose in advance.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    metric : str or callable
        Scoring function. Use 'log_loss' or 'prob_correct' with predict_proba()
        output for classification; 'mae', 'mse', or 'rmse' for regression.
    mode : str
        'max' if higher scores are better, 'min' if lower.
    k : int
        Neighborhood size. Default: 10.
    threshold : float
        After per-neighborhood normalization (best=1.0, worst=0.0), models
        below this fraction are excluded from softmax. 0.0 disables the gate;
        1.0 reduces to OLA behavior. Default: 0.5.
    temperature : float, optional
        Softmax sharpness. Lower = sharper routing toward the local best model;
        higher = softer blending. If not set, defaults to 0.1 for regression
        (min-metrics) and 1.0 for classification (max-metrics) at predict time.
    preset : str
        Neighbor search preset. Default: 'balanced'. See list_presets().

    Examples
    --------
    Regression:

        from ensemble_weights.des.knndws import KNNDWS

        router = KNNDWS(task='regression', metric='mae', mode='min', k=20)
        router.fit(X_val, y_val, {'model_a': preds_a, 'model_b': preds_b})
        weights = router.predict(X_test)   # list of {name: weight} dicts

    Classification with probabilities:

        router = KNNDWS(task='classification', metric='log_loss', mode='min', k=20)
        router.fit(X_val, y_val, {'lr': lr_probas, 'knn': knn_probas})
        weights = router.predict(X_test)
    """

    def __init__(self, task, metric='mae', mode='min', k=10,
                 threshold=0.5, temperature=None, preset='balanced', **kwargs):
        metric_name, metric_fn = resolve_metric(metric)
        finder = make_finder(preset, k, **kwargs)
        super().__init__(metric=metric_fn, mode=mode, neighbor_finder=finder)
        self.task         = task
        self.threshold    = threshold
        self._temperature = temperature
        self._metric_name = metric_name

    def fit(self, features, y, preds_dict):
        """
        Fit the routing model on validation data.

        Parameters
        ----------
        features : array-like, shape (n_val, n_features)
            Validation features. Must not overlap with train or test data.
        y : array-like, shape (n_val,)
            Validation ground-truth labels or values.
        preds_dict : dict[str, array-like]
            Validation predictions keyed by model name.
            Shape (n_val,) for scalar metrics; (n_val, n_classes) for probability metrics.
        """
        features, y, preds_dict = prep_fit_inputs(
            features, y, preds_dict, self._metric_name
        )
        super().fit(features, y, preds_dict)

    def predict(self, x, temperature=None, threshold=None):
        """
        Return per-sample model weights.

        Parameters
        ----------
        x : array-like, shape (n_features,) or (n_samples, n_features)
        temperature : float, optional
            Overrides the instance temperature for this call.
        threshold : float, optional
            Overrides the instance threshold for this call.

        Returns
        -------
        dict or list of dict
            Single sample: {model_name: weight}. Batch: list of such dicts.
        """
        t  = temperature if temperature is not None else (
             self._temperature if self._temperature is not None else
             (0.1 if self.mode == 'min' else 1.0))
        th = threshold if threshold is not None else self.threshold

        x          = np.atleast_2d(to_numpy(x))
        batch_size = x.shape[0]

        _, indices = self.model.kneighbors(x)

        # Average each model's scores over the K neighbors: (batch, n_models)
        avg_scores = self.matrix[indices].mean(axis=1)

        # Normalize per neighborhood: local best = 1.0, worst = 0.0.
        # Uniform neighborhoods (range = 0) stay as-is.
        local_min   = avg_scores.min(axis=1, keepdims=True)
        local_max   = avg_scores.max(axis=1, keepdims=True)
        local_range = local_max - local_min
        norm_scores = (avg_scores - local_min) / np.where(local_range > 0, local_range, 1.0)

        # Competence gate: zero out models below threshold.
        # Falls back to the single best if nothing passes.
        if th > 0:
            gate      = norm_scores >= th
            any_pass  = gate.any(axis=1, keepdims=True)
            gate      = np.where(any_pass, gate, norm_scores == 1.0)
            norm_scores = norm_scores * gate

        # Numerically stable softmax; masked models cannot contribute through
        # the denominator either.
        max_scores = norm_scores.max(axis=1, keepdims=True)
        exp_scores = np.exp((norm_scores - max_scores) / t)
        if th > 0:
            exp_scores = exp_scores * gate
        weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        if batch_size == 1:
            return dict(zip(self.models, weights[0]))
        return [dict(zip(self.models, w)) for w in weights]