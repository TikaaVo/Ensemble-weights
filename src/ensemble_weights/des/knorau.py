"""
KNORA-U: K-Nearest Oracles — Union.
"""
from ensemble_weights.base.knnbase import KNNBase
from ensemble_weights._config import make_finder, resolve_metric, prep_fit_inputs
from ensemble_weights.utils import to_numpy
import numpy as np


class KNORAU(KNNBase):
    """
    KNORA-U: K-Nearest Oracles — Union.

    For each test point, counts how many of its K nearest validation neighbors
    each model is competent on. Weight is proportional to vote count (linear,
    not softmax). Models with zero votes are excluded.

    The "Union" refers to the fact that the final ensemble is the union of all
    models competent on at least one neighbor. The complementary algorithm,
    KNORAE, takes the intersection — only models competent on every neighbor.

    Works best for classification with probability metrics (log_loss,
    prob_correct), where per-neighbor normalization creates a continuous
    competence scale. For regression, use threshold=1.0 to recover the binary
    oracle criterion — at 0.5, per-neighbor normalization can inflate the
    apparent competence of weaker models.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    metric : str or callable
        Recommended: 'log_loss' for classification.
    mode : str
        'max' if higher scores are better, 'min' if lower.
    k : int
        Neighborhood size. Default: 10.
    threshold : float
        Per-neighbor competence cutoff on the [0, 1] normalized scale
        (1.0 = best model on that neighbor, 0.0 = worst).
        Classification with log_loss: 0.5 (default).
        Regression: use 1.0.
    preset : str
        Neighbor search preset. Default: 'balanced'. See list_presets().

    Examples
    --------
        from ensemble_weights.des.knorau import KNORAU

        router = KNORAU(task='classification', metric='log_loss', mode='min', k=20)
        router.fit(X_val, y_val, probas_dict)
        weights = router.predict(X_test)
    """

    def __init__(self, task, metric='mae', mode='min', k=10,
                 threshold=0.5, preset='balanced', **kwargs):
        metric_name, metric_fn = resolve_metric(metric)
        finder = make_finder(preset, k, **kwargs)
        super().__init__(metric=metric_fn, mode=mode, neighbor_finder=finder)
        self.task         = task
        self.threshold    = threshold
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

    def predict(self, x, temperature=None, threshold=None):
        """
        Return per-sample model weights.

        Parameters
        ----------
        x : array-like, shape (n_features,) or (n_samples, n_features)
        temperature : ignored
            Accepted for API compatibility; KNORA-U uses linear vote counts,
            not softmax, so temperature has no effect.
        threshold : float, optional
            Overrides the instance threshold for this call.

        Returns
        -------
        dict or list of dict
            Single sample: {model_name: weight}. Batch: list of such dicts.
            Weights are proportional to vote counts and sum to 1.
        """
        th = threshold if threshold is not None else self.threshold

        x          = np.atleast_2d(to_numpy(x))
        batch_size = x.shape[0]

        _, indices      = self.model.kneighbors(x)
        neighbor_scores = self.matrix[indices]   # (batch, k, n_models)

        # Normalize per neighbor: best model on each neighbor = 1.0, worst = 0.0.
        # When all models score equally (range = 0), every model gets norm = 0,
        # so no one earns a vote — correct, since there is no signal to route on.
        n_min   = neighbor_scores.min(axis=2, keepdims=True)
        n_max   = neighbor_scores.max(axis=2, keepdims=True)
        n_range = n_max - n_min
        norm    = (neighbor_scores - n_min) / np.where(n_range > 0, n_range, 1.0)

        # votes[b, j] = number of neighbors where model j exceeds the threshold.
        votes       = (norm >= th).sum(axis=1).astype(float)   # (batch, n_models)
        total_votes = votes.sum(axis=1, keepdims=True)

        # Normalize vote counts to weights that sum to 1.
        # If no model earns any votes, fall back to uniform weights.
        any_votes = total_votes > 0
        weights   = np.where(
            any_votes,
            votes / np.where(any_votes, total_votes, 1.0),
            np.full_like(votes, 1.0 / len(self.models)),
        )

        if batch_size == 1:
            return dict(zip(self.models, weights[0]))
        return [dict(zip(self.models, w)) for w in weights]