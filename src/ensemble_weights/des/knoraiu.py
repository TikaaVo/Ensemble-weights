"""
KNORA-IU: K-Nearest Oracles — Inverse-weighted Union.
"""
from ensemble_weights.base.knnbase import KNNBase
from ensemble_weights._config import make_finder, resolve_metric, prep_fit_inputs
from ensemble_weights.utils import to_numpy
import numpy as np


class KNORAIU(KNNBase):
    """
    KNORA-IU: K-Nearest Oracles — Inverse-weighted Union.

    A distance-aware variant of KNORA-U. Each competent neighbor casts a vote
    weighted by the inverse of its distance to the test point rather than a
    uniform 1. Closer neighbors therefore have more influence on which models
    are selected.

    In KNORA-U a neighbor 2 units away and one 20 units away contribute equally
    if both are above the competence threshold. KNORA-IU down-weights the
    distant neighbor by a factor of 10, concentrating decisions on the most
    relevant part of the validation set.

    This tends to help in datasets with strong local structure and hurts when
    neighborhoods are dense and homogeneous (where distance differences carry
    little meaning). In practice it is a modest refinement over KNORA-U rather
    than a large improvement.

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
        from ensemble_weights.des.knora_iu import KNORAIU

        router = KNORAIU(task='classification', metric='log_loss', mode='min', k=20)
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
            Accepted for API compatibility; KNORA-IU uses inverse-distance
            weighted vote counts, not softmax.
        threshold : float, optional
            Overrides the instance threshold for this call.

        Returns
        -------
        dict or list of dict
            Single sample: {model_name: weight}. Batch: list of such dicts.
            Weights are proportional to distance-weighted vote sums and sum to 1.
        """
        th = threshold if threshold is not None else self.threshold

        x          = np.atleast_2d(to_numpy(x))
        batch_size = x.shape[0]

        distances, indices = self.model.kneighbors(x)   # both (batch, k)
        neighbor_scores    = self.matrix[indices]        # (batch, k, n_models)

        # Normalize per neighbor: best model = 1.0, worst = 0.0.
        # When all models score equally (range = 0), every model gets norm = 0,
        # so no one earns a vote — there is no signal to route on.
        n_min   = neighbor_scores.min(axis=2, keepdims=True)
        n_max   = neighbor_scores.max(axis=2, keepdims=True)
        n_range = n_max - n_min
        norm    = (neighbor_scores - n_min) / np.where(n_range > 0, n_range, 1.0)

        # competent[b, i, j] = True if model j passes the threshold on neighbor i.
        competent = norm >= th                         # (batch, k, n_models)

        # Inverse distance weights: closer neighbors have more influence.
        # Clamp denominators to 1e-8 to avoid division by zero for exact matches.
        inv_dist = 1.0 / np.maximum(distances, 1e-8)  # (batch, k)

        # Weighted votes: for each model, sum the inverse distances of its
        # competent neighbors. This is the only difference from KNORA-U, which
        # uses uniform weights (equivalently inv_dist = ones).
        votes = (competent * inv_dist[:, :, np.newaxis]).sum(axis=1)  # (batch, n_models)
        total = votes.sum(axis=1, keepdims=True)

        # Normalize to weights that sum to 1.
        # Uniform fallback if no model earned any votes.
        any_votes = total > 0
        weights   = np.where(
            any_votes,
            votes / np.where(any_votes, total, 1.0),
            np.full_like(votes, 1.0 / len(self.models)),
        )

        if batch_size == 1:
            return dict(zip(self.models, weights[0]))
        return [dict(zip(self.models, w)) for w in weights]