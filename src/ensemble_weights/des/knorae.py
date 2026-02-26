"""
KNORAE: K-Nearest Oracles — Eliminate.
"""
from ensemble_weights.base.knnbase import KNNBase
from ensemble_weights._config import make_finder, resolve_metric, prep_fit_inputs
from ensemble_weights.utils import to_numpy
import numpy as np


class KNORAE(KNNBase):
    """
    KNORAE: K-Nearest Oracles — Eliminate.

    For each test point, finds the largest neighborhood (up to K neighbors) in
    which at least one model is competent on every single neighbor. Only those
    models contribute, with equal weight.

    The "Eliminate" name reflects the search: start with all K neighbors and
    progressively shrink until the intersection of competent models is non-empty.
    At K=1, the best model on the nearest neighbor always wins, guaranteeing a
    result. If even K=1 yields no signal (all models equally scored), weights
    fall back to uniform.

    More aggressive than KNORAU — tends to concentrate weight on a single
    model. Performs well when one model genuinely dominates a tight local
    region; tends to underperform on noisy datasets or when no model is
    regionally dominant.

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
        Per-neighbor competence cutoff on the [0, 1] normalized scale.
        Classification with log_loss: 0.5 (default).
        Regression: use 1.0.
    preset : str
        Neighbor search preset. Default: 'balanced'. See list_presets().

    Examples
    --------
        from ensemble_weights.des.knorae import KNORAE

        router = KNORAE(task='classification', metric='log_loss', mode='min', k=20)
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
            Accepted for API compatibility; KNORA-E weights surviving models
            equally, not via softmax, so temperature has no effect.
        threshold : float, optional
            Overrides the instance threshold for this call.

        Returns
        -------
        dict or list of dict
            Single sample: {model_name: weight}. Batch: list of such dicts.
            The surviving models share weight equally.
        """
        th = threshold if threshold is not None else self.threshold

        x          = np.atleast_2d(to_numpy(x))
        batch_size = x.shape[0]
        n_models   = len(self.models)

        _, indices      = self.model.kneighbors(x)
        k               = indices.shape[1]
        neighbor_scores = self.matrix[indices]   # (batch, k, n_models)

        # Normalize per neighbor independently.
        # Range = 0 neighbors give every model norm = 0, so none pass a positive
        # threshold — correct, since identical scores carry no routing signal.
        n_min   = neighbor_scores.min(axis=2, keepdims=True)
        n_max   = neighbor_scores.max(axis=2, keepdims=True)
        n_range = n_max - n_min
        norm    = (neighbor_scores - n_min) / np.where(n_range > 0, n_range, 1.0)

        competent = norm >= th   # (batch, k, n_models)
        resolved  = np.zeros(batch_size, dtype=bool)
        weights   = np.zeros((batch_size, n_models))

        # Shrink from K down to 1. Resolve samples as soon as a non-empty
        # intersection is found. Stop early once all samples are resolved.
        for curr_k in range(k, 0, -1):
            if resolved.all():
                break

            # intersection[b, j] = True if model j is competent on all curr_k neighbors.
            intersection    = competent[:, :curr_k, :].all(axis=1)   # (batch, n_models)
            any_pass        = intersection.any(axis=1)                # (batch,)
            newly_resolved  = any_pass & ~resolved

            if newly_resolved.any():
                counts = intersection[newly_resolved].sum(axis=1, keepdims=True)
                weights[newly_resolved] = (
                    intersection[newly_resolved].astype(float) / counts
                )
                resolved |= newly_resolved

        # Uniform fallback for samples where K=1 had no signal.
        if not resolved.all():
            weights[~resolved] = 1.0 / n_models

        if batch_size == 1:
            return dict(zip(self.models, weights[0]))
        return [dict(zip(self.models, w)) for w in weights]