from ensemble_weights.models.base import BaseRouter
import numpy as np


class KNNBase(BaseRouter):
    """
    Shared base for KNN-based DES algorithms.

    Handles score matrix construction and neighbor index fitting.
    Subclasses implement predict() with their own selection or weighting logic,
    and may override fit() to add algorithm-specific post-processing
    (e.g. OLA normalizes globally after the base fit).

    The difference between most DES algorithms (KNORA-E, KNORA-U, OLA, KNN-DWS,
    META-DES, etc.) is purely in how they use the score matrix at predict time,
    or in additional fit-time bookkeeping. The core loop — compute per-sample
    scores, store in matrix, fit neighbor index — is always the same.
    """

    def __init__(self, metric, mode='max', neighbor_finder=None):
        """
        Parameters
        ----------
        metric : callable
            Per-sample scoring function: (y_true, y_pred) -> float.
        mode : str
            'max' if higher scores are better, 'min' if lower.
        neighbor_finder : NeighborFinder
            Backend used for neighborhood queries.
        """
        self.metric = metric
        self.mode = mode
        self.model = neighbor_finder
        self.matrix = None   # (n_val, n_models) score matrix; higher is always better
        self.models = None   # ordered list of model names
        self.features = None

    def _compute_scores(self, y, preds):
        """Return a 1D array of per-sample metric scores."""
        return np.vectorize(self.metric)(y, preds)

    def fit(self, features, y, preds_dict):
        """
        Build the score matrix and fit the neighbor index.

        Scores are negated for minimization metrics so the matrix is always
        higher-is-better, regardless of the underlying metric direction.
        """
        self.features = features
        self.models = list(preds_dict.keys())
        n_val, n_models = len(y), len(self.models)
        self.matrix = np.zeros((n_val, n_models))

        for j, name in enumerate(self.models):
            scores = self._compute_scores(y, preds_dict[name])
            self.matrix[:, j] = scores if self.mode == 'max' else -scores

        self.model.fit(features)