from ensemble_weights.base import BaseRouter
import numpy as np


class KNNModel(BaseRouter):
    """
    KNN-DW: distance-weighted Dynamic Ensemble Selection.

    Retrieves the K nearest validation neighbors for each test point and
    weights models by their average local score. A competence gate excludes
    models below a threshold before softmax weighting.
    """

    def __init__(self, metric, mode='max', neighbor_finder=None,
                 competence_threshold=0.5):
        """
        Parameters
        ----------
        metric : callable
            Per-sample scoring function: (y_true, y_pred) -> float.
        mode : str
            'max' if higher scores are better, 'min' if lower.
        neighbor_finder : NeighborFinder
            Backend used for neighborhood queries.
        competence_threshold : float
            After per-neighborhood normalization (best=1.0, worst=0.0), models
            scoring below this fraction of the local best are excluded from the
            blend. 0.0 disables the gate; 1.0 reduces to OLA behavior.
        """
        self.metric = metric
        self.mode = mode
        self.model = neighbor_finder
        self.competence_threshold = competence_threshold
        self.matrix = None
        self.models = None
        self.features = None

    def _compute_scores(self, y, preds):
        """Return a 1D array of per-sample metric scores."""
        return np.vectorize(self.metric)(y, preds)

    def fit(self, features, y, preds_dict):
        self.features = features
        self.models = list(preds_dict.keys())
        n_val, n_models = len(y), len(self.models)
        self.matrix = np.zeros((n_val, n_models))

        for j, name in enumerate(self.models):
            scores = self._compute_scores(y, preds_dict[name])
            # Negate minimization metrics so the matrix is always higher-is-better.
            self.matrix[:, j] = scores if self.mode == 'max' else -scores

        self.model.fit(features)

    def predict(self, x, temperature=1.0):
        """
        Parameters
        ----------
        temperature : float
            Softmax sharpness. Lower values route more decisively to the local
            best model; higher values produce softer blending. Default 0.1 for
            regression metrics; 1.0 may suit classification.
        """
        x = np.atleast_2d(x)
        batch_size = x.shape[0]

        _, indices = self.model.kneighbors(x)

        # Average each model's scores across the K neighbors: (batch_size, n_models)
        avg_scores = self.matrix[indices].mean(axis=1)

        # Normalize per neighborhood so the local best = 1.0, worst = 0.0.
        # This exposes within-neighborhood contrast to softmax independent of
        # the metric's absolute scale. Uniform neighborhoods (range=0) stay as-is.
        local_min = avg_scores.min(axis=1, keepdims=True)
        local_max = avg_scores.max(axis=1, keepdims=True)
        local_range = local_max - local_min
        norm_scores = (avg_scores - local_min) / np.where(local_range > 0, local_range, 1.0)

        # Zero out models below the competence threshold before softmax.
        # Falls back to the single best if nothing passes (guard for edge cases).
        if self.competence_threshold > 0:
            gate = norm_scores >= self.competence_threshold
            any_pass = gate.any(axis=1, keepdims=True)
            gate = np.where(any_pass, gate, norm_scores == 1.0)
            norm_scores = norm_scores * gate

        # Numerically stable softmax; zeroed models are masked after exp so
        # they cannot contribute through the denominator.
        max_scores = norm_scores.max(axis=1, keepdims=True)
        exp_scores = np.exp((norm_scores - max_scores) / temperature)
        if self.competence_threshold > 0:
            exp_scores = exp_scores * gate
        weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        if batch_size == 1:
            return dict(zip(self.models, weights[0]))
        return [dict(zip(self.models, w)) for w in weights]