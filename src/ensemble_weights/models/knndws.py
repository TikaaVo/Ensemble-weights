from ensemble_weights.models.knnbase import KNNBase
import numpy as np


class KNNDWSModel(KNNBase):
    """
    KNN-DWS: K-Nearest Neighbors with Distance-Weighted Softmax selection.

    Retrieves the K nearest validation neighbors for each test point and
    weights models by their average local score. A competence gate excludes
    models below a threshold before softmax weighting.
    """

    def predict(self, x, temperature=1.0, threshold=0.5):
        """
        Parameters
        ----------
        temperature : float
            Softmax sharpness. Lower values route more decisively to the local
            best model; higher values produce softer blending. Recommended:
            0.1 for regression metrics, 1.0 for classification.
        threshold : float
            After per-neighborhood normalization (best=1.0, worst=0.0), models
            scoring below this fraction of the local best are excluded from the
            blend. 0.0 disables the gate; 1.0 reduces to OLA behavior.
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
        if threshold > 0:
            gate = norm_scores >= threshold
            any_pass = gate.any(axis=1, keepdims=True)
            gate = np.where(any_pass, gate, norm_scores == 1.0)
            norm_scores = norm_scores * gate

        # Numerically stable softmax; zeroed models are masked after exp so
        # they cannot contribute through the denominator.
        max_scores = norm_scores.max(axis=1, keepdims=True)
        exp_scores = np.exp((norm_scores - max_scores) / temperature)
        if threshold > 0:
            exp_scores = exp_scores * gate
        weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        if batch_size == 1:
            return dict(zip(self.models, weights[0]))
        return [dict(zip(self.models, w)) for w in weights]