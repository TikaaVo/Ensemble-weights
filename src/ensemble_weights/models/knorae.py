from ensemble_weights.models.knnbase import KNNBase
import numpy as np


class KNORAEModel(KNNBase):
    """
    KNORA-E: K-Nearest Oracles — Eliminate.

    For each test point, finds the largest neighborhood (up to K neighbors) in
    which at least one model is competent on every single neighbor. Only those
    models are used, with equal weight. This is strictly more selective than
    KNORA-U, which uses any model competent on at least one neighbor.

    The "Eliminate" name reflects the search: start with all K neighbors and
    progressively eliminate the outermost neighbor until the intersection of
    competent models is non-empty, then use whatever survives.

    Fallback behavior
    -----------------
    If no model passes on all K neighbors, K is reduced to K-1 and the check
    is repeated, continuing down to K=1. At K=1, the best model on the single
    nearest neighbor always wins (or ties), so the algorithm is guaranteed to
    produce a non-empty result. If even K=1 yields no signal (all models
    equally scored, range=0), weights fall back to uniform.

    Output character
    ----------------
    KNORA-E tends to concentrate weight on fewer models than KNORA-U. When one
    model genuinely dominates the neighborhood it often receives 100% weight at
    the maximum K. In mixed neighborhoods, shrinking K finds a tighter local
    region where something is dominant. It is generally more aggressive at
    picking a winner at the cost of using fewer neighbors for the decision.
    """

    def predict(self, x, temperature=1.0, threshold=0.5):
        """
        Parameters
        ----------
        temperature : float
            Accepted for API parity with KNNDWSModel but not used — KNORA-E
            weighting is uniform among the surviving models, not softmax.
        threshold : float
            Per-neighbor competence cutoff on the [0, 1] normalized scale.
            A model is competent on a neighbor if its normalized score >=
            threshold. For classification (0/1 accuracy) any value in (0, 1]
            is equivalent. For regression, 0.5 (default) requires the model
            to be in the top half of the local performance range.
        """
        x = np.atleast_2d(x)
        batch_size = x.shape[0]
        n_models = len(self.models)

        _, indices = self.model.kneighbors(x)
        k = indices.shape[1]

        # neighbor_scores: (batch_size, k, n_models)
        neighbor_scores = self.matrix[indices]

        # Normalize per neighbor independently: best model on each neighbor = 1.0,
        # worst = 0.0. When all models are equal on a neighbor (range=0), every
        # model gets norm=0, ensuring none can pass a positive threshold there.
        n_min = neighbor_scores.min(axis=2, keepdims=True)
        n_max = neighbor_scores.max(axis=2, keepdims=True)
        n_range = n_max - n_min
        norm = (neighbor_scores - n_min) / np.where(n_range > 0, n_range, 1.0)

        # competent[b, i, j] = True if model j is competent on neighbor i for sample b.
        competent = norm >= threshold   # (batch_size, k, n_models)

        # resolved[b] = True once sample b has found a non-empty intersection.
        # weights[b] will hold the final equal-weight distribution for sample b.
        resolved = np.zeros(batch_size, dtype=bool)
        weights  = np.zeros((batch_size, n_models))

        # Shrink from K down to 1. For each level, check which unresolved samples
        # have at least one model competent on all of the first `curr_k` neighbors.
        # Resolve those samples immediately and continue with the rest.
        # Stopping early once all samples are resolved avoids unnecessary iterations.
        for curr_k in range(k, 0, -1):
            if resolved.all():
                break

            # intersection[b, j] = True if model j is competent on every one of
            # the first curr_k neighbors for sample b.
            intersection = competent[:, :curr_k, :].all(axis=1)   # (batch, n_models)

            # A sample is resolvable at this K if at least one model passes all neighbors.
            any_pass = intersection.any(axis=1)                    # (batch,)

            newly_resolved = any_pass & ~resolved
            if newly_resolved.any():
                # Equal weight among passing models; models outside intersection get 0.
                passing_counts = intersection[newly_resolved].sum(axis=1, keepdims=True)
                weights[newly_resolved] = (
                    intersection[newly_resolved].astype(float) / passing_counts
                )
                resolved |= newly_resolved

        # Uniform fallback for samples where even K=1 had no signal (all models
        # equally scored on the nearest neighbor, so norm=0 for everyone).
        if not resolved.all():
            weights[~resolved] = 1.0 / n_models

        if batch_size == 1:
            return dict(zip(self.models, weights[0]))
        return [dict(zip(self.models, w)) for w in weights]