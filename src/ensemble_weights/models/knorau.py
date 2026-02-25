from ensemble_weights.models.knnbase import KNNBase
import numpy as np


class KNORAUModel(KNNBase):
    """
    KNORA-U: K-Nearest Oracles — Union.

    For each test point, retrieves its K nearest validation neighbors and
    determines which models are "competent" on each individual neighbor.
    A model's weight is proportional to the number of neighbors it is
    competent on (its vote count). Models with zero votes are excluded.

    Unlike KNN-DWS, competence is assessed per-neighbor before any aggregation,
    and weighting is linear (vote counts) rather than exponential (softmax).
    This makes KNORA-U less sensitive to temperature tuning and more
    interpretable: a weight of 0.6 means the model was competent on 60% of
    the neighborhood.

    The "Union" in the name refers to the fact that the final ensemble is the
    union of all models that are competent on at least one neighbor. The
    complementary algorithm KNORA-E (Eliminate) takes the intersection —
    only models correct on ALL neighbors — which is much stricter and can
    reduce to a single model or an empty set.

    Works for both classification and regression:
      Classification  Per-neighbor normalization maps correct models to 1.0
                      and incorrect to 0.0; any threshold in (0, 1] selects
                      only correct models, matching the original definition.
      Regression      Per-neighbor normalization maps the best model to 1.0
                      and worst to 0.0; threshold controls how selective the
                      competence criterion is within each neighborhood.
    """

    def predict(self, x, temperature=1.0, threshold=0.5):
        """
        Parameters
        ----------
        temperature : float
            Accepted for API parity with KNNDWSModel but not used — KNORA-U
            weighting is linear (vote counts), not softmax.
        threshold : float
            Per-neighbor competence cutoff on the [0, 1] normalized scale,
            where 1.0 = best model on that neighbor, 0.0 = worst.

            For classification (0/1 accuracy): any value in (0, 1] is
            equivalent — normalized correct models always score 1.0 and
            incorrect ones score 0.0, so only correct models pass.

            For regression: 0.5 (default) requires a model to be in the top
            half of the performance range on a given neighbor to earn a vote.
            Lower values include more models; higher values are more selective.
        """
        x = np.atleast_2d(x)
        batch_size = x.shape[0]

        _, indices = self.model.kneighbors(x)

        # neighbor_scores: (batch_size, k, n_models)
        # Each entry is the score of one model on one validation neighbor.
        # The matrix is already higher-is-better (min-metrics were negated in fit).
        neighbor_scores = self.matrix[indices]

        # Normalize per neighbor so the best model on each neighbor = 1.0,
        # worst = 0.0. This makes `threshold` meaningful across different
        # datasets and metrics — it's always a fraction of the local range,
        # not an absolute score value.
        #
        # When all models score identically on a neighbor (range = 0), every
        # model gets norm = 0. With threshold > 0, none pass — that neighbor
        # contributes no votes to anyone, which is correct: there is no
        # meaningful signal about which model is better there.
        n_min = neighbor_scores.min(axis=2, keepdims=True)   # (batch, k, 1)
        n_max = neighbor_scores.max(axis=2, keepdims=True)   # (batch, k, 1)
        n_range = n_max - n_min
        norm = (neighbor_scores - n_min) / np.where(n_range > 0, n_range, 1.0)

        # competent[b, i, j] = True if model j earns a vote on neighbor i
        # for test point b.
        competent = norm >= threshold                          # (batch, k, n_models)

        # votes[b, j] = number of neighbors where model j is competent.
        # This is the core of KNORA-U: weight = vote count, not softmax.
        votes = competent.sum(axis=1).astype(float)           # (batch, n_models)

        # Normalize vote counts to produce weights that sum to 1.
        # If no model earns any votes (all neighbors had equal scores),
        # fall back to uniform weights — no useful signal in the neighborhood.
        total_votes = votes.sum(axis=1, keepdims=True)        # (batch, 1)
        any_votes = total_votes > 0
        weights = np.where(
            any_votes,
            votes / np.where(any_votes, total_votes, 1.0),
            np.full_like(votes, 1.0 / len(self.models)),
        )

        if batch_size == 1:
            return dict(zip(self.models, weights[0]))
        return [dict(zip(self.models, w)) for w in weights]