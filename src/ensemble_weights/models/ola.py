from ensemble_weights.models.knnbase import KNNBase
import numpy as np


class OLAModel(KNNBase):
    """
    OLA: Overall Local Accuracy.

    Assigns full weight to the single model with the highest average score
    in the K nearest validation neighbors of each test point. No blending.
    """

    def fit(self, features, y, preds_dict):
        super().fit(features, y, preds_dict)
        # Global normalization to [0, 1]. Argmax is invariant to this, but it
        # keeps the stored matrix consistent with KNNDWSModel for inspection.
        mat_min, mat_max = self.matrix.min(), self.matrix.max()
        if mat_max > mat_min:
            self.matrix = (self.matrix - mat_min) / (mat_max - mat_min)

    def predict(self, x, temperature=1.0, threshold=0.5):
        """
        temperature and threshold are accepted for API parity with KNNDWSModel
        but ignored; OLA always assigns weight 1.0 to the single locally-best model.
        """
        x = np.atleast_2d(x)
        batch_size = x.shape[0]

        _, indices = self.model.kneighbors(x)

        avg_scores   = self.matrix[indices].mean(axis=1)  # (batch_size, n_models)
        best_indices = np.argmax(avg_scores, axis=1)

        weights = np.zeros((batch_size, len(self.models)))
        weights[np.arange(batch_size), best_indices] = 1.0

        if batch_size == 1:
            return dict(zip(self.models, weights[0]))
        return [dict(zip(self.models, w)) for w in weights]