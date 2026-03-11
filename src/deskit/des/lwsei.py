"""
LWSE-I: Locally Weighted Stacking Ensemble (Inverse-distance).
"""
from deskit._config import make_finder
from deskit.utils import to_numpy
from scipy.optimize import nnls
import numpy as np


class LWSEI:
    """
    LWSE-I: Locally Weighted Stacking Ensemble (Inverse-distance).
    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    k : int
        Neighbourhood size. Default: 10.
    preset : str
        Neighbour search preset. Default: 'balanced'. See list_presets().
    """

    def __init__(self, task, k=10, preset='balanced', **kwargs):
        self.task = task
        self.k = k
        self._finder = make_finder(preset, k, **kwargs)
        self.models = None

        self._val_preds = None   # (n_val, n_models) regression
                                 # (n_val, n_models, n_classes) classification
        self._y_val = None   # (n_val,) true values / class indices
        self._y_onehot = None   # (n_val, n_classes) for classification
        self._is_proba = None   # True if predictions are probability arrays

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
            Shape (n_val,) for regression; (n_val, n_classes) for
            classification with probability output.
        """
        features = np.asarray(features, dtype=float)
        y = np.asarray(y)

        self.models = list(preds_dict.keys())
        first = np.asarray(list(preds_dict.values())[0])
        self._is_proba = (first.ndim == 2)

        # Stack predictions into a single matrix for fast neighbor indexing.
        if self._is_proba:
            self._val_preds = np.stack(
                [np.asarray(preds_dict[m], dtype=float) for m in self.models],
                axis=1
            )  # (n_val, n_models, n_classes)
            n_val, _, n_classes = self._val_preds.shape
            self._y_onehot = np.zeros((n_val, n_classes), dtype=float)
            self._y_onehot[np.arange(n_val), y.astype(int)] = 1.0
        else:
            self._val_preds = np.stack(
                [np.asarray(preds_dict[m], dtype=float) for m in self.models],
                axis=1
            )  # (n_val, n_models)

        self._y_val = y
        self._finder.fit(features)

    def predict(self, x, **kwargs):
        """
        Return per-sample model weights.

        Parameters
        ----------
        x : array-like, shape (n_features,) or (n_samples, n_features)

        Returns
        -------
        dict or list of dict
            Single sample: {model_name: weight}. Batch: list of such dicts.
        """
        x = np.atleast_2d(to_numpy(x))
        batch_size = x.shape[0]
        n_models = len(self.models)
        uniform = np.full(n_models, 1.0 / n_models)

        distances, indices = self._finder.kneighbors(x)   # (batch, k)

        results = []
        for b in range(batch_size):
            idx = indices[b]                              # (k,)
            dist = distances[b]                            # (k,)

            # Inverse-distance weights
            inv_dist = 1.0 / np.maximum(dist, 1e-8)
            w = inv_dist / inv_dist.sum()           # (k,)
            sqrt_w = np.sqrt(w)                          # (k,)

            if self._is_proba:
                # P: (k, n_models, n_classes) becomes (k*n_classes, n_models)
                P = self._val_preds[idx]             # (k, n_models, n_classes)
                k_, _, n_classes = P.shape
                P_flat = P.transpose(0, 2, 1).reshape(k_ * n_classes, n_models)
                y_flat = self._y_onehot[idx].reshape(k_ * n_classes)
                sqrt_wt = np.repeat(sqrt_w, n_classes)     # (k*n_classes,)
                P_wls = P_flat  * sqrt_wt[:, np.newaxis]
                y_wls = y_flat  * sqrt_wt
            else:
                P = self._val_preds[idx]               # (k, n_models)
                y_nbr = self._y_val[idx]                   # (k,)
                P_wls = P     * sqrt_w[:, np.newaxis]
                y_wls = y_nbr * sqrt_w

            # Solve
            lambda_ = 1e-6
            P_aug = np.vstack([P_wls, lambda_ * np.eye(n_models)])
            y_aug = np.concatenate([y_wls, np.zeros(n_models)])
            coeffs, _ = nnls(P_aug, y_aug, max_iter=10*n_models)

            # Normalize and fall back to uniform if degenerate.
            total = coeffs.sum()
            if total > 1e-10:
                coeffs = coeffs / total
            else:
                coeffs = uniform.copy()

            results.append(dict(zip(self.models, coeffs)))

        if batch_size == 1:
            return results[0]
        return results