"""
LWSE-U: Locally Weighted Stacking Ensemble (Uniform).
"""
from deskit._config import make_finder
from deskit.utils import to_numpy
from scipy.optimize import nnls
import numpy as np


class LWSEU:
    """
    LWSE-U: Locally Weighted Stacking Ensemble (Uniform).

    For each test point, finds its K nearest neighbours on the validation set
    and fits a non-negative least squares combination of the pool models
    directly on those K points. All neighbours are treated with equal weight
    in the NNLS fit — no distance scaling is applied.

    This is the ablation baseline for LWSE-I. Comparing the two isolates
    the contribution of inverse-distance row-weighting in the local fit.

    NNLS naturally produces sparse solutions — models that do not contribute
    to the local fit are zeroed out without needing a threshold gate.

    Works with both regression (scalar predictions) and classification
    (probability arrays).

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
        self.task    = task
        self.k       = k
        self._finder = make_finder(preset, k, **kwargs)
        self.models  = None

        self._val_preds = None
        self._y_val     = None
        self._y_onehot  = None
        self._is_proba  = None

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
        y        = np.asarray(y)

        self.models  = list(preds_dict.keys())
        first        = np.asarray(list(preds_dict.values())[0])
        self._is_proba = (first.ndim == 2)

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
        x          = np.atleast_2d(to_numpy(x))
        batch_size = x.shape[0]
        n_models   = len(self.models)
        uniform    = np.full(n_models, 1.0 / n_models)

        distances, indices = self._finder.kneighbors(x)   # (batch, k)

        results = []
        for b in range(batch_size):
            idx = indices[b]                               # (k,)

            if self._is_proba:
                P       = self._val_preds[idx]             # (k, n_models, n_classes)
                k_, _, n_classes = P.shape
                P_flat  = P.transpose(0, 2, 1).reshape(k_ * n_classes, n_models)
                y_flat  = self._y_onehot[idx].reshape(k_ * n_classes)
                coeffs, _ = nnls(P_flat, y_flat)
            else:
                P     = self._val_preds[idx]               # (k, n_models)
                y_nbr = self._y_val[idx]                   # (k,)
                coeffs, _ = nnls(P, y_nbr)

            total = coeffs.sum()
            if total > 1e-10:
                coeffs = coeffs / total
            else:
                coeffs = uniform.copy()

            results.append(dict(zip(self.models, coeffs)))

        if batch_size == 1:
            return results[0]
        return results