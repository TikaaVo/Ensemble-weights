"""
Per-sample metric functions for Dynamic Ensemble Selection.

Each function takes a single ground-truth value and a single prediction and
returns a float score. They are passed to algorithm classes via the `metric`
parameter and called once per validation sample during fit().

Scalar metrics (y_pred is a single number)
-------------------------------------------
mae            Absolute error.                                  mode='min'
mse            Squared error.                                   mode='min'
rmse           Root squared error.                              mode='min'
accuracy       1.0 if correct, 0.0 otherwise.                  mode='max'

Probability metrics (y_pred is a 1D array of class probabilities)
------------------------------------------------------------------
Use these with predict_proba() output. Pass the probability array for one
sample as y_pred; y_true should be the integer class index.

log_loss       Negative log-probability of the true class.     mode='min'
               Continuous signal — a model that assigns 0.9 to the correct
               class scores much better than one that assigns 0.51.
prob_correct   Probability assigned to the true class.         mode='max'
               Simpler alternative to log_loss; linear rather than log-scaled.

Usage
-----
    from ensemble_weights.des.knndws import KNNDWS
    from ensemble_weights.metrics import log_loss

    router = KNNDWS(task='classification', metric=log_loss, mode='min', k=20)

You can also pass the metric name as a string and it will be resolved
automatically:

    router = KNNDWS(task='classification', metric='log_loss', mode='min', k=20)
"""
import math


def mae(y_true, y_pred):
    """Mean absolute error (per sample)."""
    return abs(y_true - y_pred)


def mse(y_true, y_pred):
    """Mean squared error (per sample)."""
    return (y_true - y_pred) ** 2


def rmse(y_true, y_pred):
    """Root mean squared error (per sample)."""
    return ((y_true - y_pred) ** 2) ** 0.5


def accuracy(y_true, y_pred):
    """1.0 if y_true == y_pred, else 0.0."""
    return 1.0 if y_true == y_pred else 0.0


def log_loss(y_true, y_pred):
    """
    Negative log-probability of the true class.

    y_pred must be a 1D array of class probabilities (e.g. from predict_proba).
    y_true must be the integer class index.
    """
    return -math.log(max(float(y_pred[int(y_true)]), 1e-15))


def prob_correct(y_true, y_pred):
    """
    Probability assigned to the true class.

    y_pred must be a 1D array of class probabilities (e.g. from predict_proba).
    y_true must be the integer class index.
    """
    return float(y_pred[int(y_true)])


# Internal registry — used by algorithm classes when metric is passed as a string.
# Not part of the public API; users should import the named functions above.
_METRICS = {
    'mae':          mae,
    'mse':          mse,
    'rmse':         rmse,
    'accuracy':     accuracy,
    'log_loss':     log_loss,
    'prob_correct': prob_correct,
}

# Used in fit() validation to check that prediction shape matches the metric.
_PROB_METRICS   = frozenset({'log_loss', 'prob_correct'})
_SCALAR_METRICS = frozenset({'mae', 'mse', 'rmse', 'accuracy'})