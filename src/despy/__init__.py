"""
despy — Dynamic Ensemble Selection library.

Metrics
-------
Pass a metric name string:

    KNNDWS(task='classification', metric='log_loss', mode='min')

Or import a metric function directly:

    from despy.metrics import log_loss, mae

    KNNDWS(task='classification', metric=log_loss, mode='min')

Available built-in metrics:
    Scalar predictions (pass predict() output):
        'mae', 'mse', 'rmse', 'accuracy'

    Probability predictions (pass predict_proba() output):
        'log_loss', 'prob_correct'
"""

from despy.des.knndws   import KNNDWS
from despy.des.ola      import OLA
from despy.des.knorau   import KNORAU
from despy.des.knorae   import KNORAE
from despy.des.knoraiu import KNORAIU
from despy.router       import DynamicRouter
from despy._config      import SPEED_PRESETS, list_presets
from despy.analysis     import analyze

__all__ = [
    'KNNDWS',
    'OLA',
    'KNORAU',
    'KNORAE',
    'KNORAIU',
    'DynamicRouter',
    'SPEED_PRESETS',
    'list_presets',
    'analyze',
]