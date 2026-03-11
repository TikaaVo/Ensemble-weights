"""
deskit — Dynamic Ensemble Selection library.

Metrics
-------
Pass a metric name string:

    DEWSU(task='classification', metric='log_loss', mode='min')

Or import a metric function directly:

    from deskit.metrics import log_loss, mae

    DEWSU(task='classification', metric=log_loss, mode='min')

Available built-in metrics:
    Scalar predictions (pass predict() output):
        'mae', 'mse', 'rmse', 'accuracy'

    Probability predictions (pass predict_proba() output):
        'log_loss', 'prob_correct'
"""

from deskit.des.dewsu   import DEWSU
from deskit.des.ola      import OLA
from deskit.des.knorau   import KNORAU
from deskit.des.knorae   import KNORAE
from deskit.des.knoraiu import KNORAIU
from deskit.router       import DynamicRouter
from deskit._config      import SPEED_PRESETS, list_presets

__all__ = [
    'DEWSU',
    'OLA',
    'KNORAU',
    'KNORAE',
    'KNORAIU',
    'DynamicRouter',
    'SPEED_PRESETS',
    'list_presets'
]