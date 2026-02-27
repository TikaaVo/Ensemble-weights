"""
ensemble_weights — Dynamic Ensemble Selection library.

Recommended usage
-----------------
Import algorithm classes directly from their modules:

    from ensemble_weights.des.knndws import KNNDWS
    from ensemble_weights.des.ola    import OLA
    from ensemble_weights.des.knorau import KNORAU
    from ensemble_weights.des.knorae import KNORAE

Or import them from the top level:

    from ensemble_weights import KNNDWS, OLA, KNORAU, KNORAE

For benchmark loops where you need to select an algorithm by string:

    from ensemble_weights import DynamicRouter

    for method in ['knn-dws', 'ola', 'knora-u', 'knora-e']:
        router = DynamicRouter(task='classification', method=method, ...)

Metrics
-------
Pass a metric name string and it is resolved automatically:

    KNNDWS(task='classification', metric='log_loss', mode='min')

Or import a metric function directly:

    from ensemble_weights.metrics import log_loss, mae

    KNNDWS(task='classification', metric=log_loss, mode='min')

Available built-in metrics:
    Scalar predictions (pass predict() output):
        'mae', 'mse', 'rmse', 'accuracy'

    Probability predictions (pass predict_proba() output):
        'log_loss', 'prob_correct'
"""

from ensemble_weights.des.knndws   import KNNDWS
from ensemble_weights.des.ola      import OLA
from ensemble_weights.des.knorau   import KNORAU
from ensemble_weights.des.knorae   import KNORAE
from ensemble_weights.des.knoraiu import KNORAIU
from ensemble_weights.router       import DynamicRouter
from ensemble_weights._config      import SPEED_PRESETS, list_presets

__all__ = [
    'KNNDWS',
    'OLA',
    'KNORAU',
    'KNORAE',
    'KNORAIU',
    'DynamicRouter',
    'SPEED_PRESETS',
    'list_presets',
]