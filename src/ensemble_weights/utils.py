import numpy as np

def to_numpy(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    elif hasattr(x, 'numpy'):
        return x.numpy()
    elif isinstance(x, (list, tuple)):
        return np.array(x)
    else:
        return np.asarray(x)


def add_batch_dim(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(0)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        if isinstance(x, tf.Tensor):
            return tf.expand_dims(x, 0)
    except ImportError:
        pass
    return np.expand_dims(x, axis=0)