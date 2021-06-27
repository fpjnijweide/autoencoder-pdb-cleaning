import numpy as np
from tensorflow import keras as keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def JSD(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = keras.backend.clip(y_true, keras.backend.epsilon(), 1)
    y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1)
    means = 0.5 * (y_true + y_pred)
    divergence_tensor = 0.5 * keras.losses.kld(y_true, means) + 0.5 * keras.losses.kld(y_pred, means)
    return divergence_tensor


def prob_distance(x, y):
    res = JSD(x, y)
    return res


def generate_samplespace(func, x_min, x_max, sampling_density):
    grid = np.linspace(x_min, x_max, sampling_density + 1)
    probs = np.diff(func.cdf(grid))
    return probs / np.sum(probs)  # ensuring it is normalized
