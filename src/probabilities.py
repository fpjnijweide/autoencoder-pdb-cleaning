import numpy as np
from tensorflow import keras as keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from scipy.stats import wasserstein_distance

wasserstein_wrongsignature = np.vectorize(wasserstein_distance, signature='(i),(i),(i),(i)->()', excluded=['u_values', 'v_values'])

def JSD(y_true, y_pred,current_bins=None):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = keras.backend.clip(y_true, keras.backend.epsilon(), 1)
    y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1)
    means = 0.5 * (y_true + y_pred)
    divergence_tensor = 0.5 * keras.losses.kld(y_true, means) + 0.5 * keras.losses.kld(y_pred, means)
    return divergence_tensor

def JSD_nontensor(y_true,y_pred):
    means = 0.5 * (y_true + y_pred)
    divergence_tensor = 0.5 * keras.losses.kld(y_true, means) + 0.5 * keras.losses.kld(y_pred, means)
    return divergence_tensor

def wasserstein_nontensor(y_true,y_pred,current_bins):
    return wasserstein_wrongsignature(current_bins,current_bins,y_true,y_pred)

def wasserstein(y_true,y_pred,current_bins):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = keras.backend.clip(y_true, keras.backend.epsilon(), 1)
    y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1)
    return wasserstein_nontensor(current_bins,current_bins,y_true,y_pred)

# def prob_distance(x, y):
#     res = JSD(x, y)
#     return res


def generate_samplespace(func, x_min, x_max, sampling_density):
    grid = np.linspace(x_min, x_max, sampling_density + 1)
    probs = np.diff(func.cdf(grid))
    return probs / np.sum(probs)  # ensuring it is normalized



if __name__ == '__main__':
    a = np.array([0, 0, 0, 1, 0, ])
    b = np.array([1, 0, 0, 0, 0, ])
    current_bins = np.arange(a.size)

    current_bins_repeat = np.repeat(current_bins[np.newaxis, :], 3, axis=0)
    a_repeat = np.repeat(a[np.newaxis, :], 3, axis=0)
    b_repeat = np.repeat(b[np.newaxis, :], 3, axis=0)

    # these are longer
    c = np.array([[0.3, 0.7], [0.2, 0.8], [0.5, 0.5]])
    d = c.copy()
    current_bins_c = np.arange(c.shape[1])
    d[2, 0], d[2, 1] = 1, 0

    print(wasserstein_nontensor(a_repeat,b_repeat,current_bins))
    print(wasserstein_nontensor(c,d,current_bins_c))