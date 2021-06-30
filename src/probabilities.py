import numpy as np
from tensorflow import keras as keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from scipy.stats import wasserstein_distance

def JSD(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = keras.backend.clip(y_true, keras.backend.epsilon(), 1)
    y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1)
    means = 0.5 * (y_true + y_pred)
    divergence_tensor = 0.5 * keras.losses.kld(y_true, means) + 0.5 * keras.losses.kld(y_pred, means)
    return divergence_tensor

def wasserstein(y_true,y_pred):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = keras.backend.clip(y_true, keras.backend.epsilon(), 1)
    y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1)
    w = np.vectorize(wasserstein_distance, signature='(i),(i)->()')
    return w(y_true,y_pred)

def prob_distance(x, y):
    res = JSD(x, y)
    return res


def generate_samplespace(func, x_min, x_max, sampling_density):
    grid = np.linspace(x_min, x_max, sampling_density + 1)
    probs = np.diff(func.cdf(grid))
    return probs / np.sum(probs)  # ensuring it is normalized



if __name__ == '__main__':
    a = np.array([[0.3,0.7], [0.2, 0.8], [0.5,0.5]])
    b=a.copy()
    a_weights = np.ones(a.shape)
    b_weights = np.ones(b.shape)
    b[2,0],b[2,1]=1,0
    w = np.vectorize(wasserstein_distance,signature='(i),(i)->()')
    w(a,b,a_weights,b_weights)
    w(a,b)
    #w = np.vectorize(wasserstein_distance,excluded=['u_weights','v_weights'])