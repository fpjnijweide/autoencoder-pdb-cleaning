import numpy as np
from tensorflow import keras as keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from scipy.stats import wasserstein_distance
import tensorflow as tf


def _cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None):
    # Rewritten version of scipy stats cdf distance
    tf.debugging.assert_equal(u_values,v_values)
    # Compute the differences between pairs of successive values of u and v.

    u_values = ops.convert_to_tensor_v2(u_values)
    v_values = ops.convert_to_tensor_v2(v_values)
    u_weights = ops.convert_to_tensor_v2(u_weights)
    v_weights = ops.convert_to_tensor_v2(v_weights)

    u_values = math_ops.cast(u_values, u_weights.dtype)
    v_values = math_ops.cast(v_values, u_weights.dtype)
    v_weights = math_ops.cast(v_weights, u_weights.dtype)

    # u_weights = tf.convert_to_tensor(u_weights,dtype=tf.float32)
    # v_weights = tf.convert_to_tensor(v_weights,dtype=tf.float32)
    # v_values = tf.convert_to_tensor(v_values,dtype=tf.float32)
    # u_values = tf.convert_to_tensor(u_values,dtype=tf.float32)

    all_values = tf.repeat(u_values,2)

    deltas = tf.experimental.numpy.diff(all_values)
    # Get the respective positions of the values of u and v among the values of
    # both distributions.

    u_sorter=tf.range(tf.squeeze(u_values.shape))
    v_sorter=tf.range(tf.squeeze(v_values.shape))

    u_cdf_indices = (tf.repeat(u_sorter,2)+1)[:-1 ]
    v_cdf_indices = (tf.repeat(v_sorter,2)+1)[:-1 ]

    # tf.print(keras.backend.eval(u_weights))
    # tf.print(keras.backend.eval(v_weights))
    # tf.print(keras.backend.eval(u_values))

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = tf.concat((np.zeros([u_weights.shape[0],1]), tf.cumsum(u_weights,axis=1)),axis=1)
        u_cdf = tf.gather(u_sorted_cumweights,tf.cast(u_cdf_indices,dtype=tf.int32),axis=1) / tf.expand_dims(u_sorted_cumweights[:,-1],1)

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = tf.concat((np.zeros([u_weights.shape[0],1]), tf.cumsum(v_weights,axis=1)),axis=1)
        v_cdf = tf.gather(v_sorted_cumweights,tf.cast(v_cdf_indices,dtype=tf.int32),axis=1) / tf.expand_dims(v_sorted_cumweights[:,-1],1)

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return tf.math.reduce_sum(tf.multiply(tf.abs(u_cdf - v_cdf), deltas),axis=1)
    if p == 2:
        return tf.sqrt(tf.math.reduce_sum(tf.multiply(tf.square(u_cdf - v_cdf), deltas),axis=1))
    return tf.experimental.numpy.power(tf.math.reduce_sum(tf.multiply(tf.experimental.numpy.power(tf.abs(u_cdf - v_cdf), p),deltas)), 1/p)

def wasserstein_wrongsignature(u_values, v_values, u_weights=None, v_weights=None):
    return _cdf_distance(1, u_values, v_values, u_weights, v_weights)

wasserstein_wrongsignature_old = np.vectorize(wasserstein_distance, signature='(i),(i),(i),(i)->()', excluded=['u_values', 'v_values'])

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

def wasserstein_unscaled(y_true,y_pred,current_bins):
    return wasserstein_wrongsignature(current_bins,current_bins,y_true,y_pred)

def wasserstein_rescaled(y_true,y_pred,current_bins):
    scaling_factor = tf.math.log(tf.cast(2,dtype=tf.float32))/(current_bins[-1]-current_bins[0])
    return wasserstein_unscaled(y_true,y_pred,current_bins)*scaling_factor

def wasserstein_loss_rescaled(y_true,y_pred,current_bins):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = keras.backend.clip(y_true, keras.backend.epsilon(), 1)
    y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1)
    return wasserstein_rescaled(y_true,y_pred,current_bins)

# def prob_distance(x, y):
#     res = JSD(x, y)
#     return res


def generate_samplespace(func, x_min, x_max, sampling_density):
    grid = np.linspace(x_min, x_max, sampling_density + 1)
    probs = np.diff(func.cdf(grid))
    return probs / np.sum(probs)  # ensuring it is normalized

def em_loss(y_coefficients, y_pred):
    # From https://gist.github.com/mjdietzx/a8121604385ce6da251d20d018f9a6d6
    return tf.reduce_mean(tf.multiply(y_coefficients, y_pred))

if __name__ == '__main__':
    a = np.array([0, 0, 0, 1, 0, ])
    b = np.array([1, 0, 0, 0, 0, ])
    current_bins = np.array([4, 6 , 9, 10, 12])

    current_bins_repeat = np.repeat(current_bins[np.newaxis, :], 3, axis=0)
    a_repeat = np.repeat(a[np.newaxis, :], 3, axis=0)
    b_repeat = np.repeat(b[np.newaxis, :], 3, axis=0)

    # these are longer
    c = np.array([[0.3, 0.7], [0.2, 0.8], [0.5, 0.5]])
    d = c.copy()
    current_bins_c = np.arange(c.shape[1])
    d[2, 0], d[2, 1] = 1, 0

    print(wasserstein_wrongsignature_old(current_bins,current_bins,a_repeat,b_repeat))
    print(_cdf_distance(1, current_bins, current_bins, a_repeat,b_repeat))

    #
    # print(wasserstein_unscaled(a_repeat,b_repeat,current_bins))
    # print(wasserstein_unscaled(c,d,current_bins_c))