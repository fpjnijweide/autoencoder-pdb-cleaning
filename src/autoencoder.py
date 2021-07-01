import math
import sys

import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from src.gkernel import GaussianKernel3
from src.probabilities import JSD, wasserstein_loss_rescaled

activation_types_default = [keras.backend.sin, keras.backend.cos, keras.activations.linear, 'relu',
                            'swish']  # Activation layer types
hidden_layers_default = 3  # amount of hidden layers
encoding_dim_default = None  # set equal to BN size
loss_function_default = 'JSD'
training_method_default = 'semi'
activity_regularizer_default = keras.regularizers.l2(10 ** -4)
epochs_default = 100
VAE_default = False
CNN_default = False
kernel_landmarks_default = 100
CNN_layers_default = 1
CNN_filters_default = 64
CNN_kernel_size_default = 3
input_layer_type_default = 'gaussian_noise'
gaussian_noise_sigma_default = lambda SD: (0.01 / SD) * 100
activity_regularizer_default.__name__ = "L2, 10^-4"

verbosity = 0

class GaussianNoisePerNeuron(keras.layers.Layer):
    # adapted from standard Keras GaussianNoise layer
    def __init__(self, stddev, **kwargs):
        super(GaussianNoisePerNeuron, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            # noise_columns = [np.random.normal(0, scale=s, size=(array_ops.shape(inputs)[0])) for s in self.stddev]
            # noise = np.vstack(noise_columns).T
            batch_size = array_ops.shape(inputs)[0]
            vector_length = len(self.stddev)
            noise = tf.squeeze(tf.stack(
                [keras.backend.random_normal(shape=[batch_size, 1], mean=0., stddev=s, dtype=inputs.dtype) for s in
                 self.stddev], axis=1))
            # noise = keras.backend.random_normal(shape=array_ops.shape(inputs),mean=0.,stddev=self.stddev,dtype=inputs.dtype)
            output = inputs + noise
            output = tf.reshape(output, [-1, vector_length])
            # output = tf.ensure_shape(output, len(self.stddev))
            return output
            # return inputs

        return keras.backend.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(GaussianNoisePerNeuron, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class VAE_model(keras.Model):
    def __init__(self, encoder, decoder, loss_func, **kwargs):
        super(VAE_model, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_func = loss_func
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x, y = data
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            #             reconstruction_loss = self.loss_func(data, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            reconstruction_loss = tf.reduce_mean(
                self.loss_func(y, reconstruction)
            )

            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# def wasserstein_custom_loss(y_true, y_pred, sizes_sorted, loss_func,bins,is_this_bin_categorical):
#     total_loss = 0
#     loss_list = []
#     i = 0
#     for column_nr, size in enumerate(sizes_sorted):
#         if not is_this_bin_categorical[column_nr]:
#             new_loss = wasserstein_loss_rescaled(y_true[:, i:i + size], y_pred[:, i:i + size], bins[column_nr])
#         else:
#             new_loss = JSD(y_true[:, i:i + size], y_pred[:, i:i + size])
#
#         missing_rows_clean_for_this_col_bool = (tf.reduce_max(y_true[:, i:i + size], 1) == tf.reduce_min(y_true[:, i:i + size], 1)) & (size > 1)
#         missing_rows_for_this_col = tf.where(missing_rows_clean_for_this_col_bool)
#         zeros_we_need = tf.math.count_nonzero(missing_rows_clean_for_this_col_bool)
#
#         new_loss2 = tf.tensor_scatter_nd_update(new_loss,missing_rows_for_this_col,tf.zeros(zeros_we_need,dtype=new_loss.dtype))
#         loss_list.append(new_loss2)
#         i += size
#     good_loss = tf.math.add_n(loss_list)
#     return good_loss


def custom_loss(y_true, y_pred, sizes_sorted, loss_func,bins,is_this_bin_categorical,bin_widths):
    Wasserstein = False
    if loss_func == 'JSD':
        loss_func = JSD
    elif loss_func == "CCE":
        loss_func = keras.losses.categorical_crossentropy
    elif "wasserstein" in loss_func or "Wasserstein" in loss_func:
        Wasserstein = True
    else:
        loss_func = keras.losses.get(loss_func)


    loss_list = []
    i = 0
    for column_nr, size in enumerate(sizes_sorted):
        missing_rows_clean_for_this_col_bool = (tf.reduce_max(y_true[:, i:i + size], 1) == tf.reduce_min(y_true[:, i:i + size], 1)) & (size > 1)
        missing_rows_for_this_col = tf.where(missing_rows_clean_for_this_col_bool)
        zeros_we_need = tf.math.count_nonzero(missing_rows_clean_for_this_col_bool)

        if not Wasserstein:
            new_loss = loss_func(y_true[:, i:i + size], y_pred[:, i:i + size])
        else:
            if not is_this_bin_categorical[column_nr]:
                new_loss = wasserstein_loss_rescaled(y_true[:, i:i + size], y_pred[:, i:i + size], bins[column_nr] + 0.5*bin_widths[column_nr])
            else:
                new_loss = JSD(y_true[:, i:i + size], y_pred[:, i:i + size])

        new_loss2 = tf.tensor_scatter_nd_update(new_loss,missing_rows_for_this_col,tf.zeros(zeros_we_need,dtype=new_loss.dtype))
        loss_list.append(new_loss2)
        i += size
    good_loss = tf.math.add_n(loss_list)
    return good_loss


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def train_network(epochs, df, hard_evidence, activation_types, hidden_layers, encoding_dim, sizes_sorted, loss_function,
                  training_method, activity_regularizer, input_layer_type, labeled_data_percentage, VAE, CNN,
                  kernel_landmarks, CNN_layers, CNN_filters, CNN_kernel_size, gaussian_noise_sigma,is_this_bin_categorical,missing_rows_dirty,missing_rows_clean):
    x_train, y_train, x_train_nolabel = None, None, None

    df = df.copy(deep=True)
    hard_evidence = hard_evidence.copy(deep=True)

    # missing_rows_total = np.unique(np.concatenate([missing_rows_dirty,missing_rows_clean]))

    if training_method == 'supervised':
        # df.drop(missing_rows_clean)
        # hard_evidence.drop(missing_rows_clean)
        x_train, y_train = df, hard_evidence
    elif training_method == "unsupervised":
        # df.drop(missing_rows_total)
        x_train = df
    elif training_method == "supervised_2_percent":
        # df.drop(missing_rows_clean)
        # hard_evidence.drop(missing_rows_clean)
        x_train, _, y_train, _ = sklearn.model_selection.train_test_split(df, hard_evidence, test_size=0.98)
    elif training_method == "semi" or training_method == "semi_supervised" or training_method == "semisupervised" or training_method == "semi_sup_first" or training_method == "semi_mixed":
        x_train, x_train_nolabel, y_train, _ = sklearn.model_selection.train_test_split(df, hard_evidence, test_size=(
                                                                                                                             100 - labeled_data_percentage) / 100)  # semi supervised
    else:
        raise Exception("Invalid training method")

    x_train = np.float32(x_train)
    if y_train is not None:
        y_train = np.float32(y_train)
    if x_train_nolabel is not None:
        x_train_nolabel = np.float32(x_train_nolabel)

    # types = ['relu','relu','relu','relu','relu']
    input_dim = sum(sizes_sorted)
    # this is our input placeholder
    #     input_layer = keras.layers.Input(shape=(None,input_dim))
    input_layer = keras.layers.Input(shape=(input_dim,))

    # "encoded" is the encoded representation of the input
    if input_layer_type == 'dense' and not CNN:
        x = keras.layers.Dense(input_dim, activation='relu', activity_regularizer=activity_regularizer)(input_layer)
    elif CNN:
        x = tf.expand_dims(input_layer, axis=2)
        x = keras.layers.Conv1D(input_shape=(0, input_dim), filters=CNN_filters, kernel_size=CNN_kernel_size,
                                activation='relu')(x)
        x = keras.layers.MaxPooling1D(pool_size=2)(x)
        x = keras.layers.Flatten()(x)

    elif input_layer_type == 'gaussian_noise':
        x = GaussianNoisePerNeuron(gaussian_noise_sigma)(input_layer)
    elif input_layer_type == 'gaussian_dropout':
        x = keras.layers.GaussianDropout(0.01)(input_layer)
    elif input_layer_type == 'sqrt_softmax':
        x = keras.layers.Lambda(keras.backend.sqrt)(input_layer)
        x = keras.layers.Softmax()(x)
    elif input_layer_type == "gaussian_kernel":

        x = GaussianKernel3(kernel_landmarks, input_dim)(input_layer)

    encode_ratio = 0.1
    middle = hidden_layers // 2
    for i in range(hidden_layers):
        if CNN and i < CNN_layers - 1:

            x = tf.expand_dims(x, axis=2)
            x = keras.layers.Conv1D(filters=CNN_filters, kernel_size=CNN_kernel_size, activation='relu')(x)
            x = keras.layers.MaxPooling1D(pool_size=2)(x)
            x = keras.layers.Flatten()(x)

        elif VAE and i == middle:

            z_mean = keras.layers.Dense(encoding_dim, name="z_mean")(x)
            z_log_var = keras.layers.Dense(encoding_dim, name="z_log_var")(x)
            z = Sampling()([z_mean, z_log_var])
            latent_inputs = keras.layers.Input(shape=(encoding_dim,))
            x = tf.add(latent_inputs, 0)

        else:
            ratio = 2 ** (math.log2(encode_ratio) + abs(i - middle))
            size = encoding_dim if i == middle else max((min(input_dim, int(ratio * input_dim))), encoding_dim)
            #         print(size)
            if len(activation_types) > 1:
                x = keras.layers.concatenate(
                    [keras.layers.Dense(size, activation=type, activity_regularizer=activity_regularizer)(x) for type in
                     activation_types], axis=1)
            else:
                x = keras.layers.Dense(size, activation=activation_types[0], activity_regularizer=activity_regularizer)(
                    x)

    final_layer_list = [keras.layers.Dense(size, activation='softmax', activity_regularizer=activity_regularizer)(x) for
                        size in sizes_sorted]

    decoded = keras.layers.concatenate(final_layer_list, axis=1)

    if VAE:
        encoder = keras.models.Model(input_layer, [z_mean, z_log_var, z], name="encoder")
        decoder = keras.models.Model(latent_inputs, decoded, name="decoder")
        autoencoder = VAE_model(encoder, decoder, loss_function)
        autoencoder.compile(optimizer='adam', metrics=['accuracy'])  # semi supervised
        # autoencoder.compile(optimizer='adam', metrics=['accuracy'],run_eagerly=True)  # semi supervised
    else:
        autoencoder = keras.models.Model(input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])  # semi supervised
        # autoencoder.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'],run_eagerly=True)  # semi supervised

    hist = keras.callbacks.History()

    if training_method == 'supervised' or training_method == "supervised_2_percent":
        autoencoder.fit(x_train, y_train, epochs=epochs, batch_size=32, shuffle=True, callbacks=[hist],
                        verbose=verbosity)
    elif training_method == "semi" or training_method == "semi_supervised" or training_method == "semisupervised":
        # SEMI SUPERVISED
        autoencoder.fit(x_train_nolabel, x_train_nolabel, epochs=epochs, batch_size=32, shuffle=True, callbacks=[hist],
                        verbose=verbosity)
        autoencoder.fit(x_train, y_train, epochs=epochs, batch_size=32, shuffle=True, callbacks=[hist],
                        verbose=verbosity)
    elif training_method == "unsupervised":
        # UNSUPERVISED
        autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=32, shuffle=True, callbacks=[hist],
                        verbose=verbosity)
    elif training_method == "semi_sup_first":
        autoencoder.fit(x_train, y_train, epochs=epochs, batch_size=32, shuffle=True, callbacks=[hist],
                        verbose=verbosity)
        autoencoder.fit(x_train_nolabel, x_train_nolabel, epochs=epochs, batch_size=32, shuffle=True, callbacks=[hist],
                        verbose=verbosity)
    elif training_method == "semi_mixed":
        for i in range(epochs):
            autoencoder.fit(x_train_nolabel, x_train_nolabel, epochs=1, batch_size=32, shuffle=True, callbacks=[hist],
                            verbose=verbosity)
            autoencoder.fit(x_train, y_train, epochs=1, batch_size=32, shuffle=True, callbacks=[hist],
                            verbose=verbosity)
    else:
        raise Exception("Invalid training method")
    return autoencoder

if __name__=='__main__':
    sys.path.append('..')

    sizes_sorted=[3,2,1,4]
    bins=[np.array([0,1,2]),np.array([0,1]),np.array([0]),np.array([0,1,2,3])]
    bin_widths=[np.ones(3),np.ones(2),np.ones(1),np.ones(4)]
    is_this_bin_categorical=[False,False,False,False]

    y_true = np.random.randint(0, 2, size=(4, 10)).astype(np.float64)
    y_true[:,5]=1
    y_true[2,[0,1,2]]=(1/3)
    y_pred = np.random.random(size=(4,10))

    import pandas as pd

    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred)

    from src.pdb import normalize_df

    pdb_col = 0
    for original_col,size in enumerate(sizes_sorted):
        y_true.iloc[:,pdb_col:pdb_col+size]=normalize_df(y_true.iloc[:,pdb_col:pdb_col+size])
        y_pred.iloc[:, pdb_col:pdb_col + size] = normalize_df(y_pred.iloc[:, pdb_col:pdb_col + size])
        pdb_col+=size
    old_loss="JSD"

    y_true=np.array(y_true)
    y_pred=np.array(y_pred)

    loss_function = lambda y_true, y_pred: custom_loss(y_true, y_pred, sizes_sorted, old_loss, bins,
                                                       is_this_bin_categorical,bin_widths)


    loss = loss_function(y_true, y_pred)
    y_true = tf.keras.backend.clip(y_true, 1e-7, 1)
    y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1)


