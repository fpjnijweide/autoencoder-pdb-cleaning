import math
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from gkernel import GaussianKernel3
from probabilities import JSD

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


def custom_loss(y_true, y_pred, sizes_sorted, loss_func):
    i = 0
    total_loss = 0
    if loss_func == 'JSD':
        loss_func = JSD
    elif loss_func == "CCE":
        loss_func = keras.losses.categorical_crossentropy
    else:
        loss_func = keras.losses.get(loss_func)

    loss_list = []

    for size in sizes_sorted:
        new_loss = loss_func(y_true[:, i:i + size], y_pred[:, i:i + size])
        loss_list.append(new_loss)
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
                  kernel_landmarks, CNN_layers, CNN_filters, CNN_kernel_size, gaussian_noise_sigma):
    x_train, y_train, x_train_nolabel = None, None, None

    if training_method == 'supervised':
        x_train, y_train = df, hard_evidence
    elif training_method == "unsupervised":
        x_train = df
    elif training_method == "supervised_2_percent":
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
    else:
        autoencoder = keras.models.Model(input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])  # semi supervised

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


