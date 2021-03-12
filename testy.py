# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
#  # Automated experiments
# 
#  This file contains an extended version of the code shown in continuous_variable_training.ipynb. Several options are added, such as convolutional layers, Gaussian kernels, and the ability to use a variational autoencoder. Methods were written so that a queue of experiments can be made, while the amount of outputs is heavily reduced.
# 
#  After running the experiments, the data was copied into a text editor, and using regular expressions, lines that do not contain curly brackets (such lines only contain information which is useful to see training progress) were removed. This data can be pasted into a program such as Google Sheets or Microsoft Excel, and split into multiple columns (like in a .csv file) using the semicolon as delimiter. The result of this can be seen in the .xlsx file included in this folder.
# 
#  For detailed explanations on the code, please look at continuous_variable_training.ipynb
# 
#  PS. these experiments will take several days to run on modern desktop computers. You might want to comment some of the lines in the second code cell to reduce the amount of measurements.

# %%
# Settings

USE_GPU = False

TEST_RUN_EPOCHS = False  # Whether to force the number of epochs below. Useful to test for errors without having to wait for hours of training
TEST_RUN_EPOCH_NR = 1
LOAD_DATA = True  # Whether to add results to previous .csv data
verbosity = 0

# %%
get_ipython().run_line_magic('matplotlib', 'inline')

import os

if USE_GPU:
    gpu_string = "_gpu"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    gpu_string = ""

import csv
import tensorflow as tf
import scipy.stats
import scipy.spatial
import numpy as np
import tensorflow.keras as keras
import pyAgrum as gum
import pandas as pd
import sklearn.model_selection
import math
import gc
import gkernel  # gkernel.py in this folder
import dill
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from IPython.display import clear_output

print('Imports done')

# %%
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print(tf.__version__)
print("Num GPUs Available: " + str(len(tf.config.experimental.list_physical_devices('GPU'))))
tf.test.is_gpu_available()

# %%
# # Default Variables for experiments.You probably want to use the cell above to change experiment vars.
BN_size_default = 3  # amount of BN variables, minimum 3

mu_default = 0
sigma_default = 0.02  # Distribution of the noise
gaussian_noise_sigma_default = lambda SD: (0.01 / SD) * 100
sampling_density_default = 4  # How many bins the quasi-continuous variables use for distributing their probabilities. Higher_default = better approximation of continuous distributions

activation_types_default = [keras.backend.sin, keras.backend.cos, keras.activations.linear, 'relu',
                            'swish']  # Activation layer types
hidden_layers_default = 3  # amount of hidden layers
encoding_dim_default = BN_size_default  # The dimensionality of the middle layer
loss_function_default = 'JSD'
training_method_default = 'semi'

activity_regularizer_default = keras.regularizers.l2(10 ** -4)
activity_regularizer_default.__name__ = "L2: 10^-4"

input_layer_type_default = 'gaussian_noise'
labeled_data_percentage_default = 2

epochs_default = 100
VAE_default = False
CNN_default = False
kernel_landmarks_default = 100

CNN_layers_default = 1
CNN_filters_default = 64
CNN_kernel_size_default = 3

use_gaussian_noise_default = True
use_missing_entry_default = False
missing_entry_prob_default = 0.01

defaults = dict(BN_size=BN_size_default, mu=mu_default, sigma=sigma_default, sampling_density=sampling_density_default,
                gaussian_noise_sigma=gaussian_noise_sigma_default, activation_types=activation_types_default,
                hidden_layers=hidden_layers_default, encoding_dim=encoding_dim_default,
                loss_function=loss_function_default, training_method=training_method_default,
                activity_regularizer=activity_regularizer_default, input_layer_type=input_layer_type_default,
                labeled_data_percentage=labeled_data_percentage_default, epochs=epochs_default, VAE=VAE_default,
                CNN=CNN_default, kernel_landmarks=kernel_landmarks_default, CNN_layers=CNN_layers_default,
                CNN_filters=CNN_filters_default, CNN_kernel_size=CNN_kernel_size_default,
                use_gaussian_noise=use_gaussian_noise_default, use_missing_entry=use_missing_entry_default,
                missing_entry_prob=missing_entry_prob_default)

parameters = list(defaults.keys())

# %%
experiments = []
experiment_config_list = []
experiment_config_strings = []
experiment_config_results = []
experiment_config_results2 = []


def load_from_csv(input_string):
    with open(input_string, 'r') as fp:
        reader = csv.reader(fp)
        li = list(reader)
    newlist = []
    for row in li:
        newrow = []
        for entry in row[1:]:
            if entry == '':
                break
            else:
                newrow.append(float(entry))
        newlist.append(newrow)
    return newlist


def find(inputlist, search, key=lambda z: z):
    for i in range(len(inputlist)):
        if key(inputlist[i]) == search:
            return i
    return None


def str_noneguard(obj):
    if hasattr(obj, '__name__'):
        return obj.__name__
    if obj is None:
        return ''
    if isinstance(obj, list):
        return str([str_noneguard(x) for x in obj])
    return str(obj)


def freeze(d):
    # thanks to https://stackoverflow.com/a/13264725
    if isinstance(d, dict):
        return frozenset((key, freeze(value)) for key, value in d.items())
    elif isinstance(d, list):
        return tuple(freeze(value) for value in d)
    return d


def gen_experiment(config_string, input_dict={}, parameter=None, vars=None):
    if parameter is None:
        vars = [None]

    for x in vars:
        if (x is None or x == 'default') and (not parameter is None):
            x = defaults[parameter]

        new_experiment_config = input_dict.copy()
        if parameter == 'input_layer_type' and x == 'VAE':
            new_experiment_config['VAE'] = True
        elif parameter == 'input_layer_type' and x == 'CNN':
            new_experiment_config['CNN'] = True
        elif parameter == 'missing_entry':
            new_experiment_config['use_missing_entry'] = True
            new_experiment_config['use_gaussian_noise'] = False
            new_experiment_config['missing_entry_prob'] = x
        elif parameter == 'missing_entry_combined':
            new_experiment_config['use_missing_entry'] = True
            new_experiment_config['use_gaussian_noise'] = True
            new_experiment_config['missing_entry_prob'] = x
        elif parameter == 'missing_entry_no_denoising':
            new_experiment_config['use_missing_entry'] = True
            new_experiment_config['use_gaussian_noise'] = False
            new_experiment_config['input_layer_type'] = 'dense'
            new_experiment_config['missing_entry_prob'] = x
        elif parameter == 'kernel_landmarks':
            new_experiment_config['input_layer_type'] = 'gaussian_kernel'
            new_experiment_config[parameter] = x
        elif parameter == 'CNN_kernel_size' or parameter == 'CNN_filters':
            new_experiment_config['CNN'] = True
            new_experiment_config[parameter] = x
        elif parameter == 'gaussian_noise_sigma':
            new_experiment_config['input_layer_type'] = 'gaussian_noise'
            new_experiment_config[parameter] = x
        elif parameter is not None:
            new_experiment_config[parameter] = x

        full_string = str(config_string + "    " + str_noneguard(parameter) + "    " + str_noneguard(x))

        if new_experiment_config in experiment_config_list:
            mapping = experiment_config_list.index(new_experiment_config)
        else:
            mapping = len(experiment_config_list)
            experiment_config_list.append(new_experiment_config)
            experiment_config_strings.append(full_string)
            experiment_config_results.append([])
            experiment_config_results2.append([])

        experiments.append(
            {'config_string': config_string, 'input_dict': input_dict, 'parameter': parameter, 'vars': vars,
             'current_var': x, 'config': new_experiment_config, 'full_string': full_string, 'mapping': mapping})


ground_config_strings = ["CCE, SD=4", "JSD, SD=4", "CCEu, SD=4", "JSDu, SD=4", "CCE, SD=100", "JSD, SD=100",
                         "CCEu, SD=100", "JSDu, SD=100"]

for config_string in ground_config_strings:
    ground_config = defaults.copy()
    if "CCE" in config_string:
        ground_config['loss_function'] = 'CCE'
    elif "JSD" in config_string:
        ground_config['loss_function'] = 'JSD'
    elif "MSE" in config_string:
        ground_config['loss_function'] = 'MSE'
    elif "KLD" in config_string:
        ground_config['loss_function'] = 'KLD'

    if "u," in config_string:
        ground_config['training_method'] = 'unsupervised'
    if "SD=100" in config_string:
        ground_config['sampling_density'] = 100

    # 0
    if ground_config['training_method'] != 'unsupervised':
        gen_experiment(config_string, ground_config, 'training_method',
                       ["supervised", "supervised_2_percent", "semi", "semi_sup_first", "semi_mixed", "unsupervised"])

    if ground_config['loss_function'] != 'CCE':
        # 1
        activation_list = [defaults['activation_types'], ['relu'], ['relu'] * 5,
                           [keras.backend.sin, keras.backend.cos, keras.activations.linear],
                           [keras.backend.sin, keras.backend.cos, keras.activations.linear, 'relu', 'sigmoid']]
        gen_experiment(config_string, ground_config, 'activation_types', activation_list)

        # 2
        gen_experiment(config_string, ground_config, 'input_layer_type',
                       ['dense', 'gaussian_noise', 'gaussian_dropout', 'sqrt_softmax', 'gaussian_kernel', 'CNN', 'VAE'])

        # 3
        gen_experiment(config_string, ground_config, 'encoding_dim', [2, 3, 6])

        # 4
        gen_experiment(config_string, ground_config, 'hidden_layers', [3, 5, 7, 9, 27])

        # 5
        regularizer_list = [None, keras.regularizers.l2(0.01), activity_regularizer_default,
                            keras.regularizers.l1(0.01), keras.regularizers.l1(10 ** -4)]
        regularizer_strings = ["none", "L2: 0.01", "L2: 10^-4", "L1: 0.01", "L1: 10^-4"]
        for i in range(len(regularizer_list)):
            try:
                regularizer_list[i].__name__ = regularizer_strings[i]
            except:
                pass
        gen_experiment(config_string, ground_config, 'activity_regularizer', regularizer_list)

        sigma_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        gen_experiment(config_string, ground_config, 'sigma', sigma_list)

        # 7
        if ground_config['sampling_density'] == 100:
            gen_experiment(config_string, ground_config, 'BN_size', [2, 3, 4, 5])
        else:
            gen_experiment(config_string, ground_config, 'BN_size', [2, 3, 4, 5, 10, 20, 30])

    # 8
    if ground_config['training_method'] != 'unsupervised':
        gen_experiment(config_string, ground_config, 'labeled_data_percentage',
                       [99, 50, 20, 10, 5, 2, 1, 0.5, 0.25, 0.125, 0.05, 0.01])
    # 9
    if ground_config['sampling_density'] != 100:
        gen_experiment(config_string, ground_config, 'sampling_density', [4, 15, 25, 50, 100, 150, 300])

    if ground_config['loss_function'] != 'CCE':
        # 10-13

        gaussian_noise_sigma_strings = ["lambda SD: 0.01", "lambda SD: 0.02", "lambda SD: 0.05", "lambda SD: 0.1",
                                        "lambda SD: 0.2", "lambda SD: (0.01/SD)*100", "lambda SD: (0.02/SD)*100",
                                        "lambda SD: (0.05/SD)*100", "lambda SD: (0.1/SD)*100",
                                        "lambda SD: (0.2/SD)*100"]
        gaussian_noise_sigma_list = [lambda SD: 0.01, lambda SD: 0.02, lambda SD: 0.05, lambda SD: 0.1, lambda SD: 0.2,
                                     lambda SD: (0.01 / SD) * 100, lambda SD: (0.02 / SD) * 100,
                                     lambda SD: (0.05 / SD) * 100, lambda SD: (0.1 / SD) * 100,
                                     lambda SD: (0.2 / SD) * 100]
        for i in range(len(gaussian_noise_sigma_list)):
            gaussian_noise_sigma_list[i].__name__ = gaussian_noise_sigma_strings[i]
        gen_experiment(config_string, ground_config, 'gaussian_noise_sigma', gaussian_noise_sigma_list)

        gen_experiment(config_string, ground_config, 'missing_entry', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
        gen_experiment(config_string, ground_config, 'missing_entry_combined', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
        gen_experiment(config_string, ground_config, 'missing_entry_no_denoising', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5])

if LOAD_DATA:
    try:
        experiment_config_results = load_from_csv("experiment_config_results" + gpu_string + ".csv")
        experiment_config_results2 = load_from_csv("experiment_config_results2" + gpu_string + ".csv")
    except:
        print('could not load data')

print("Experiments: " + str(len(experiments)))
print("Experiment configs: " + str(len(experiment_config_list)))
print("\n\n\n----------DONE---------\n\n\n")

for experiment in experiments:
    print(experiment['full_string'])


# %%


# %%
# @dispatch.add_dispatch_support
def JSD(y_true, y_pred):
    #     y_pred = ops.convert_to_tensor_v2(y_pred)
    #     y_true = math_ops.cast(y_true, y_pred.dtype)
    #     y_true = keras.backend.clip(y_true, keras.backend.epsilon(), 1)
    #     y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1)
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = keras.backend.clip(y_true, keras.backend.epsilon(), 1)
    y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1)
    means = 0.5 * (y_true + y_pred)
    divergence_tensor = 0.5 * keras.losses.kld(y_true, means) + 0.5 * keras.losses.kld(y_pred, means)

    return divergence_tensor


def generate_samplespace(func, x_min, x_max, sampling_density):
    grid = np.linspace(x_min, x_max, sampling_density + 1)
    probs = np.diff(func.cdf(grid))
    return probs / np.sum(probs)  # ensuring it is normalized


def normalize_df(df):
    newdf = df.div(df.sum(axis=1), axis=0)
    SD = len(newdf.columns)
    return newdf.fillna(1 / SD)


def prob_distance(x, y):
    xt = x.T  # this is bad practice, but I cannot debug the statements below without doing this
    yt = y.T
    res = scipy.spatial.distance.jensenshannon(xt, yt, base=2.0)
    res2 = JSD(x, y)
    return res, res2


def make_bn(BN_size, sampling_density):
    bn = gum.BayesNet("Quasi-Continuous")
    a = bn.add(gum.LabelizedVariable("A", "A binary variable", 2))
    bn.cpt(a)[:] = [0.4, 0.6]

    if BN_size > 1:
        b = bn.add(gum.RangeVariable("B", "A range variable", 0, sampling_density - 1))
        bn.addArc(a, b)
        first = generate_samplespace(scipy.stats.truncnorm(-10, 3), -10, 3, sampling_density)
        second = generate_samplespace(scipy.stats.truncnorm(-2, 6), -2, 6, sampling_density)
        bn.cpt(b)[{'A': 0}] = first
        bn.cpt(b)[{'A': 1}] = second

    if BN_size > 2:
        c = bn.add(gum.RangeVariable("C", "Another quasi continuous variable", 0, sampling_density - 1))
        bn.addArc(b, c)
        l = []
        for i in range(sampling_density):
            # the size and the parameter of gamma depends on the parent value
            k = (i * 30.0) / sampling_density
            l.append(generate_samplespace(scipy.stats.gamma(k + 1), 4, 5 + k, sampling_density))
        bn.cpt(c)[:] = l

        for d in range(BN_size - 3):
            # new variable
            d = bn.add(gum.RangeVariable("D" + str(d), "Another quasi continuous variable", 0, sampling_density - 1))
            l = []
            bn.addArc(c, d)
            for i in range(sampling_density):
                # the size and the parameter of gamma depends on the parent value
                k = (i * 30.0) / sampling_density
                l.append(generate_samplespace(scipy.stats.gamma(k + 1), 4, 5 + k, sampling_density))
            bn.cpt(d)[:] = l

    return bn


def make_df(use_previous_df, bn, mu, sigma, use_gaussian_noise, use_missing_entry, missing_entry_prob):
    if use_previous_df:
        original_database = pd.read_csv("good_db.csv")
    else:
        gum.generateCSV(bn, "databases/database_original" + gpu_string + ".csv", 10000)
        original_database = pd.read_csv("databases/database_original" + gpu_string + ".csv")
    original_database = original_database.reindex(sorted(original_database.columns), axis=1)
    original_database.to_csv("databases/database_original" + gpu_string + ".csv")

    size_dict = {}
    for column_name in original_database.columns:
        size_dict[column_name] = bn.variable(column_name).domainSize()

    shape = [original_database.shape[0], sum(size_dict.values())]

    df_cols_sorted = sorted(list(original_database.columns))
    sizes_sorted = [size_dict[x] for x in df_cols_sorted]
    sizes_sorted_with_leading_zero = [0] + sizes_sorted

    data = np.ones(original_database.shape[0] * original_database.shape[1])
    row = list(range(original_database.shape[0])) * original_database.shape[1]
    col = []
    for i in range(original_database.values.T.shape[0]):
        for item in original_database.values.T[i]:
            col.append(item + sum(sizes_sorted_with_leading_zero[0:i + 1]))
    # print(col[20000])

    input3 = scipy.sparse.coo_matrix((data, (row, col)), shape=tuple(shape)).todense()

    first_id2 = df_cols_sorted[:]
    second_id2 = [list(range(x)) for x in sizes_sorted]

    arrays3 = [np.repeat(first_id2, sizes_sorted), [item for sublist in second_id2 for item in sublist]]
    tuples2 = list(zip(*arrays3))
    index2 = pd.MultiIndex.from_tuples(tuples2, names=['Variable', 'Value'])

    hard_evidence = pd.DataFrame(input3, columns=index2)
    hard_evidence.to_csv("databases/ground_truth" + gpu_string + ".csv")

    df = hard_evidence + 0

    if use_gaussian_noise:
        noise = np.random.normal(mu, sigma, hard_evidence.shape)
        df = df + noise
        df = df.clip(lower=0, upper=1)
    if use_missing_entry:
        # TODO FIX
        amount_of_variables = len(sizes_sorted)
        rows = len(df)
        total_entries = amount_of_variables * rows  # amount of probability distributions in the PDB
        missing_entry_nrs = np.random.choice(total_entries, size=round(total_entries * missing_entry_prob),
                                             replace=False)
        m = missing_entry_nrs[:]  # using an alias for shorter code
        col_index = 0
        for attribute_nr, size in enumerate(sizes_sorted):
            entries_this_col = m[(m >= rows * attribute_nr) & (m < rows * (attribute_nr + 1))]
            rows_this_col = entries_this_col - (rows * attribute_nr)
            df.iloc[rows_this_col, col_index:col_index + size] = 1

            col_index += size

    for col in df_cols_sorted:
        df[col] = normalize_df(df[col])

    df.to_csv("databases/noisy_data" + gpu_string + ".csv")

    return df, hard_evidence, sizes_sorted


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


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
            #             print("kl_loss before reduce mean", kl_loss)
            #             print("kl_loss after reduce sum 1", tf.reduce_sum(kl_loss, axis=1))
            #             print("kl_loss after reduce mean", tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)))
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
        x = keras.layers.GaussianNoise(gaussian_noise_sigma)(input_layer)
    elif input_layer_type == 'gaussian_dropout':
        x = keras.layers.GaussianDropout(0.01)(input_layer)
    elif input_layer_type == 'sqrt_softmax':
        x = keras.layers.Lambda(keras.backend.sqrt)(input_layer)
        x = keras.layers.Softmax()(x)
    elif input_layer_type == "gaussian_kernel":

        x = gkernel.GaussianKernel3(kernel_landmarks, input_dim)(input_layer)

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


def measure_performance(df, hard_evidence, autoencoder, sizes_sorted):
    test_data = df.head(10000)

    verify_data = hard_evidence.iloc[test_data.index]
    results = pd.DataFrame(autoencoder.predict(test_data))

    results.to_csv("databases/post_cleaning" + gpu_string + ".csv")

    i = 0
    distances_before = []
    distances_before2 = []
    distances_after = []
    distances_after2 = []
    for size in sizes_sorted:
        dist_before, dist_before2 = prob_distance(verify_data.iloc[:, i:i + size], test_data.iloc[:, i:i + size])
        dist_after, dist_after2 = prob_distance(verify_data.iloc[:, i:i + size], results.iloc[:, i:i + size])
        distances_before.append(np.nansum(dist_before))
        distances_after.append(np.nansum(dist_after))
        distances_before2.append(np.nansum(dist_before2))
        distances_after2.append(np.nansum(dist_after2))
        i += size

    avg_distance_before = np.nansum(distances_before)
    avg_distance_after = np.nansum(distances_after)
    avg_distance_before2 = np.nansum(distances_before2)
    avg_distance_after2 = np.nansum(distances_after2)
    noise_left = 100 * avg_distance_after / avg_distance_before
    noise_left2 = 100 * avg_distance_after2 / avg_distance_before2
    return noise_left, noise_left2


def custom_loss2(y_true, y_pred, sizes_sorted, loss_func):
    i = 0
    total_loss = 0
    #     print("YEAH CUSTOM LOSS " + str(loss_func))
    #     print(loss_func)
    if loss_func == 'JSD':
        loss_func = JSD
        #         print("YEAH CUSTOM LOSS2 " + str(loss_func))
        for size in sizes_sorted:
            total_loss += loss_func(y_true[:, i:i + size], y_pred[:, i:i + size])
            i += size
        return total_loss
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


# %%
# -------------
def run_experiment(epochs=epochs_default, use_previous_df=False, BN_size=BN_size_default,
                   sampling_density=sampling_density_default, mu=mu_default, sigma=sigma_default,
                   activation_types=activation_types_default, hidden_layers=hidden_layers_default,
                   encoding_dim=encoding_dim_default, loss_function=loss_function_default,
                   training_method=training_method_default, activity_regularizer=activity_regularizer_default,
                   input_layer_type=input_layer_type_default, labeled_data_percentage=labeled_data_percentage_default,
                   VAE=VAE_default, CNN=CNN_default, kernel_landmarks=kernel_landmarks_default,
                   CNN_layers=CNN_layers_default, CNN_filters=CNN_filters_default,
                   CNN_kernel_size=CNN_kernel_size_default, gaussian_noise_sigma=gaussian_noise_sigma_default,
                   use_gaussian_noise=use_gaussian_noise_default, use_missing_entry=use_missing_entry_default,
                   missing_entry_prob=missing_entry_prob_default):
    if TEST_RUN_EPOCHS:
        epochs = TEST_RUN_EPOCH_NR

    bn = make_bn(BN_size, sampling_density)
    sigma = (sigma / sampling_density) * 100
    gaussian_noise_sigma = gaussian_noise_sigma(sampling_density)
    df, hard_evidence, sizes_sorted = make_df(use_previous_df, bn, mu, sigma, use_gaussian_noise, use_missing_entry,
                                              missing_entry_prob)

    if loss_function != 'MSE':
        old_loss = loss_function[:]
        loss_function = lambda y_true, y_pred: custom_loss2(y_true, y_pred, sizes_sorted, old_loss)

    autoencoder = train_network(epochs, df, hard_evidence, activation_types, hidden_layers, encoding_dim, sizes_sorted,
                                loss_function, training_method, activity_regularizer, input_layer_type,
                                labeled_data_percentage, VAE, CNN, kernel_landmarks, CNN_layers, CNN_filters,
                                CNN_kernel_size, gaussian_noise_sigma)
    noise_left, noise_left2 = measure_performance(df, hard_evidence, autoencoder, sizes_sorted)
    result = 100 - noise_left
    result2 = 100 - noise_left2
    del autoencoder
    gc.collect()
    keras.backend.clear_session()
    return result, result2


runs = 0
lowest_results = 0

while lowest_results < 10:
    lowest_results = min([len(x) for x in experiment_config_results])
    lowest_results_forcpu = min([len(experiment_config_results[x['mapping']]) for x in experiments if
                                 (x['config']['sampling_density'] * x['config']['BN_size']) < 100])
    lowest_results_forgpu = min([len(experiment_config_results[x['mapping']]) for x in experiments if
                                 (x['config']['sampling_density'] * x['config']['BN_size']) >= 100])
    highest_results = max([len(x) for x in experiment_config_results])
    print("\n\n----- LOWEST RESULTS: " + str(lowest_results) + ", HIGHEST: " + str(highest_results) + " ------\n\n")
    for i in (range(len(experiments))):
        experiment = experiments[i]
        x = experiment
        previous_runs = len(experiment_config_results[experiment['mapping']])
        if previous_runs == lowest_results:
            # if previous_runs==lowest_results_forcpu and (x['config']['sampling_density']*x['config']['BN_size'])<100:
            # if previous_runs==lowest_results_forgpu and (x['config']['sampling_density']*x['config']['BN_size'])>=100:
            if runs == 0:
                print(i)
            result, result2 = run_experiment(**experiment['config'])
            print("(" + str(i) + ") " + experiment['full_string'] + ";    " + str(result) + ", " + str(result2))
            experiment_config_results[experiment['mapping']].append(result)
            experiment_config_results2[experiment['mapping']].append(result2)
            #             experiment_configs_and_results[freeze(experiment['config'])].append(result)
            if runs % 10 == 0 and runs > 0:
                clear_output(wait=True)
            with open("experiments" + gpu_string, "wb") as dill_file:
                dill.dump(experiments, dill_file)
            experiment_configs_csv = [[experiment_config_strings[i]] + experiment_config_results[i] for i in
                                      range(len(experiment_config_results))]
            experiment_configs2_csv = [[experiment_config_strings[i]] + experiment_config_results2[i] for i in
                                       range(len(experiment_config_results))]
            with open("experiment_config_results" + gpu_string + ".csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(experiment_configs_csv)
            with open("experiment_config_results2" + gpu_string + ".csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(experiment_configs2_csv)
            runs += 1
    lowest_results = min([len(x) for x in experiment_config_results])
    lowest_results_forcpu = min([len(experiment_config_results[x['mapping']]) for x in experiments if
                                 (x['config']['sampling_density'] * x['config']['BN_size']) < 100])
    lowest_results_forgpu = min([len(experiment_config_results[x['mapping']]) for x in experiments if
                                 (x['config']['sampling_density'] * x['config']['BN_size']) >= 100])
# TODO LOADING
