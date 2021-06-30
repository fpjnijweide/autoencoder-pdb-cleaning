import gc
import os

import numpy as np
import pandas as pd
import scipy.stats
from tensorflow import keras as keras

from .autoencoder import custom_loss, train_network, activation_types_default, hidden_layers_default, \
    encoding_dim_default, loss_function_default, training_method_default, activity_regularizer_default, epochs_default, \
    VAE_default, CNN_default, kernel_landmarks_default, CNN_layers_default, CNN_filters_default, \
    CNN_kernel_size_default, input_layer_type_default, gaussian_noise_sigma_default
from .bayesian_networks import make_bn
from .pdb import make_df
from .probabilities import JSD, wasserstein_rescaled

TEST_RUN_EPOCHS = False  # Whether to force the number of epochs below. Useful to test for errors without having to wait for hours of training
TEST_RUN_EPOCH_NR = 1

### Underlying data parameters
BN_size_default = 3  # amount of BN variables, minimum 3
mu_default = 0
sigma_default = 0.02  # Distribution of the noise
sampling_density_default = None  # How many bins the quasi-continuous variables use for distributing their probabilities. Higher_default = better approximation of continuous distributions
labeled_data_percentage_default = 2
use_gaussian_noise_default = True
use_missing_entry_default = False
missing_entry_prob_default = 0.01
rows_default = 10000
use_file_default = None

### AE parameters
activity_regularizer_default.__name__ = "L2, 10^-4"

defaults = dict(BN_size=BN_size_default, mu=mu_default, sigma=sigma_default, sampling_density=sampling_density_default,
                gaussian_noise_sigma=gaussian_noise_sigma_default, activation_types=activation_types_default,
                hidden_layers=hidden_layers_default, encoding_dim=encoding_dim_default,
                loss_function=loss_function_default, training_method=training_method_default,
                activity_regularizer=activity_regularizer_default, input_layer_type=input_layer_type_default,
                labeled_data_percentage=labeled_data_percentage_default, epochs=epochs_default, VAE=VAE_default,
                CNN=CNN_default, kernel_landmarks=kernel_landmarks_default, CNN_layers=CNN_layers_default,
                CNN_filters=CNN_filters_default, CNN_kernel_size=CNN_kernel_size_default,
                use_gaussian_noise=use_gaussian_noise_default, use_missing_entry=use_missing_entry_default,
                missing_entry_prob=missing_entry_prob_default, rows=rows_default, use_file=use_file_default)

parameters = list(defaults.keys())
gpu_string = ""


def run_experiment(full_string=None, epochs=epochs_default, use_previous_df=False, BN_size=BN_size_default,
                   sampling_density=sampling_density_default, mu=mu_default, sigma=sigma_default,
                   activation_types=activation_types_default, hidden_layers=hidden_layers_default,
                   encoding_dim=encoding_dim_default, loss_function=loss_function_default,
                   training_method=training_method_default, activity_regularizer=activity_regularizer_default,
                   input_layer_type=input_layer_type_default, labeled_data_percentage=labeled_data_percentage_default,
                   VAE=VAE_default, CNN=CNN_default, kernel_landmarks=kernel_landmarks_default,
                   CNN_layers=CNN_layers_default, CNN_filters=CNN_filters_default,
                   CNN_kernel_size=CNN_kernel_size_default, gaussian_noise_sigma=gaussian_noise_sigma_default,
                   use_gaussian_noise=use_gaussian_noise_default, use_missing_entry=use_missing_entry_default,
                   missing_entry_prob=missing_entry_prob_default, rows=rows_default, use_file=use_file_default):
    if TEST_RUN_EPOCHS:
        epochs = TEST_RUN_EPOCH_NR




    if use_file is None:
        bn = make_bn(BN_size, sampling_density)
    else:
        bn = None
    df, hard_evidence, sizes_sorted, gaussian_noise_layer_sigma_new, original_database, bins,is_this_bin_categorical = \
        make_df(use_file, bn,mu, sigma,use_gaussian_noise,use_missing_entry,missing_entry_prob,rows,full_string,
                sampling_density,gaussian_noise_sigma)

    if loss_function != 'MSE':
        old_loss = loss_function[:]
        loss_function = lambda y_true, y_pred: custom_loss(y_true, y_pred, sizes_sorted, old_loss,bins,is_this_bin_categorical)

    if encoding_dim is None:
        encoding_dim = len(original_database.columns)


    autoencoder = train_network(epochs, df, hard_evidence, activation_types, hidden_layers, encoding_dim, sizes_sorted,
                                loss_function, training_method, activity_regularizer, input_layer_type,
                                labeled_data_percentage, VAE, CNN, kernel_landmarks, CNN_layers, CNN_filters,
                                CNN_kernel_size, gaussian_noise_layer_sigma_new,is_this_bin_categorical)

    JSD_before, JSD_after, flip_TP, flip_TN, flip_FP, flip_FN, entropy_before, entropy_after,\
    wasserstein_JSD_before, wasserstein_JSD_after = measure_performance(df,hard_evidence,autoencoder,sizes_sorted,
        rows,full_string,original_database,bins,is_this_bin_categorical)

    if not VAE:
        autoencoder.save("./output_data/" + full_string + "/model.h5")
    else:
        autoencoder.save("./output_data/" + full_string + "/model.tf")
    del autoencoder
    gc.collect()
    keras.backend.clear_session()
    return JSD_before, JSD_after, flip_TP, flip_TN, flip_FP, flip_FN, entropy_before, entropy_after,wasserstein_JSD_before, wasserstein_JSD_after


def measure_performance(df, hard_evidence, autoencoder, sizes_sorted, rows, full_string, original_database, bins,is_this_bin_categorical,output_data_string=None):
    EXPECTATION_CONTINUOUS = True

    test_data = df.head(rows)

    verify_data = hard_evidence.iloc[test_data.index]
    results = pd.DataFrame(autoencoder.predict(test_data))

    if output_data_string is None:
        results.to_csv("./output_data/" + full_string + "/post_cleaning" + gpu_string + ".csv")
    else:
        filename_no_extension = os.path.splitext(output_data_string)[0]
        results.to_csv(filename_no_extension + "_post_cleaning_probabilities" + gpu_string + ".csv")

    pdb_col = 0
    JSD_before = []
    JSD_after = []
    flip_TP, flip_TN, flip_FP, flip_FN = [], [], [], []
    entropy_before, entropy_after = [], []
    wasserstein_JSD_before,wasserstein_JSD_after = [],[]

    ######### regenerate "bins" variable so that we can regenerate the original db
    cleaned_database_non_pdb = pd.DataFrame().reindex_like(original_database)

    for original_col, categories in enumerate(sizes_sorted):

        ground_truth_attribute = verify_data.iloc[:, pdb_col:pdb_col + categories]
        cleaned_attribute = results.iloc[:, pdb_col:pdb_col + categories]
        dirty_attribute = test_data.iloc[:, pdb_col:pdb_col + categories]

        if is_this_bin_categorical[original_col] or not EXPECTATION_CONTINUOUS:
            ground_truth_val = np.argmax(ground_truth_attribute.values, 1)
            clean_val = np.argmax(cleaned_attribute.values, 1)
            dirty_val = np.argmax(dirty_attribute.values, 1)
        else:
            # do expectation for continuous if needed, but then fit it into a bin
            # numerical using expectation
            if len(bins[original_col]) > 1:
                bin_width = bins[original_col][1] - bins[original_col][0]
            else:
                bin_width = 0
            real_ground_truth_val = np.dot(ground_truth_attribute.values, bins[original_col] + 0.5 * bin_width)
            real_clean_val = np.dot(cleaned_attribute.values, bins[original_col] + 0.5 * bin_width)
            real_dirty_val = np.dot(dirty_attribute.values, bins[original_col] + 0.5 * bin_width)

            ground_truth_val = np.searchsorted(bins[original_col], real_ground_truth_val, side='right') - 1
            clean_val = np.searchsorted(bins[original_col], real_clean_val, side='right') - 1
            dirty_val = np.searchsorted(bins[original_col], real_dirty_val, side='right') - 1

        ground_truth_missing = np.max(ground_truth_attribute.values, 1) == np.min(ground_truth_attribute.values, 1)
        clean_missing = np.max(cleaned_attribute.values, 1) == np.min(cleaned_attribute.values, 1)
        dirty_missing = np.max(dirty_attribute.values, 1) == np.min(dirty_attribute.values, 1)

        column_JSD_before = JSD(ground_truth_attribute, dirty_attribute)[~ground_truth_missing]
        column_JSD_after = JSD(ground_truth_attribute, cleaned_attribute)[~ground_truth_missing]



        JSD_before.append(np.nansum(column_JSD_before))
        JSD_after.append(np.nansum(column_JSD_after))

        if not is_this_bin_categorical[original_col]:
            column_wasserstein_JSD_before = wasserstein_rescaled(ground_truth_attribute, dirty_attribute,bins[original_col])[~ground_truth_missing]
            column_wasserstein_JSD_after = wasserstein_rescaled(ground_truth_attribute, cleaned_attribute,bins[original_col])[~ground_truth_missing]
        else:
            column_wasserstein_JSD_before = column_JSD_before
            column_wasserstein_JSD_after = column_JSD_after

        wasserstein_JSD_before.append(np.nansum(column_wasserstein_JSD_before))
        wasserstein_JSD_after.append(np.nansum(column_wasserstein_JSD_after))



        # If we are working on data where no noise was added at all, then F1 and accuracy scores and such are not applicable
        # We can use TP and FN to pass upwards how many values were flipped
        # if verify_data.equals(test_data):
        # # Changed values & missing entries made non-missing
        # TP = np.count_nonzero((ground_truth_missing & ~clean_missing) | (  (~ground_truth_missing & ~clean_missing)  &    (ground_truth_val != clean_val)))
        # # Value to missing
        # FP=np.count_nonzero(~ground_truth_missing & clean_missing)
        # # Missing to missing
        # FN=np.count_nonzero(ground_truth_missing & clean_missing)
        # # Value stayed the same
        # TN=np.count_nonzero((~ground_truth_missing & ~clean_missing)  & (ground_truth_val==clean_val))
        # else:

        # Don't count instances where ground truth was missing, because we simply have no idea.
        # True positive: Was missing or wrong, now correct
        column_TP = ((ground_truth_val != dirty_val) | dirty_missing) & ~((ground_truth_val != clean_val) | clean_missing)
        # False positive: Was correct, now missing or wrong
        column_FP = ~((ground_truth_val != dirty_val) | dirty_missing) & ((ground_truth_val != clean_val) | clean_missing)
        # False negative: Was incorrect/missing and still is
        column_FN = ((ground_truth_val != dirty_val) | dirty_missing) & ((ground_truth_val != clean_val) | clean_missing)
        # True negative: was correct and stayed correct
        column_TN = ~((ground_truth_val != dirty_val) | dirty_missing) & ~((ground_truth_val != clean_val) | clean_missing)

        flip_TP.append(np.count_nonzero(column_TP[~ground_truth_missing]))
        flip_FN.append(np.count_nonzero(column_FN[~ground_truth_missing]))
        flip_FP.append(np.count_nonzero(column_FP[~ground_truth_missing]))
        flip_TN.append(np.count_nonzero(column_TN[~ground_truth_missing]))

        column_entropy_before = scipy.stats.entropy(dirty_attribute, axis=1).sum()
        column_entropy_after = scipy.stats.entropy(cleaned_attribute, axis=1).sum()

        entropy_before.append(column_entropy_before)
        entropy_after.append(column_entropy_after)

        if is_this_bin_categorical[original_col]:
            real_ground_truth_val = bins[original_col][ground_truth_val]
            real_clean_val = bins[original_col][clean_val]
            real_dirty_val = bins[original_col][dirty_val]
        else:
            if not EXPECTATION_CONTINUOUS:
                # numerical, but using argmax instead of expectation
                if len(bins[original_col]) > 1:
                    bin_width = bins[original_col][1] - bins[original_col][0]
                else:
                    bin_width = 0
                real_ground_truth_val = bins[original_col][ground_truth_val] + 0.5 * bin_width
                real_clean_val = bins[original_col][clean_val] + 0.5 * bin_width
                real_dirty_val = bins[original_col][dirty_val] + 0.5 * bin_width
            else:
                # numerical using expectation
                if len(bins[original_col]) > 1:
                    bin_width = bins[original_col][1] - bins[original_col][0]
                else:
                    bin_width = 0
                real_ground_truth_val = np.dot(ground_truth_attribute.values,bins[original_col] + 0.5*bin_width)
                real_clean_val = np.dot(cleaned_attribute.values, bins[original_col] + 0.5*bin_width)
                real_dirty_val = np.dot(dirty_attribute.values, bins[original_col] + 0.5*bin_width)

        cleaned_database_non_pdb.iloc[:, original_col] = real_clean_val


        pdb_col += categories

    if output_data_string is None:
        cleaned_database_non_pdb.to_csv("./output_data/" + full_string + "/post_cleaning_non_pdb" + gpu_string + ".csv")
    else:
        filename_no_extension = os.path.splitext(output_data_string)[0]
        cleaned_database_non_pdb.to_csv(filename_no_extension + "_FINAL_CLEANED" + gpu_string + ".csv")


    JSD_before = np.nansum(JSD_before)
    JSD_after = np.nansum(JSD_after)

    wasserstein_JSD_before = np.nansum(wasserstein_JSD_before)
    wasserstein_JSD_after = np.nansum(wasserstein_JSD_after)

    flip_TP = np.nansum(flip_TP)
    flip_FN = np.nansum(flip_FN)
    flip_FP = np.nansum(flip_FP)
    flip_TN = np.nansum(flip_TN)

    entropy_before = np.nansum(entropy_before)
    entropy_after = np.nansum(entropy_after)

    # noise_left = 100 * avg_distance_after / avg_distance_before
    return JSD_before, JSD_after, flip_TP, flip_TN, flip_FP, flip_FN, entropy_before, entropy_after, wasserstein_JSD_before, wasserstein_JSD_after
