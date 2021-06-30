import os

from .autoencoder import train_network, epochs_default, activation_types_default, encoding_dim_default, \
    training_method_default, input_layer_type_default, CNN_default, VAE_default, CNN_layers_default, \
    CNN_kernel_size_default, gaussian_noise_sigma_default, CNN_filters_default, kernel_landmarks_default, \
    activity_regularizer_default, loss_function_default, hidden_layers_default, custom_loss
from .experiments import sampling_density_default, use_gaussian_noise_default, missing_entry_prob_default, rows_default, \
    use_file_default, use_missing_entry_default, labeled_data_percentage_default, sigma_default, mu_default, \
    BN_size_default, defaults, measure_performance
from .pdb import make_df
import tensorflow.keras as keras


def clean_config(filename_out, autoencoder_filename, full_string=None, epochs=epochs_default, use_previous_df=False,
                 BN_size=BN_size_default,
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

    bn = None
    df, hard_evidence, sizes_sorted, gaussian_noise_layer_sigma_new, original_database, bins, is_this_bin_categorical = \
        make_df(use_file, bn, mu, sigma, use_gaussian_noise,use_missing_entry,missing_entry_prob,rows,full_string,
                sampling_density,gaussian_noise_sigma,filename_out)

    if loss_function != 'MSE':
        old_loss = loss_function[:]
        loss_function = lambda y_true, y_pred: custom_loss(y_true, y_pred, sizes_sorted, old_loss)

    if encoding_dim is None:
        encoding_dim = len(original_database.columns)


    if autoencoder_filename is None:
        autoencoder = train_network(epochs, df, hard_evidence, activation_types, hidden_layers, encoding_dim,
                                    sizes_sorted,
                                    loss_function, training_method, activity_regularizer, input_layer_type,
                                    labeled_data_percentage, VAE, CNN, kernel_landmarks, CNN_layers, CNN_filters,
                                    CNN_kernel_size, gaussian_noise_layer_sigma_new,is_this_bin_categorical)
    else:
        autoencoder = keras.models.load_model(autoencoder_filename)
    measure_performance(df, hard_evidence,
                        autoencoder,
                        sizes_sorted,
                        rows,
                        full_string,
                        original_database,
                        bins, is_this_bin_categorical,filename_out
                        )

    if autoencoder_filename is None:
        filename_no_extension = os.path.splitext(filename_out)[0]
        if not VAE:
            autoencoder.save(filename_no_extension + ".h5")
        else:
            autoencoder.save(filename_no_extension + ".tf")


def clean(filename_in, filename_out, autoencoder_filename):
    config = defaults
    config['use_file'] = filename_in
    config['sigma'] = 0
    config['loss_function'] = 'JSD'
    config['training_method'] = 'unsupervised'

    clean_config(filename_out, autoencoder_filename, **config)
