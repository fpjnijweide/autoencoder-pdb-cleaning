import os
import csv
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import pandas as pd
import dill
import signal
import sys

from src.ExperimentsConfig import ExperimentsConfig
from src.experiments import run_experiment, defaults
from src.autoencoder import activity_regularizer_default
from src.helper_methods import DelayedKeyboardInterrupt, clean_directory, clean
from src.pdb import load_from_csv

try:
    from IPython.display import display
    from IPython.display import clear_output

    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    display = print
    clear_output = os.system('cls' if os.name == 'nt' else "printf '\033c'")  # %%

def main():
    print('Imports done')
    USE_GPU = False  # Turn on to use GPU and to append "_gpu" behind all saved files


    LOAD_DATA = True  # Whether to add results to previously saved .csv data

    ############################################################

    if USE_GPU:
        gpu_string = ""
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        gpu_string = ""



    # print(tf.python.client.device_lib.list_local_devices())
    print(tf.__version__)
    print("Num GPUs Available: " + str(len(tf.config.experimental.list_physical_devices('GPU'))))
    tf.test.is_gpu_available()

    for sig in (signal.SIGABRT, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, clean)


    experiments_config = ExperimentsConfig()

    ground_config_strings_real_data_real_world_example = ["JSDu, surgical_case_durations NO_ADDED_NOISE",
                                                          "JSDu, LBP RA NO_ADDED_NOISE",
                                                          "JSDu, surgical_case_durations NO_ADDED_NOISE SD=4",
                                                          "JSDu, LBP RA NO_ADDED_NOISE SD=4",
                                                          "JSDu, surgical_case_durations NO_ADDED_NOISE SD=100",
                                                          "JSDu, LBP RA NO_ADDED_NOISE SD=100",
                                                          "Wassersteinu, surgical_case_durations NO_ADDED_NOISE",
                                                          "Wassersteinu, LBP RA NO_ADDED_NOISE",
                                                          "Wassersteinu, surgical_case_durations NO_ADDED_NOISE SD=4",
                                                          "Wassersteinu, LBP RA NO_ADDED_NOISE SD=4",
                                                          "Wassersteinu, surgical_case_durations NO_ADDED_NOISE SD=100",
                                                          "Wassersteinu, LBP RA NO_ADDED_NOISE SD=100"]


    ground_config_strings_real_data = ["JSD, surgical_case_durations", "JSDu, surgical_case_durations", "JSD, LBP RA",
                                       "JSDu, LBP RA",
                                       "JSD, surgical_case_durations, SD=4", "JSDu, surgical_case_durations, SD=4",
                                       "JSD, LBP RA, SD=4", "JSDu, LBP RA, SD=4",
                                       "JSD, surgical_case_durations, SD=100", "JSDu, surgical_case_durations, SD=100",
                                       "JSD, LBP RA, SD=100", "JSDu, LBP RA, SD=100",
                                       "Wasserstein, surgical_case_durations", "Wassersteinu, surgical_case_durations", "Wasserstein, LBP RA",
                                       "Wassersteinu, LBP RA",
                                       "Wasserstein, surgical_case_durations, SD=4", "Wassersteinu, surgical_case_durations, SD=4",
                                       "Wasserstein, LBP RA, SD=4", "Wassersteinu, LBP RA, SD=4",
                                       "Wasserstein, surgical_case_durations, SD=100", "Wassersteinu, surgical_case_durations, SD=100",
                                       "Wasserstein, LBP RA, SD=100", "Wassersteinu, LBP RA, SD=100"]

    ground_config_strings_synthetic = ["Wasserstein, SD=4", "JSD, SD=4", "Wassersteinu, SD=4", "JSDu, SD=4", "Wasserstein, SD=100", "JSD, SD=100",
                                       "Wassersteinu, SD=100", "JSDu, SD=100"]

    ground_config_strings = ground_config_strings_real_data_real_world_example + ground_config_strings_real_data + ground_config_strings_synthetic



    for config_string in ground_config_strings:
        ground_config = defaults.copy()
        if "CCE" in config_string:
            ground_config['loss_function'] = 'CCE'
        elif "wasserstein" in config_string or "Wasserstein" in config_string:
            ground_config['loss_function'] = 'Wasserstein'
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
        elif "SD=4" in config_string:
            ground_config['sampling_density'] = 4

        if "surgical_case_durations" in config_string:
            ground_config['use_file'] = "./input_data/surgical_case_durations.csv"
        elif "LBP RA" in config_string:
            if "small" in config_string:
                ground_config['use_file'] = "./input_data/Dataset - LBP RA small.csv"
            else:
                ground_config['use_file'] = "./input_data/Dataset - LBP RA.csv"

        if ground_config['use_file'] is not None:
            if "NO_ADDED_NOISE" in config_string:
                # do not add noise, we work directly on ground truth
                experiments_config.gen_experiment(config_string, ground_config, 'sigma', [0])
            else:
                # If we are using a file
                sigma_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
                experiments_config.gen_experiment(config_string, ground_config, 'sigma', sigma_list)
                experiments_config.gen_experiment(config_string, ground_config, 'missing_entry', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
                experiments_config.gen_experiment(config_string, ground_config, 'missing_entry_combined', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
        else:
            # 0
            if ground_config['training_method'] != 'unsupervised':
                experiments_config.gen_experiment(config_string, ground_config, 'training_method',
                               ["supervised", "supervised_2_percent", "semi", "semi_sup_first", "semi_mixed",
                                "unsupervised"])

            if not (ground_config['loss_function'] == 'Wasserstein' and ground_config['sampling_density'] == 4):
                # 1
                activation_list = [defaults['activation_types'], ['relu'], ['relu'] * 5,
                                   [keras.backend.sin, keras.backend.cos, keras.activations.linear],
                                   [keras.backend.sin, keras.backend.cos, keras.activations.linear, 'relu', 'sigmoid']]
                experiments_config.gen_experiment(config_string, ground_config, 'activation_types', activation_list)

                # 2
                experiments_config.gen_experiment(config_string, ground_config, 'input_layer_type',
                               ['dense', 'gaussian_noise', 'gaussian_dropout', 'sqrt_softmax', 'gaussian_kernel', 'CNN',
                                'VAE'])

                # 3
                experiments_config.gen_experiment(config_string, ground_config, 'encoding_dim', [2, 3, 6])

                # 4
                experiments_config.gen_experiment(config_string, ground_config, 'hidden_layers', [3, 5, 7, 9, 27])

                # 5
                regularizer_list = [None, keras.regularizers.l2(0.01), activity_regularizer_default,
                                    keras.regularizers.l1(0.01), keras.regularizers.l1(10 ** -4)]
                regularizer_strings = ["none", "L2, 0.01", "L2, 10^-4", "L1, 0.01", "L1, 10^-4"]
                for i in range(len(regularizer_list)):
                    try:
                        regularizer_list[i].__name__ = regularizer_strings[i]
                    except:
                        pass
                experiments_config.gen_experiment(config_string, ground_config, 'activity_regularizer', regularizer_list)

                sigma_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
                experiments_config.gen_experiment(config_string, ground_config, 'sigma', sigma_list)

                # 7
                experiments_config.gen_experiment(config_string, ground_config, 'BN_size', [2, 3, 4, 5, 10, 20, 30])

            # 8
            if ground_config['training_method'] != 'unsupervised':
                experiments_config.gen_experiment(config_string, ground_config, 'labeled_data_percentage',
                               [99, 50, 20, 10, 5, 2, 1, 0.5, 0.25, 0.125, 0.05, 0.01])
            # 9
            if ground_config['sampling_density'] != 100:
                experiments_config.gen_experiment(config_string, ground_config, 'sampling_density', [4, 15, 25, 50, 100, 150, 300])

            if not (ground_config['loss_function'] == 'Wasserstein' and ground_config['sampling_density'] == 4):
                # 10-13
                gaussian_noise_sigma_strings = ["lambda SD, 0.01", "lambda SD, 0.02", "lambda SD, 0.05", "lambda SD, 0.1",
                                                "lambda SD, 0.2", "lambda SD, (0.01 over SD) cdot 100",
                                                "lambda SD, (0.02 over SD) cdot 100",
                                                "lambda SD, (0.05 over SD) cdot 100", "lambda SD, (0.1 over SD) cdot 100",
                                                "lambda SD, (0.2 over SD) cdot 100"]
                gaussian_noise_sigma_list = [lambda SD: 0.01, lambda SD: 0.02, lambda SD: 0.05, lambda SD: 0.1,
                                             lambda SD: 0.2,
                                             lambda SD: (0.01 / SD) * 100, lambda SD: (0.02 / SD) * 100,
                                             lambda SD: (0.05 / SD) * 100, lambda SD: (0.1 / SD) * 100,
                                             lambda SD: (0.2 / SD) * 100]
                for i in range(len(gaussian_noise_sigma_list)):
                    gaussian_noise_sigma_list[i].__name__ = gaussian_noise_sigma_strings[i]
                experiments_config.gen_experiment(config_string, ground_config, 'gaussian_noise_sigma', gaussian_noise_sigma_list)

                experiments_config.gen_experiment(config_string, ground_config, 'missing_entry', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
                experiments_config.gen_experiment(config_string, ground_config, 'missing_entry_combined', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
                experiments_config.gen_experiment(config_string, ground_config, 'missing_entry_no_denoising',
                               [0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
                # 14
                experiments_config.gen_experiment(config_string, ground_config, 'rows', [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6])

    if LOAD_DATA:
        try:
            experiments_config.JSD_before = load_from_csv("./results/experiment_config_JSD_before" + gpu_string + ".csv")
            experiments_config.JSD_after = load_from_csv("./results/experiment_config_JSD_after" + gpu_string + ".csv")

            experiments_config.flip_TP = load_from_csv("./results/experiment_config_flip_TP" + gpu_string + ".csv")
            experiments_config.flip_TN = load_from_csv("./results/experiment_config_flip_TN" + gpu_string + ".csv")
            experiments_config.flip_FP = load_from_csv("./results/experiment_config_flip_FP" + gpu_string + ".csv")
            experiments_config.flip_FN = load_from_csv("./results/experiment_config_flip_FN" + gpu_string + ".csv")

            experiments_config.entropy_before = load_from_csv(
                "./results/experiment_config_entropy_before" + gpu_string + ".csv")
            experiments_config.entropy_after = load_from_csv("./results/experiment_config_entropy_after" + gpu_string + ".csv")

            experiments_config.wasserstein_JSD_before = load_from_csv("./results/experiment_config_wasserstein_JSD_before" + gpu_string + ".csv")
            experiments_config.wasserstein_JSD_after = load_from_csv("./results/experiment_config_wasserstein_JSD_after" + gpu_string + ".csv")

            experiments_config.continuous_MSE_before = load_from_csv("./results/experiment_config_continuous_MSE_before" + gpu_string + ".csv")
            experiments_config.continuous_MSE_after = load_from_csv("./results/experiment_config_continuous_MSE_after" + gpu_string + ".csv")

        except:
            print('could not load data')

    print("Experiments: " + str(len(experiments_config.experiments)))
    print("Experiment configs: " + str(len(experiments_config.configs)))
    print("\n\n\n----------DONE---------\n\n\n")

    with open("./results/experiments" + gpu_string, "wb") as dill_file:
        dill.dump(experiments_config.experiments, dill_file)

    with pd.option_context("display.max_rows", 1000):
        display(pd.DataFrame(pd.DataFrame(experiments_config.experiments).loc[:, "full_string_list"].values.tolist()))

    # %%


    # %%
    # -------------


    clean_directory()
    pd.set_option('display.max_columns', None)

    runs = 0
    lowest_results = 0

    while lowest_results < 10:
        lowest_results = min([len(x) for x in experiments_config.JSD_after])
        highest_results = max([len(x) for x in experiments_config.JSD_after])
        print("\n\n----- LOWEST RESULTS: " + str(lowest_results) + ", HIGHEST: " + str(highest_results) + " ------\n\n")
        for i in reversed(range(len(experiments_config.experiments))):
            experiment = experiments_config.experiments[i]
            x = experiment
            previous_runs = len(experiments_config.JSD_after[experiment['mapping']])
            if previous_runs == lowest_results:
                if runs % 10 == 0 and runs > 0:
                    clear_output(wait=True)
                JSD_before, JSD_after, flip_TP, flip_TN, flip_FP, flip_FN, entropy_before, entropy_after,\
                    wasserstein_JSD_before, wasserstein_JSD_after,continuous_MSE_before,continuous_MSE_after = run_experiment(
                    experiment['full_string'], **experiment['config'])

                if JSD_before > 0:
                    JSD_reduction = 100 - ((JSD_after / JSD_before) * 100)
                elif JSD_before == JSD_after:
                    JSD_reduction = 0
                else:
                    JSD_reduction = -np.inf
                accuracy = 100 * ((flip_TP + flip_TN) / (flip_TP + flip_TN + flip_FP + flip_FN))
                f1_score = 100 * ((flip_TP) / (flip_TP + 0.5 * (flip_FP + flip_FN)))
                if entropy_before > 0:
                    entropy_reduction = 100 - ((entropy_after / entropy_before) * 100)
                elif entropy_before == entropy_after:
                    entropy_reduction = 0
                else:
                    entropy_reduction = -np.inf

                if wasserstein_JSD_before > 0:
                    wasserstein_JSD_reduction = 100 - ((wasserstein_JSD_after / wasserstein_JSD_before) * 100)
                elif wasserstein_JSD_before == wasserstein_JSD_after:
                    wasserstein_JSD_reduction = 0
                else:
                    wasserstein_JSD_reduction = -np.inf

                if continuous_MSE_before > 0:
                    continuous_MSE_reduction = 100 - ((continuous_MSE_after / continuous_MSE_before) * 100)
                elif continuous_MSE_before == continuous_MSE_after:
                    continuous_MSE_reduction = 0
                else:
                    continuous_MSE_reduction = -np.inf

                result_prints = pd.DataFrame(
                    [*experiment['full_string_list'], JSD_reduction, accuracy, f1_score, entropy_reduction,wasserstein_JSD_reduction,continuous_MSE_reduction]).T
                result_prints.columns = ["Base config", "Parameter", "Value", "Noise reduction", "Accuracy", "F1 score",
                                         "Entropy reduction","Wasserstein/JSD reduction","Continuous MSE reduction"]
                result_prints.index = [runs]
                display(result_prints)
                # print("(" + str(i) + ") " + experiment['full_string'] + ";    " + "Q: " + str(JSD_reduction) + " ACC: " + str(accuracy) + " F1: " + str(f1_score) + " H_red: " + str(entropy_reduction))

                experiments_config.JSD_before[experiment['mapping']].append(JSD_before)
                experiments_config.JSD_after[experiment['mapping']].append(JSD_after)

                experiments_config.flip_TP[experiment['mapping']].append(flip_TP)
                experiments_config.flip_TN[experiment['mapping']].append(flip_TN)
                experiments_config.flip_FP[experiment['mapping']].append(flip_FP)
                experiments_config.flip_FN[experiment['mapping']].append(flip_FN)

                experiments_config.entropy_before[experiment['mapping']].append(entropy_before)
                experiments_config.entropy_after[experiment['mapping']].append(entropy_after)

                experiments_config.wasserstein_JSD_before[experiment['mapping']].append(wasserstein_JSD_before)
                experiments_config.wasserstein_JSD_after[experiment['mapping']].append(wasserstein_JSD_after)

                experiments_config.continuous_MSE_before[experiment['mapping']].append(continuous_MSE_before)
                experiments_config.continuous_MSE_after[experiment['mapping']].append(continuous_MSE_after)

                experiment_config_JSD_before_csv = [[experiments_config.strings[i]] + experiments_config.JSD_before[i] for i
                                                    in range(len(experiments_config.JSD_before))]
                experiment_config_JSD_after_csv = [[experiments_config.strings[i]] + experiments_config.JSD_after[i] for i in
                                                   range(len(experiments_config.JSD_after))]

                experiment_config_flip_TP_csv = [[experiments_config.strings[i]] + experiments_config.flip_TP[i] for i in
                                                 range(len(experiments_config.flip_TP))]
                experiment_config_flip_TN_csv = [[experiments_config.strings[i]] + experiments_config.flip_TN[i] for i in
                                                 range(len(experiments_config.flip_TN))]
                experiment_config_flip_FP_csv = [[experiments_config.strings[i]] + experiments_config.flip_FP[i] for i in
                                                 range(len(experiments_config.flip_FP))]
                experiment_config_flip_FN_csv = [[experiments_config.strings[i]] + experiments_config.flip_FN[i] for i in
                                                 range(len(experiments_config.flip_FN))]

                experiment_config_entropy_before_csv = [[experiments_config.strings[i]] + experiments_config.entropy_before[i]
                                                        for i in range(len(experiments_config.entropy_before))]
                experiment_config_entropy_after_csv = [[experiments_config.strings[i]] + experiments_config.entropy_after[i]
                                                       for i in range(len(experiments_config.entropy_after))]

                experiment_config_wasserstein_JSD_before_csv = [[experiments_config.strings[i]] + experiments_config.wasserstein_JSD_before[i] for i
                                                    in range(len(experiments_config.wasserstein_JSD_before))]
                experiment_config_wasserstein_JSD_after_csv = [[experiments_config.strings[i]] + experiments_config.wasserstein_JSD_after[i] for i in
                                                   range(len(experiments_config.wasserstein_JSD_after))]

                experiment_config_continuous_MSE_before_csv = [[experiments_config.strings[i]] + experiments_config.continuous_MSE_before[i] for i
                                                    in range(len(experiments_config.continuous_MSE_before))]
                experiment_config_continuous_MSE_after_csv = [[experiments_config.strings[i]] + experiments_config.continuous_MSE_after[i] for i
                                                    in range(len(experiments_config.continuous_MSE_after))]

                with DelayedKeyboardInterrupt():
                    if not os.path.exists("./results/"):
                        os.makedirs("./results/")
                    with open("./results/experiment_config_JSD_before" + gpu_string + ".csv", "w",
                              newline="") as f: csv.writer(f).writerows(experiment_config_JSD_before_csv)
                    with open("./results/experiment_config_JSD_after" + gpu_string + ".csv", "w",
                              newline="") as f: csv.writer(f).writerows(experiment_config_JSD_after_csv)
                    with open("./results/experiment_config_flip_TP" + gpu_string + ".csv", "w", newline="") as f: csv.writer(
                        f).writerows(experiment_config_flip_TP_csv)
                    with open("./results/experiment_config_flip_TN" + gpu_string + ".csv", "w", newline="") as f: csv.writer(
                        f).writerows(experiment_config_flip_TN_csv)
                    with open("./results/experiment_config_flip_FP" + gpu_string + ".csv", "w", newline="") as f: csv.writer(
                        f).writerows(experiment_config_flip_FP_csv)
                    with open("./results/experiment_config_flip_FN" + gpu_string + ".csv", "w", newline="") as f: csv.writer(
                        f).writerows(experiment_config_flip_FN_csv)
                    with open("./results/experiment_config_entropy_before" + gpu_string + ".csv", "w",
                              newline="") as f: csv.writer(f).writerows(experiment_config_entropy_before_csv)
                    with open("./results/experiment_config_entropy_after" + gpu_string + ".csv", "w",
                              newline="") as f: csv.writer(f).writerows(experiment_config_entropy_after_csv)
                    with open("./results/experiment_config_wasserstein_JSD_before" + gpu_string + ".csv", "w",
                              newline="") as f: csv.writer(f).writerows(experiment_config_wasserstein_JSD_before_csv)
                    with open("./results/experiment_config_wasserstein_JSD_after" + gpu_string + ".csv", "w",
                              newline="") as f: csv.writer(f).writerows(experiment_config_wasserstein_JSD_after_csv)
                    with open("./results/experiment_config_continuous_MSE_before" + gpu_string + ".csv", "w",
                              newline="") as f: csv.writer(f).writerows(experiment_config_continuous_MSE_before_csv)
                    with open("./results/experiment_config_continuous_MSE_after" + gpu_string + ".csv", "w",
                              newline="") as f: csv.writer(f).writerows(experiment_config_continuous_MSE_after_csv)

                    with open("./results/experiments" + gpu_string, "wb") as dill_file:
                        dill.dump(experiments_config.experiments, dill_file)

                runs += 1
        lowest_results = min([len(x) for x in experiments_config.JSD_after])

    clean_directory()

if __name__ == "__main__":
    main()
