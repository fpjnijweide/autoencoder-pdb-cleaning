from experiments import defaults
from helper_methods import str_noneguard


class ExperimentsConfig:
    # %%
    experiments = []
    configs = []
    strings = []

    JSD_before = []
    JSD_after = []

    # noise reduction performance score: (noise after) / (noise before) in %

    flip_TP = []  # good flip
    flip_TN = []  # good left unflipped
    flip_FP = []  # unneeded flip
    flip_FN = []  # wrong flip or unflipped

    entropy_before = []
    entropy_after = []

    def gen_experiment(self, config_string, input_dict={}, parameter=None, vars=None):
        if parameter is None:
            vars = [None]

        for x in vars:
            if (not parameter == 'activity_regularizer') and (x is None or x == 'default') and (not parameter is None):
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
            full_string_list = (config_string, str_noneguard(parameter), str_noneguard(x))

            if new_experiment_config in self.configs:
                mapping = self.configs.index(new_experiment_config)
            else:
                mapping = len(self.configs)
                self.configs.append(new_experiment_config)
                self.strings.append(full_string)

                self.JSD_before.append([])
                self.JSD_after.append([])

                self.flip_TP.append([])
                self.flip_TN.append([])
                self.flip_FP.append([])
                self.flip_FN.append([])

                self.entropy_before.append([])
                self.entropy_after.append([])

            self.experiments.append(
                {'config_string': config_string, 'input_dict': input_dict, 'parameter': parameter, 'vars': vars,
                 'current_var': x, 'config': new_experiment_config, 'full_string': full_string, 'mapping': mapping,
                 'full_string_list': full_string_list})
