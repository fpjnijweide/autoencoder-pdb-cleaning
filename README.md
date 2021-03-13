<!-- [![DOI](https://zenodo.org/badge/263904926.svg)](https://zenodo.org/badge/latestdoi/263904926) -->

# Autoencoder-based cleaning in probabilistic databases

This is the source code for the paper "Autoencoder-based cleaning in probabilistic databases".

## Installation

Simply install all the packages in requirements.txt (`pip install -r requirements.txt`). If using Linux, install.sh can be used to generate a venv.

## Instructions for usage

- automated_experiments.ipynb contains the code used to generate the results. When it is started, it will start creating many different autoencoders and PDB combinations for data cleaning, constantly saving new results into experiment_config_results.csv. It will also save the dictionary containing experiment configurations into the "experiments" file using dill. Turning the USE_GPU flag to True will ensure any saved files receive the "_gpu" suffix and that the GPU is used. This allows for training with the CPU and GPU simultaneously.
- plot_results.ipynb can be used to generate the figures used in the paper. This requires experiment_config_results.csv to be populated with results first, and for an "experiments" file (with the experiment configurations stored using dill) to exist. Confidence intervals only start appearing from n=2 measurements per configuration.
- merge_results.ipynb contains code to merge results stored in multiple .csv files (which happens when running experiments on both the CPU and GPU)

## Other files

- gkernel.py contains some methods written by Norio Tamada for RBF kernels. All credits for this file go to them (https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/gkernel/)
- The "databases" folder contains the databases, ground truth data, noisy data and cleaned data from the last experiment
- The "databases_used_in_paper" folder contains a similar set of databases that we used for the tables in the paper.
- The "pictures" folder contains the .svg files of the figures shown in the paper.
- model.h5 contains the autoencoder created during the last experiment
