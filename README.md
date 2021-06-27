<!-- [![DOI](https://zenodo.org/badge/263904926.svg)](https://zenodo.org/badge/latestdoi/263904926) -->

# Autoencoder-based cleaning in probabilistic databases

This is the source code for the paper "Autoencoder-based cleaning in probabilistic databases".

## Installation

Simply install all the packages in requirements.txt (`pip install -r requirements.txt`). If using Linux, install.sh can be used to generate a venv.

## Instructions for usage

- clean_one_file.py allows for cleaning a .csv file using the methods we described. takes 2 or 3 arguments: the file you want to clean, the name you want the cleaned file to have, and an optional filename of a previously trained autoencoder (usually a .h5) file that it will use for this cleaning. If the autoencoder filename is not specified, the program will train an autoencoder by itself and save it to the same folder as the cleaned .csv.

    Example usage (delete the last arguments if you want to train an autoencoder from scratch):
```
python3 clean_one_file.py input_data/surgical_case_durations.csv output_data/cleaned_db.csv "output_data/JSDu, SD=4    rows    10000/model.h5"
```
- run_experiments.py contains the code used to generate the results. When it is started, it will start creating many different autoencoders and PDB combinations for data cleaning, constantly saving new results into the results folder. It will also save the dictionary containing experiment configurations into the "output_data/experiments" file using dill. Turning the USE_GPU flag to True will ensure that the GPU is used. We do not recommend this, as training was usually faster on the CPU.
- figures/plot_results.ipynb can be used to generate the figures used in the paper. This requires the results folder to be populated with results first, and for an "experiments" file in the output_data folder (with the experiment configurations stored using dill) to exist. Confidence intervals only start appearing from n=2 measurements per configuration.
- results/merge_results.ipynb contains code to merge results stored in multiple .csv files (which happens when running experiments on multiple devices". Make sure they are stored within different folders such as "results_laptop" and "result_desktop" and change the lsit of suffixes in "mergelist" accordingly

## Other files

- The "src" folder contains methods used to generate the autoencoder, bayesian network of the underlying PDB, and many other helper methods
- The "output_data" folder contains the databases, ground truth PDB, noisy PDB, cleaned PDB and cleaned database of all the experiments that were executed since cloning this repository. It also contains all the trained autoencoders of each experiment as .h5 files.
- The "databases_used_in_paper" folder contains a similar set of databases that we used for the tables in the paper.
- The "figures" folder contains the .svg files of the figures shown in the paper.
- The "input_data" folder contains data we used to generate some of the figures and tables in the paper. You can add your own data here, as long as it is a .csv
