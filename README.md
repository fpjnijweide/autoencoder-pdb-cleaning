[![DOI](https://zenodo.org/badge/263904926.svg)](https://zenodo.org/badge/latestdoi/263904926)

# Installation

This is the source code for the paper "Autoencoder-based cleaning of non-categorical data in probabilistic databases".
The paper can be found at https://essay.utwente.nl/82344/

The install process for the generating of a virtual environment and such can be run using install.sh or install.bat.
This will require:
- Python 3
- GraphViz (on Windows)
- openssl for generating an SSL certificate for the Jupyter server (this can be skipped, but it might be unsafe)
- (optional) NVIDIA machine learning libraries for speedup (https://www.tensorflow.org/install/gpu)

Alternatively, just run the .ipynb notebooks with your own pre-installed version of Python and Jupyter.
Any missing Python libraries can be installed with `pip3 install -r requirements.txt`


# Explation of files

- experiments_explanation.ipynb contains a detailed explanation of the steps taken to generate the results, but it is missing some autoencoder methods that were included later on in the research.
- automated_experiments.ipynb contains the code actually used to generate the results, but it has fewer detailed explanations
- experiment_results.xlsx contains the data extracted from the output of automated_experiments.ipynb, which was used to generate tables and figures in the paper
- gkernel.py contains some methods written by Norio Tamada for Gaussian kernels. All credits for this file go to them (https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/gkernel/)
- database.csv contains the data sampled from the Bayesian network during the last experiment
- model.h5 contains the autoencoder created during the last experiment