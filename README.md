This is the source code for the paper "Autoencoder-based cleaning of non-categorical data in probabilistic databases".

The install process for the generating of a virtual environment and such can be run using install.sh or install.bat.
This will require:
- Python 3
- GraphViz (on Windows)
- openssl for generating an SSL certificate for the Jupyter server (this can be skipped, but it might be unsafe)
- (optional) NVIDIA machine learning libraries for speedup (https://www.tensorflow.org/install/gpu)

Alternatively, just run the .ipynb notebook with your own pre-installed version of Python and Jupyter.
Any missing Python libraries can be installed with `pip3 install -r requirements.txt`