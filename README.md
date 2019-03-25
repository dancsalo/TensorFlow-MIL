## Deep Multiple Instance Learning

This repository contains a TensorFlow implementation of the CNN-MIL
combination described in [Classifying and Segmenting Microscopy Images with
Deep Multiple Instance Learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4908336/pdf/btw252.pdf) from Brendan Frey`s lab.

### Getting Started
 * Install requirements (will install CPU version of TensorFlow): `pip install requirements.txt`.
 * Follow instructions on (tf_cnnvis)[https://github.com/InFoCusp/tf_cnnvis] README. `tf_cnnvis/` should be
 at the root after installation.


### Options
The following options are available for running the model:

* `-e`, Number of epochs for which to train the model
* `-r`, Specify the seed
* `-b`, Batch size for training
* `-s`, Where to save model
* `-m`, Name of model
* `-t`, Whether to train (1) or load model (0)

#### Datasets
The `datasets.py` file contains a small cluttered MNIST dataset. Each
image is 72 x 72 pixels and contains four numbers: three are 0's, the other one is a number
1 - 9 excluding 0. The locations of these numbers in the image are semi-random.
This dataset is a much smaller version of what the authors use in the paper.