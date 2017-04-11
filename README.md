## Deep Multiple Instance Learning

This repository contains a TensorFlow implementation of the CNN-MIL
combination described in [Classifying and Segmenting Microscopy Images with
Deep Multiple Instance Learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4908336/pdf/btw252.pdf) from Brendan Frey's lab.
The unique contribution of this paper is the noisyAND module.
The implemented model uses the MNIST dataset for classification.

### Dependencies
 * Python 3.5 or greater
 * Tensorflow 1.0 or greater
 * Current version of TensorBase (for Layers and Model class)

### Instructions
To run the demo, clone the repo onto your local machine, navigate into the repository,
using the command line/terminal, and type `python Conv_MIL_Mnist.py {argparser flags}`. 
 Replace `{argparser flags}` with the appropriate choice of options as described in
 the section below.

### Argparser
The following options are available for running the model:
* `-n`, run_num, default = 0, saves all model files `under /save_directory/model_directory/Model[n]`
* `-e`, epochs, default = 1, number of epochs for which to train the model
* `-r`, restore, default = 0, binary value indicating whether to restore from a model.
* `-m`, model_restore, default = 1, restores from `/save_directory/model_directory/Model[n]`
* `-f`, file_epoch, default = 1,  restores model with the following checkpoint: `part_[f].ckpt.meta`
* `-t`, train, default=1,  # Binary to train model. 0 = No train.
* `-v`, eval, default=1)  # Binary to evalulate model. 0 = No eval.
* `-l`, learn_rate, default=1e-3, learning rate
* `-i`, vis, default = 0, binary value indicating whether to enable visualizations
* `-g`, gpu, default = 0, options are [0, 1, 2]. `0` or `1` designates which GPU to use. `2` designates using all available GPUs.