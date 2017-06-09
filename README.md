## Deep Multiple Instance Learning

This repository contains a TensorFlow implementation of the CNN-MIL
combination described in [Classifying and Segmenting Microscopy Images with
Deep Multiple Instance Learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4908336/pdf/btw252.pdf) from Brendan Frey's lab.
The unique contribution of this paper is the noisyAND module.
The implemented model uses the MNIST dataset for classification.

### Paper Summary

The architecture presented in the paper involves a CNN, the noisyAND layer,
followed by an additional fully connected layer with a joint cross entropy loss.

![alt text][/png/arch.png]

The joint cross entropy loss is summed over all classes and compares the
output of the MIL layer, `P_i`, and the output of the additional fully connected layer, `y_i`,
to the class layer for the entire image, `t_i`. The authors claim that the
additional fully connect layer will learn dependencies between classes.

![alt text][/png/loss.png]

### Dependencies
 * Python 3.5 or greater
 * Tensorflow 1.0 or greater
 * Latest commit of TensorBase as of this writing: [d7be3da](https://github.com/dancsalo/TensorBase/commit/d7be3dafab9f88ee42b74eec4459c42fda6ba15c).

### Instructions
To run the demo, clone the repo onto your local machine, navigate into the repository,
using the command line/terminal, and type `python Conv_MIL_Mnist.py {argparser flags}`. 
 Replace `{argparser flags}` with the appropriate choice of options as described in
 the section below. You will also have to clone TensorBase and place it in
 the same folder as this repo.

### Argparser
The following options are available for running the model:

* `-n`, run_num, default = 0, saves all model files under `/save_directory/model_directory/Model[n]`
* `-e`, epochs, default = 1, number of epochs for which to train the model
* `-r`, restore, default = 0, binary value indicating whether to restore from a model.
* `-m`, model_restore, default = 1, restores from `/save_directory/model_directory/Model[n]`
* `-f`, file_epoch, default = 1,  restores model with the following checkpoint: `part_[f].ckpt.meta`
* `-t`, train, default=1,  # Binary to train model. 0 = No train.
* `-v`, eval, default=1)  # Binary to evalulate model. 0 = No eval.
* `-l`, learn_rate, default=1e-3, learning rate
* `-g`, gpu, default = 0, accepts a single integer or a string "all" to use all available.