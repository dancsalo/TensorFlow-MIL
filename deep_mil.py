#!/usr/bin/env python

import argparse
import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Layer
from keras.models import Sequential, load_model

from datasets import Mnist
from tf_cnnvis import deconv_visualization


class NoisyAnd(Layer):
    """Custom NoisyAND layer from the Deep MIL paper"""

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NoisyAnd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = 10  # fixed, controls the slope of the activation
        self.b = self.add_weight(name='b',
                                 shape=(1, input_shape[3]),
                                 initializer='uniform',
                                 trainable=True)
        super(NoisyAnd, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        res = (tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)) / (
                tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]


def define_model(input_shape, num_classes):
    """Define Deep FCN for MIL, layer-by-layer from original paper"""
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(1000, (3, 3), activation='relu'))
    model.add(Conv2D(num_classes, (1, 1), activation='relu'))
    model.add(NoisyAnd(num_classes))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def train(epochs, seed, batch_size, dataset):
    """Train FCN"""
    np.random.seed(seed)

    model = define_model(dataset.input_shape, dataset.num_classes)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(dataset.x_train, dataset.y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(dataset.x_test, dataset.y_test))
    return model


def visualize(sess, model, dataset):
    """Save png of deconvolution image from first image in test set"""
    deconv_visualization(sess, {model.input: dataset.x_test[0:1, :, :, :]})


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Deep MIL Arguments')
    parser.add_argument('-e', '--NUM_EPOCHS', default=1, type=int)  # Number of epochs for which to train the model
    parser.add_argument('-r', '--SEED', default=123)  # specify the seed
    parser.add_argument('-b', '--BATCH_SIZE', default=128, type=int)  # batch size for training
    parser.add_argument('-s', '--SAVE_DIRECTORY', default='conv_mil/', type=str)  # Where to save model
    parser.add_argument('-m', '--MODEL_NAME', default='model.h5', type=str)  # To save individual run
    parser.add_argument('-t', '--TRAIN', default=1, type=int)  # whether to train (1) or load model (0)
    parser.add_argument('-v', '--VISUALIZE', default=1, type=int)  # whether to visualize the output
    flags = vars(parser.parse_args())

    # Build MNIST dataset
    dataset = Mnist()

    # Make save directory if it doesn't exist
    if not os.path.exists(flags['SAVE_DIRECTORY']):
        os.makedirs(flags['SAVE_DIRECTORY'])

    filepath = os.path.join(flags['SAVE_DIRECTORY'], flags['MODEL_NAME'])
    with tf.Graph().as_default():
        with tf.Session() as sess:
            K.set_session(sess)

            # Train or load model
            if flags['TRAIN']:
                model = train(epochs=flags['NUM_EPOCHS'],
                              seed=flags['SEED'],
                              batch_size=flags['BATCH_SIZE'],
                              dataset=dataset)
                model.save(filepath)
            else:
                model = load_model(filepath)

            # Visualize with tf_cnnvis
            visualize(sess, model, dataset)


if __name__ == "__main__":
    main()
