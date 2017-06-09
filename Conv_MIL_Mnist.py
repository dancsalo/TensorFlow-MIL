#!/usr/bin/env python

"""
Author: Dan Salo
Initial Commit: 12/1/2016

Purpose: Implement Convolutional Multiple Instance Learning for distributed learning over MNIST dataset
"""

import sys
sys.path.append('../')

from TensorBase.tensorbase.base import Model
from TensorBase.tensorbase.base import Layers

import tensorflow as tf
import numpy as np
import argparse
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm


class ConvMil(Model):
    def __init__(self, flags_input):
        """ Initialize from Model class in TensorBase """
        super().__init__(flags_input)
        self.valid_results = list()
        self.test_results = list()
        self.checkpoint_rate = 5  # save after this many epochs
        self.valid_rate = 5  # validate after this many epochs

    def _data(self):
        """ Define all data-related parameters. Called by TensorBase. """
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.num_train_images = self.mnist.train.num_examples
        self.num_valid_images = self.mnist.validation.num_examples
        self.num_test_images = self.mnist.test.num_examples
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        self.y = tf.placeholder(tf.int32, shape=[None, self.flags['NUM_CLASSES']], name='y')

    def _summaries(self):
        """ Write summaries out to TensorBoard. Called by TensorBase. """
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("XEntropy_Loss_Pi", self.xentropy_p)
        tf.summary.scalar("XEntropy Loss_yi", self.xentropy_y)
        tf.summary.scalar("Weight_Decay_Loss", self.weight)

    def _network(self):
        """ Define neural network. Uses Layers class of TensorBase. Called by TensorBase. """
        with tf.variable_scope("model"):
            net = Layers(self.x)
            net.conv2d(5, 64)
            net.maxpool()
            net.conv2d(3, 64)
            net.conv2d(3, 64)
            net.maxpool()
            net.conv2d(3, 128)
            net.conv2d(3, 128)
            net.maxpool()
            net.conv2d(1, self.flags['NUM_CLASSES'], activation_fn=tf.nn.sigmoid)
            net.noisy_and(self.flags['NUM_CLASSES'])
            self.P_i = net.get_output()
            net.fc(self.flags['NUM_CLASSES'])
            self.y_hat = net.get_output()
            self.logits = tf.nn.softmax(self.y_hat)

    def _optimizer(self):
        """ Set up loss functions and choose optimizer. Called by TensorBase. """
        const = 1/self.flags['BATCH_SIZE']
        self.xentropy_p = const * tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.P_i, name='xentropy_p'))
        self.xentropy_y = const * tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat, name='xentropy_y'))
        self.weight = self.flags['WEIGHT_DECAY'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.xentropy_p + self.xentropy_y + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.flags['LEARNING_RATE']).minimize(self.cost)

    def train(self):
        """ Run training function for num_epochs. Save model upon completion. """
        print('Training for %d epochs' % self.flags['NUM_EPOCHS'])

        for self.epoch in range(1, self.flags['NUM_EPOCHS'] + 1):
            for _ in tqdm(range(self.num_train_images)):

                # Get minibatches of data
                batch_x, batch_y = self.mnist.train.next_batch(self.flags['BATCH_SIZE'])
                batch_x = self.reshape_batch(batch_x)

                # Run a training iteration

                summary, loss, _ = self.sess.run([self.merged, self.cost, self.optimizer],
                                                   feed_dict={self.x: batch_x, self.y: batch_y})
                self._record_training_step(summary)
            if self.step % (self.flags['display_step']) == 0:
                # Record training metrics every display_step interval
                self._record_train_metrics(loss)

            ## Epoch finished
            # Save model
            if self.epoch % self.checkpoint_rate == 0:
                self._save_model(section=self.epoch)
            # Perform validation
            if self.epoch % self.valid_rate == 0:
                self.evaluate('valid')

    def evaluate(self, dataset):
        """ Evaluate network on the valid/test set. """

        # Initialize to correct dataset
        print('Evaluating images in %s set' % dataset)
        if dataset == 'valid':
            batch_x, batch_y = self.mnist.validation.next_batch(self.flags['BATCH_SIZE'])
            batch_x = self.reshape_batch(batch_x)
            num_images = self.num_valid_images
            results = self.valid_results
        else:
            batch_x, batch_y = self.mnist.test.next_batch(self.flags['BATCH_SIZE'])
            batch_x = self.reshape_batch(batch_x)
            num_images = self.num_test_images
            results= self.test_results

        # Loop through all images in eval dataset
        for _ in tqdm(range(num_images)):
            logits = self.sess.run([self.logits], feed_dict={self.x: batch_x})
            predictions = np.reshape(logits, [-1, self.flags['NUM_CLASSES']])
            correct_prediction = np.equal(np.argmax(self.valid_batch_y, 1), np.argmax(predictions, 1))
            results = np.concatenate((results, correct_prediction))

        # Calculate average accuracy and record in text file
        self.record_eval_metrics(dataset)

    #########################
    ##   Helper Functions  ##
    #########################

    def reshape_batch(self, batch):
        """ Reshape vector into image. Do not need if data that is loaded in is already in image-shape"""
        return np.reshape(batch, [self.flags['BATCH_SIZE'], 28, 28, 1])

    def _record_train_metrics(self, loss):
        """ Records the metrics at every display_step iteration """
        print("Batch Number: " + str(self.step) + ", Total Loss= " + "{:.6f}".format(loss))

    def _record_eval_metrics(self, dataset):
        """ Record the accuracy on the eval dataset """
        if dataset == 'valid':
            accuracy = np.mean(self.valid_results)
        else:
            accuracy = np.mean(self.test_results)
        print("Accuracy on %s Set: %f" % (dataset, float(accuracy)))
        file = open(self.flags['restore_directory'] + dataset + 'Accuracy.txt', 'w')
        file.write('%s set accuracy:' % dataset)
        file.write(str(accuracy))
        file.close()


def main():

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Faster R-CNN Networks Arguments')
    parser.add_argument('-n', '--RUN_NUM', default=0)  # Saves all under /save_directory/model_directory/Model[n]
    parser.add_argument('-e', '--NUM_EPOCHS', default=1)  # Number of epochs for which to train the model
    parser.add_argument('-r', '--RESTORE_META', default=0)  # Binary to restore from a model. 0 = No restore.
    parser.add_argument('-m', '--MODEL_RESTORE', default=1)  # Restores from /save_directory/model_directory/Model[n]
    parser.add_argument('-f', '--FILE_EPOCH', default=1)  # Restore filename: 'part_[f].ckpt.meta'
    parser.add_argument('-t', '--TRAIN', default=1)  # Binary to train model. 0 = No train.
    parser.add_argument('-v', '--EVAL', default=1)  # Binary to evaluate model. 0 = No eval.
    parser.add_argument('-l', '--LEARNING_RATE', default=1e-3, type=float)  # learning Rate
    parser.add_argument('-g', '--GPU', default=0)  # specify which GPU to use
    parser.add_argument('-s', '--SEED', default=123)  # specify the seed
    parser.add_argument('-d', '--MODEL_DIRECTORY', default='summaries/', type=str)  # To save all models
    parser.add_argument('-a', '--SAVE_DIRECTORY', default='conv_mil/', type=str)  # To save individual run
    parser.add_argument('-i', '--DISPLAY_STEP', default=500, type=int)  # how often to display metrics
    parser.add_argument('-b', '--BATCH_SIZE', default=128, type=int)  # size of minibatch
    parser.add_argument('-w', '--WEIGHT_DECAY', default=1e-7, type=float)  # decay on all Weight variables
    parser.add_argument('-c', '--NUM_CLASSES', default=10, type=int)  # number of classes. proly hard code.
    flags = vars(parser.parse_args())

    # Run model. Train and/or Eval.
    model = ConvMil(flags)
    if int(flags['TRAIN']) == 1:
        model.train()
    if int(flags['EVAL']) == 1:
        model.evaluate('test')

if __name__ == "__main__":
    main()
