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


# Global Dictionary of Flags
flags = {  # data_directory not needed
    'save_directory': 'summaries/',
    'model_directory': 'conv_mil/',
    'num_classes': 10,
    'image_dim': 28,
    'batch_size': 128,
    'display_step': 500,
    'weight_decay': 1e-7,
}


class ConvMil(Model):
    def __init__(self, flags_input, run_num):
        """ Initialize from Model class in TensorBase """
        super().__init__(flags_input, run_num)
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
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')
        self.y = tf.placeholder(tf.int32, shape=[None, flags['num_classes']], name='y')

    def _summaries(self):
        """ Write summaries out to TensorBoard. Called by TensorBase. """
        tf.summary.scalar("Total Loss", self.cost)
        tf.summary.scalar("Cross Entropy Loss", self.xentropy)
        tf.summary.scalar("Weight Decay Loss", self.weight)

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
            net.conv2d(1, self.flags['num_classes'], activation_fn=tf.nn.sigmoid)
            net.noisy_and(self.flags['num_classes'])
            self.y_hat = net.get_output()
            self.logits = tf.nn.softmax(self.y_hat)

    def _optimizer(self):
        """ Set up loss functions and choose optimizer. Called by TensorBase. """
        const = 1/self.flags['batch_size']
        self.xentropy = const * tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat, name='xentropy'))
        self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.xentropy + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=flags['learning_rate']).minimize(self.cost)

    def train(self):
        """ Run training function for num_epochs. Save model upon completion. """
        self.print_log('Training for %d epochs' % self.flags['num_epochs'])

        for self.epoch in range(1, self.flags['num_epochs'] + 1):
            for _ in tqdm(range(self.num_train_images)):

                # Get minibatches of data
                batch_x, batch_y = self.mnist.train.next_batch(flags['batch_size'])
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
            batch_x, batch_y = self.mnist.validation.next_batch(flags['batch_size'])
            batch_x = self.reshape_batch(batch_x)
            num_images = self.num_valid_images
            results = self.valid_results
        else:
            batch_x, batch_y = self.mnist.test.next_batch(flags['batch_size'])
            batch_x = self.reshape_batch(batch_x)
            num_images = self.num_test_images
            results= self.test_results

        # Loop through all images in eval dataset
        for _ in tqdm(range(num_images)):
            logits = self.sess.run([self.logits], feed_dict={self.x: batch_x})
            predictions = np.reshape(logits, [-1, self.flags['num_classes']])
            correct_prediction = np.equal(np.argmax(self.valid_batch_y, 1), np.argmax(predictions, 1))
            results = np.concatenate((results, correct_prediction))

        # Calculate average accuracy and record in text file
        self.record_eval_metrics(dataset)

    #########################
    ##   Helper Functions  ##
    #########################

    def reshape_batch(self, batch):
        """ Reshape vector into image. Do not need if data that is loaded in is already in image-shape"""
        return np.reshape(batch, [flags['batch_size'], self.flags['image_dim'], self.flags['image_dim'], 1])

    def _print_metrics(self):
        """ Helper function used by Model class in TensorBase """
        self.print_log("Seed: %d" % flags['seed'])

    def _record_train_metrics(self, loss):
        """ Records the metrics at every display_step iteration """
        self.print_log("Batch Number: " + str(self.step) + ", Total Loss= " + "{:.6f}".format(loss))

    def _record_eval_metrics(self, dataset):
        """ Record the accuracy on the eval dataset """
        if dataset == 'valid':
            accuracy = np.mean(self.valid_results)
        else:
            accuracy = np.mean(self.test_results)
        self.print_log("Accuracy on %s Set: %f" % (dataset, accuracy))
        file = open(self.flags['restore_directory'] + dataset + 'Accuracy.txt', 'w')
        file.write('%s set accuracy:' % dataset)
        file.write(str(accuracy))
        file.close()


def main():

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Faster R-CNN Networks Arguments')
    parser.add_argument('-n', '--run_num', default=0)  # Saves all under /save_directory/model_directory/Model[n]
    parser.add_argument('-e', '--epochs', default=1)  # Number of epochs for which to train the model
    parser.add_argument('-r', '--restore', default=0)  # Binary to restore from a model. 0 = No restore.
    parser.add_argument('-m', '--model_restore',default=1)  # Restores from /save_directory/model_directory/Model[n]
    parser.add_argument('-f', '--file_epoch', default=1)  # Restore filename: 'part_[f].ckpt.meta'
    parser.add_argument('-t', '--train', default=1)  # Binary to train model. 0 = No train.
    parser.add_argument('-v', '--eval', default=1)  # Binary to evalulate model. 0 = No eval.
    parser.add_argument('-l', '--learn_rate', default=1e-3)  # learning Rate
    parser.add_argument('-i', '--vis', default=0)  # enable visualizations
    parser.add_argument('-g', '--gpu', default=0)  # specifiy which GPU to use
    args = vars(parser.parse_args())

    # Set Arguments
    flags['seed'] = 1234  # Random seed
    flags['run_num'] = int(args['run_num'])
    flags['num_epochs'] = int(args['epochs'])
    flags['restore_num'] = int(args['model_restore'])
    flags['file_epoch'] = int(args['file_epoch'])
    flags['learning_rate'] = float(args['learn_rate'])
    flags['vis'] = True if (int(args['vis']) == 1) else False
    flags['gpu'] = int(args['gpu'])

    # Choose to restore or not.
    if args['restore'] == 0:
        flags['restore'] = False
    else:
        flags['restore'] = True
        flags['restore_file'] = 'part_' + str(args['file_epoch']) + '.ckpt.meta'

    # Run model. Train and/or Eval.
    model = ConvMil(flags, flags['run_num'])
    if int(args['train']) == 1:
        model.train()
    if int(args['eval']) == 1:
        model.evaluate('test')


if __name__ == "__main__":
    main()
