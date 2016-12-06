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
from TensorBase.tensorbase.data import Mnist

import tensorflow as tf
import numpy as np


# Global Dictionary of Flags
flags = {
    'data_directory': 'MNIST_data/',
    'save_directory': 'summaries/',
    'model_directory': 'conv_mil/',
    'restore': True,
    'restore_file': 'part_3.ckpt.meta',
    'datasets': 'MNIST',
    'num_classes': 10,
    'image_dim': 28,
    'batch_size': 128,
    'display_step': 500,
    'weight_decay': 1e-7,
    'lr_decay': 0.999,
    'lr_iters': [(5e-3, 5000), (5e-3, 7500), (5e-4, 10000), (5e-5, 10000)]
}


class ConvMil(Model):
    def __init__(self, flags_input, run_num):
        super().__init__(flags_input, run_num)
        self.print_log("Seed: %d" % flags['seed'])
        self.valid_results = list()
        self.test_results = list()

    def _define_data(self):
        self.data = Mnist(self.flags)
        self.num_train_images = self.data.num_train_images
        self.num_valid_images = self.data.num_valid_images
        self.num_test_images = self.data.num_test_images

    def _set_placeholders(self):
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')
        self.y = tf.placeholder(tf.int32, shape=[None, flags['num_classes']], name='y')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def _set_summaries(self):
        tf.scalar_summary("Total Loss", self.cost)
        tf.scalar_summary("Cross Entropy Loss", self.xentropy)
        tf.scalar_summary("Weight Decay Loss", self.weight)
        tf.image_summary("x", self.x)

    def _network(self):
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
            net.conv2d(1, 64)
            net.conv2d(1, 32)
            net.conv2d(1, self.flags['num_classes'], activation_fn=tf.nn.sigmoid)
            net.noisy_and(self.flags['num_classes'])
            self.y_hat = net.get_output()
            self.logits = tf.nn.softmax(self.y_hat)

    def _optimizer(self):
        const = 1/self.flags['batch_size']
        self.xentropy = const * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.y_hat, self.y, name='xentropy'))
        self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.xentropy + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

    def _generate_train_batch(self):
        self.train_batch_y, train_batch_x = self.data.next_train_batch(self.flags['batch_size'])
        self.train_batch_x = np.reshape(train_batch_x, [self.flags['batch_size'], self.flags['image_dim'], self.flags['image_dim'], 1])

    def _generate_valid_batch(self):
        self.valid_batch_y, valid_batch_x, valid_number, batch_size = self.data.next_valid_batch(self.flags['batch_size'])
        self.valid_batch_x = np.reshape(valid_batch_x, [batch_size, self.flags['image_dim'], self.flags['image_dim'], 1])
        return valid_number

    def _generate_test_batch(self):
        self.test_batch_y, test_batch_x, test_number, batch_size = self.data.next_test_batch(self.flags['batch_size'])
        self.test_batch_x = np.reshape(test_batch_x, [batch_size, self.flags['image_dim'], self.flags['image_dim'], 1])
        return test_number

    def _run_train_iter(self):
        rate = self.learn_rate * self.flags['lr_decay']
        self.summary, _ = self.sess.run([self.merged, self.optimizer],
                                        feed_dict={self.x: self.train_batch_x, self.y: self.train_batch_y,
                                                   self.lr: rate})

    def _run_train_summary_iter(self):
        rate = self.learn_rate * self.flags['lr_decay']
        self.summary, self.loss, _ = self.sess.run([self.merged, self.cost, self.optimizer],
                                                   feed_dict={self.x: self.train_batch_x, self.y: self.train_batch_y,
                                                              self.lr: rate})

    def _run_valid_iter(self):
        logits = self.sess.run([self.logits], feed_dict={self.x: self.valid_batch_x})
        predictions = np.reshape(logits, [-1, self.flags['num_classes']])
        correct_prediction = np.equal(np.argmax(self.valid_batch_y, 1), np.argmax(predictions, 1))
        self.valid_results = np.concatenate((self.valid_results, correct_prediction))

    def _run_test_iter(self):
        logits = self.sess.run([self.logits], feed_dict={self.x: self.test_batch_x})
        predictions = np.reshape(logits, [-1, self.flags['num_classes']])
        correct_prediction = np.equal(np.argmax(self.test_batch_y, 1), np.argmax(predictions, 1))
        self.test_results = np.concatenate((self.test_results, correct_prediction))

    def _record_train_metrics(self):
        self.print_log("Batch Number: " + str(self.step) + ", Total Loss= " + "{:.6f}".format(self.loss))

    def _record_valid_metrics(self):
        accuracy = np.mean(self.valid_results)
        self.print_log("Accuracy on Validation Set: %f" % accuracy)
        file = open(self.flags['restore_directory'] + 'ValidAccuracy.txt', 'w')
        file.write('Test set accuracy:')
        file.write(str(accuracy))
        file.close()

    def _record_test_metrics(self):
        accuracy = np.mean(self.test_results)
        self.print_log("Accuracy on Test Set: %f" % accuracy)
        file = open(self.flags['restore_directory'] + 'TestAccuracy.txt', 'w')
        file.write('Test set accuracy:')
        file.write(str(accuracy))
        file.close()


def main():
    flags['seed'] = np.random.randint(1, 1000, 1)[0]
    model_mil = ConvMil(flags, run_num=1)
    model_mil.valid()


if __name__ == "__main__":
    main()
