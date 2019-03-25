import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist


class Dataset:
    def __init__(self, num_classes, img_rows, img_cols):
        self.num_classes = num_classes
        self.img_rows = img_rows
        self.img_cols = img_cols


class Mnist(Dataset):
    def __init__(self):
        super(Mnist, self).__init__(num_classes=10, img_rows=72, img_cols=72)
        self.input_shape = (1, self.img_rows, self.img_cols) if K.image_data_format() == 'channels_first' \
            else (self.img_rows, self.img_cols, 1)
        self.x_train, self.x_test, self.y_train, self.y_test = self.build_data()

    def expand_img(self, embed_img, train_img):
        """Transforms a single MNIST digit into a larger image with 0's around it"""
        blank = np.zeros((self.img_rows, self.img_cols))
        r = 36 - 28
        indexes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        imgs = [embed_img, embed_img, embed_img, train_img]
        np.random.shuffle(indexes)
        for (x, y), img in zip(indexes, imgs):
            xcor = np.random.randint(36 * x, 36 * x + r)
            ycor = np.random.randint(36 * y, 36 * y + r)
            blank[xcor:xcor + 28, ycor:ycor + 28] = img
        return blank

    def build_data(self):
        """Returns the train and test datasets and their labels"""

        # load original mnist dataset and expand each number with embedded "0"s
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        embedding_img = x_train[1]
        x_train = np.array([self.expand_img(embedding_img, img) for img, label in zip(x_train, y_train)])
        x_test = np.array([self.expand_img(embedding_img, img) for img, label in zip(x_test, y_test)])

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)

        # normalize and cast
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        return x_train, x_test, y_train, y_test
