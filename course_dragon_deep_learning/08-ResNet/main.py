import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(2019)
tf.random.set_seed(2019)
assert tf.__version__.startswith('2.')

def mnist_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_images = np.expand_dims(train_images, axis = 3)
    test_images = np.expand_dims(test_images, axis = 3)
    train_labels = tf.one_hot(train_labels, depth = 10).numpy()
    test_labels = tf.one_hot(test_labels, depth = 10).numpy()
    ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(256)
    ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(10000).batch(256)
    return ds_train,ds_test

def conv3by3(channels, strides=1,kernel=(3,3)):
    return layers.Conv2D(channels, kernel, strides=strides, padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer())

class ResNetBlock(keras.Model):
    def __init__(self, channels, strides=1, residual_path=False):
        super(ResNetBlock, self).__init__()
        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path

        self.conv1 = conv3by3(channels,strides)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = conv3by3(channels)
        self.bn2 = layers.BatchNormalization()

        if residual_path:
            self.down_conv = conv3by3(channels,strides,kernel=(1,1))
            self.down_bn = layers.BatchNormalization()

    def call(self, inputs, training=True):
        residual = inputs
        x = self.bn1(inputs, training)
        x = layers.ReLU(x)
        x = self.conv1(x)
        x = self.bn2(x, training)
        x = layers.ReLU(x)
        x = self.conv2(x)

        if self.residual_path:
            residual = self.down_bn(inputs, training)
            residual = layers.ReLU(residual)
            residual = self.down_conv(residual)

        x = x + residual
        return x

class ResNet(keras.Model):
    def __init__(self, block_list, num_classes, initial_filters=16, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.block_list = block_list
        self.num_blocks = len(block_list)
        self.blocks = keras.Sequential(name='dynamic_blocks')

        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.conv_initial = conv3by3(self.out_channels)

        ###### build all blocks

