import os
import tensorflow as tf
from tensorflow import keras
import argparse
import numpy as np

from VGGNet import VGG16

####### 设置系统变量，将tensorflow的输出信息等级设低，不报warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

argparser = argparse.ArgumentParser()
argparser.add_argument('--train_dir', type=str, default='./', help = "Directory where to write event logs and checkpoints")
argparser.add_argument('--max_steps', type=int, default=1000000, help = "Number of batches to run.")
argparser.add_argument('--log_device_placement', action='store_true', help = "whether to log device placement.")
argparser.add_argument('--log_frequency', type=int, default=10, help = "how often to log results to the console.")


def normalize(X_train, X_test):
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    mean = np.mean(X_train)
    std = np.std(X_train)
    print('mean:', mean, 'std:', std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

def prepare_cifar(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return x,y

def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits))

def main():
    tf.random.set_seed(2019)
    print('loading data...')
    (train_images, train_labels),(test_images, test_labels) = keras.datasets.cifar10.load_data()
    print("train_image shape:", train_images.shape, "train_labels shape:", train_labels.shape)
    print("test_images shape:", test_images.shape, "test_labels shape:", test_labels.shape)

    ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).map(prepare_cifar).shuffle(50000).batch(256)
    ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).map(prepare_cifar).shuffle(10000).batch(256)
    print("processing data done!")

    model = VGG16([32,32,3])
    criteon = keras.losses.CategoricalCrossentropy(from_logits = True)
    metric = keras.metrics.CategoricalAccuracy()
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)

    for epoch in range(250):
        for step,(x,y) in enumerate(ds_train):
            y =  tf.squeeze(y, axis = 1)
            y = tf.one_hot(y, depth = 10)
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = criteon(y, logits)
                metric.update_state(y, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 40 == 0:
                print('epoch:', epoch, 'step:', step, 'loss:', float(loss), 'acc:', metric.result().numpy())
                metric.reset_states()

        if  epoch % 1 == 0:
            metric = keras.metrics.CategoricalAccuracy()
            for x,y in ds_test:
                y = tf.squeeze(y, axis = 1)
                y = tf.one_hot(y, depth = 10)
                logits = model.predict(x)
                metric.update_state(y, logits)
            print("test_acc:", metric.result().numpy())
            metric.reset_states()


if __name__ == "__main__":
    main()

