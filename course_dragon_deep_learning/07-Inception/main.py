import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(2019)
np.random.seed(2019)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


######### 数据处理模块
def mnist_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_images = np.expand_dims(train_images, axis = 3)
    test_images = np.expand_dims(test_images, axis = 3)
    ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(256)
    ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(10000).batch(256)
    return ds_train,ds_test

class ConvBNRelu(keras.Model):
    def __init__(self, ch, kernel_size = 3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = keras.Sequential([
            layers.Conv2D(ch, kernel_size, strides=strides, padding=padding),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x, training=True):
        x = self.model(x, training=training)
        return x

class InceptionBL(keras.Model):
    def __init__(self, ch, strides=1):
        super(InceptionBL, self).__init__()
        self.ch = ch
        self.strides = strides
        self.conv1 = ConvBNRelu(ch, strides = strides)
        self.conv2 = ConvBNRelu(ch, kernel_size = 3, strides=strides)
        self.conv3_1 = ConvBNRelu(ch, kernel_size = 3, strides=strides)
        self.conv3_2 = ConvBNRelu(ch, kernel_size=3, strides=1)

        self.pool = layers.MaxPool2D(3, strides=1, padding='same')
        self.pool_conv = ConvBNRelu(ch, strides=strides)

    def call(self, x, training=True):
        x1 = self.conv1(x, training)
        x2 = self.conv2(x, training)
        x3_1 = self.conv3_1(x, training)
        x3_2 = self.conv3_2(x3_1, training)

        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training)
        x = tf.concat([x1, x2, x3_2, x4], axis=3)
        return x


class Inception(keras.Model):
    def __init__(self, num_layers, num_classes, init_ch = 16, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.init_ch = init_ch
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.conv1 = ConvBNRelu(init_ch)
        self.blocks = keras.Sequential(name = 'dynamic_blocks')

        for block_id in range(num_layers):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBL(self.out_channels, strides=2)
                else:
                    block = InceptionBL(self.out_channels, strides=1)
                self.blocks.add(block)
            self.out_channels *= 2
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, x, training = True):
        out = self.conv1(x, training)
        out = self.blocks(out, training)
        out = self.avg_pool(out)
        out = self.fc(out)
        return out



if __name__ == '__main__':
    ds_train, ds_test = mnist_dataset()
    ############# 模型开始训练
    model = Inception(2, 10)
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()

    optimizer = keras.optimizers.Adam(lr=1e-3)
    criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
    acc = keras.metrics.Accuracy()

    for epoch in range(100):
        for step, (x, y) in enumerate(ds_train):
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = criteon(tf.one_hot(y, depth=10), logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 10 == 0:
                print("epoch:", epoch, "step:", step, "loss:", loss.numpy())

        for x, y in ds_test:
            logits = model(x, training=False)
            pred = tf.argmax(logits, axis=1)
            acc.update_state(y, pred)
        print("epoch:", epoch, "accuracy:", acc.result().numpy())
        acc.reset_states()



