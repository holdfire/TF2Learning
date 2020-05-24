import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

class Regressor(keras.layers.Layer):

    def __init__(self):
        super(Regressor, self).__init__()

        self.w = self.add_variable('meanless_name', [13,1])
        self.b = self.add_variable('meanless_name', [1])

        print(self.w.shape, self.b.shape)
        print(type(self.w), tf.is_tensor(self.w), self.w.name)
        print(type(self.b), tf.is_tensor(self.b), self.b.name)

    def call(self, x):
        x = tf.matmul(x, self.w) + self.b
        return x


def main():
    tf.random.set_seed(2019)
    np.random.seed(2019)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    (train_data, train_labels), (test_data, test_labels) = keras.datasets.boston_housing.load_data()
    train_data, test_data = train_data.astype(np.float32), test_data.astype(np.float32)
    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)

    ds_train = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(64)
    ds_test = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(102)

    model = Regressor()
    criteon = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate = 1e-2)

    for epoch in range(200):
        for step, (x,y) in enumerate(ds_train):
            with tf.GradientTape() as tape:
                logits = model(x)
                logits = tf.squeeze(logits, axis=1)
                loss = criteon(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(epoch, 'loss:', loss.numpy())

        if epoch % 10 == 0:
            for x,y in ds_test:
                logits = model(x)
                logits =  tf.squeeze(logits, axis = 1)
                loss = criteon(y, logits)
                print(epoch, 'test loss:', loss.numpy())


if __name__ == '__main__':
    main()

