import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

########## 数据处理模块
def prepare_fashion_mnist_data(x,y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x,y

def fashion_mnist_dataset():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    train_labels = tf.one_hot(train_labels, depth = 10)
    test_labels = tf.one_hot(test_labels, depth = 10)
    ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).map(prepare_fashion_mnist_data).shuffle(60000).batch(100)
    ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).map(prepare_fashion_mnist_data).shuffle(10000).batch(1000)
    return ds_train, ds_test

########### 模型+训练
def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    ds_train, ds_test = fashion_mnist_dataset()

    model = keras.Sequential([
        layers.Reshape(target_shape = (28*28,), input_shape = (28, 28)),
        layers.Dense(200, activation='relu'),
        # layers.Dense(200, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(10)])

    # ####### 方式一：构建动态图
    # model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    #               loss = keras.losses.CategoricalCrossentropy(from_logits = True),
    #               metrics = ['accuracy'])
    # model.fit(ds_test.repeat(), epochs = 10, steps_per_epoch = 600,
    #           validation_data= ds_test.repeat(),
    #           validation_steps = 2)

    ####### 方式二：构建静态图
    optimizer = keras.optimizers.Adam(lr=0.001)
    criteon = keras.losses.CategoricalCrossentropy(from_logits = True)
    metric = keras.metrics.Accuracy()

    for epoch in range(20):
        for step, (x,y) in enumerate(ds_train):
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = criteon(y, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print("epoch:", epoch, "step:", step, "loss:", loss.numpy())

        if epoch % 5 == 0:
            for x,y in ds_test:
                logits = model(x)
                acc = metric(y, logits)
                print("epoch:", epoch, "acc:", acc.numpy())



if __name__ == '__main__':
    main()


