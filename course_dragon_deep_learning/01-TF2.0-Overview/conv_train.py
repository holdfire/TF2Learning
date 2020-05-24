import os
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from tensorflow import keras
from tensorflow.keras import datasets,layers,models

############ 先来处理数据
def mnist_dataset():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    # 对数据做归一化，以及通过numpy转换数据类型
    train_images, test_images = train_images / np.float32(255), test_images / np.float32(255)
    train_labels, test_labels = train_labels.astype(np.int64), test_labels.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    return train_dataset, test_dataset

train_ds,test_ds = mnist_dataset()
train_ds = train_ds.shuffle(60000).batch(100)
test_ds = test_ds.batch(100)



############ 构造模型
model = keras.Sequential([
    layers.Reshape(target_shape = (28,28,1), input_shape = (28,28,)),
    layers.Conv2D(filters = 2, kernel_size = 5, padding='same', activation = 'relu'),
    layers.MaxPool2D((2,2), (2,2), padding = 'same'),
    layers.Conv2D(filters = 4, kernel_size = 5, padding = 'same', activation = 'relu'),
    layers.MaxPool2D((2,2), (2,2), padding = 'same'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(rate=0.4),
    layers.Dense(10)])

compute_loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
compute_accuracy= keras.metrics.SparseCategoricalAccuracy()
optimizer = keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.5)


############ 训练过程
def train_step(model, optimizer, images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training = True)
        loss = compute_loss(labels, logits)
        compute_accuracy(labels, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(model, optimizer, datasets, log_freq = 50):
    avg_loss = keras.metrics.Mean('loss',dtype = tf.float32)
    for images,labels in datasets:
        loss = train_step(model, optimizer, images, labels)
        avg_loss(loss)

        if tf.equal(optimizer.iterations % log_freq, 0):
            print('step: ', int(optimizer.iterations),
                  'loss: ', avg_loss.result().numpy(),
                  'acc: ', compute_accuracy.result().numpy() )
            avg_loss.reset_states()
            compute_accuracy.reset_states()

############ 测试过程
def tes(model, dataset, step_num):
    avg_loss = keras.metrics.Mean('loss', dtype = tf.float32)
    for (images, labels) in dataset:
        logits = model(images, training=False)
        avg_loss(compute_loss(labels, logits))
        compute_accuracy(labels, logits)
    print('Model test set loss: {:0.4f}  accuracy: {:0.2f}%' .format(avg_loss.result(), compute_accuracy.result()*100))
    print('loss:', avg_loss.result(), 'acc: ', compute_accuracy.result())

# where to save checkpoints, tensorboard summaries, ect
MODE_DIR = 'D:/github_repository/TensorFlow/Dragon_DeepLearningWithTF2.0/01-TF2.0-Overview/tensorflow_mnist'

def apply_clean():
    if tf.io.gfile.exists(MODE_DIR):
        print('Removing existing model dir:{}'.format(MODE_DIR))
        tf.io.gfile.rmtree(MODE_DIR)

#apply_clean()

checkpoint_dir = os.path.join(MODE_DIR, 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

NUM_TRAIN_EPOCHES = 5

for i in range(NUM_TRAIN_EPOCHES):
    start = time.time()
    train(model, optimizer, train_ds, log_freq = 500)
    end = time.time()
    print('Train time for epoch #{} ({} total steps): {}'.format(i+1, int(optimizer.iterations), end - start))
    checkpoint.save(checkpoint_prefix)
    print('saved checkpoint.')

export_path = os.path.join(MODE_DIR, 'export')
tf.saved_model.save(model, export_path)
print('saved SavedModel for exporting.')


