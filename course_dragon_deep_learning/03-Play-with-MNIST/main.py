import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


(train_images, train_labels), _ = keras.datasets.mnist.load_data()
print('datasets:', train_images.shape, train_labels.shape)

train_images = tf.convert_to_tensor(train_images, dtype=tf.float32) / 255.0
ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
ds = ds.batch(32).repeat(10)

model = keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

model.build(input_shape = (None, 28*28))
model.summary()

optimizer = keras.optimizers.SGD(lr=0.01)
acc_meter = keras.metrics.Accuracy()

for step, (x,y) in enumerate(ds):
    with tf.GradientTape() as tape:
        x = tf.reshape(x, (-1, 28*28))
        logits = model(x)
        y_onehot = tf.one_hot(y, depth = 10)
        loss = tf.square(logits - y_onehot)
        loss = tf.reduce_sum(loss) / 32

    acc_meter.update_state(tf.argmax(logits, axis=1), y)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if step % 200 == 0:
        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
        acc_meter.reset_states()
