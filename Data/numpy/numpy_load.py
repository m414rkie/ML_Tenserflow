#! usr/bin/python3

# This code is from the tutorial on how to load numpy arrays into Tensorflow
# Taken from Tensorflow.org

# Jon Parsons
# 6-23-2020

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# It is the hand-drawn numerals dataset
# data file is a .npz file found here:
DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

# get the data
path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']
# set up the the datasets, combines values into an array(s)
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

# just for practice lets look at the images
fig = plt.figure(1)
for i in range(6):
    plt.subplot(3,2,i+1)
    plt.tight_layout()
    plt.imshow(train_examples[i],cmap='gray',interpolation='none')
    plt.title("Label: {}".format(train_labels[i]))
    plt.xticks([])
    plt.yticks([])
fig.savefig("input_data.png",bbox_inches='tight')

# shuffle and batch
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# make a simple model. no normalizing or flattening.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=10)
print(model.history.history.keys())
model.evaluate(test_dataset)

# lets see how the simple model does
pre = plt.figure(4)
ims = np.random.randint(1,500,6)
for i, im in enumerate(ims):
    prediction = model.predict(test_examples[im].reshape(1,28,28,1))
    plt.subplot(3,2,i+1)
    plt.tight_layout()
    plt.imshow(test_examples[im], cmap='gray', interpolation='none')
    plt.title("Prediction: {}\n Actual: {}".format(prediction.argmax(), \
                test_labels[im]))
pre.savefig("pred.png",bbox_inches='tight')
