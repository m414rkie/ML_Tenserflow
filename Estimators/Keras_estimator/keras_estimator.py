#! usr/bin/python3

# Code from tutorial at tensorflow.org. Creates a simple custom estimator.

# Jon Parsons
# 7-5-2020

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

# make a simple model
model = tf.keras.models.Sequential([tf.keras.layers.Dense(16, activation='relu', \
    input_shape=(4,)), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(3)])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
    optimizer='adam')
print(model.summary())

# create an input function to control the input of data. No arguments and
# returns a tf.data.Dataset
def input_fn():
    split = tfds.Split.TRAIN
    dataset = tfds.load('iris', split=split, as_supervised=True)
    dataset = dataset.map(lambda features, labels: ({'dense_input':features}, \
            labels))
    dataset = dataset.batch(32).repeat()
    return dataset

# test the input function
for features_batch, labels_batch in input_fn().take(1):
    print(features_batch)
    print(labels_batch)

# making an estimator
import tempfile
model_dir = tempfile.mkdtemp()
keras_estimator = tf.keras.estimator.model_to_estimator(keras_model=model, \
        model_dir=model_dir)

# train and evaluate
keras_estimator.train(input_fn=input_fn, steps=500)
eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
print('Eval result: {}'.format(eval_result))
