#! usr/bin/python3

# Program from the tutorial for loading data into tensorflow using pandas
# Found at Tensorflow.org

# dataset from Clevelan Clini Foundation for Heart Disease

# Jon Parsons
# 6-23-2020

import pandas as pd
import tensorflow as tf
# get the file
csv_file = tf.keras.utils.get_file('heart.csv', \
        'https://storage.googleapis.com/applied-dl/heart.csv')

# get the file data into pandas
df = pd.read_csv(csv_file)
# print the first few entries and see data types
print(df.head())
print(df.dtypes)
# gonna have to convert the 'tha1' column from object to discrete numeric
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes
# see what it is now
print(df.head())

# begin formatting the data for use in tensorflow
target = df.pop('target')

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

for feat, targ in dataset.take(5):
    print('Features: {}, Target: {}'.format(feat, targ))

# set constant
tf.constant(df['thal'])

# shuffle and batch
train_dataset = dataset.shuffle(len(df)).batch(1)

def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)])

    model.compile(optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)

# we can also use a dict.
# and stack layers as seen below
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1)(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)
model_func.compile(optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

# this method preserces column structure by converting to a dict and taking slices
dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), \
    target.values)).batch(16)

for dict_slice in dict_slices.take(1):
    print(dict_slice)

model_func.fit(dict_slices, epochs=15)
