#! usr/bin/python3

# This program is taken from the tutorials at Tensorflow.org. This tutorial
# deals with how to handle csv data.
# Dataset is the Titanic data set and will predict likelyhood of survival.

# Jon Parsons
# 6-22-2020

import numpy as np

import tensorflow as tf
import functools

import pandas as pd
# define numpy printoptions for niceness of numbers
np.set_printoptions(precision=3, suppress=True)

# Where the data is
TRAIN_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

# get the data
train_path = tf.keras.utils.get_file("train.csv",TRAIN_URL)
test_path = tf.keras.utils.get_file("eval.csv",TEST_URL)

# explicitly handle the 'survived' column since it is the predictee
LABEL_COL = 'survived'
LABELS = [0,1]

# get the data using the inbuilt csv maker from tensorflow
## NOTE: **kwargs is passing keyworded arguements of unknown number, kwargs is
# just convention and not needed.
def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path, batch_size=5, label_name=LABEL_COL, na_value="?",
        num_epochs=1, ignore_errors=True, **kwargs)
    return dataset

raw_train_data = get_dataset(train_path)
raw_test_data = get_dataset(test_path)

# function for displaying data in batch size chunks
def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key,value.numpy()))

# look at some data
show_batch(raw_train_data)

# This section is not needed for this dataset but illustrates how to manually
# insert column titles into data
COL_NAMES = ['survived','sex','age','n_siblings_spouses','parch','fare', \
            'class','deck','embark_town','alone']

temp_set = get_dataset(train_path, column_names=COL_NAMES)
# does it compare to the first set shown??????
show_batch(temp_set)

# Which columns are to be used? Set them here
SELECT_COLS = ['survived','age','n_siblings_spouses','class','deck','alone']
temp_set = get_dataset(train_path, select_columns=SELECT_COLS)
show_batch(temp_set)

## Section on preprocessing
# redo select columns
SELECT_COLS = ['survived','age','n_siblings_spouses','parch','fare']
DEFAULTS = [0,0.0,0.0,0.0,0.0] # matches number of columns
temp_set = get_dataset(train_path, select_columns=SELECT_COLS, \
            column_defaults=DEFAULTS)
show_batch(temp_set)

# get some example data
example_batch, labels_batch = next(iter(temp_set))

# pack together the columns
def pack(features, label):
    return tf.stack(list(features.values()), axis=-1), label

packed_set = temp_set.map(pack)

# how does the newly packed data look?
for features, labels in packed_set.take(1):
    print(features.numpy(),"\n",labels.numpy())

example_batch, labels_batch = next(iter(temp_set))

# Now we seperate out numeric fields
# this is a general preprocessor that will select lists of numeric features
# and pack into a single column
class PackNumericFeatures(object):
    def __init__(self, names): # initialize the instance, will have property
        self.names = names     # names

    def __call__(self, features, labels): # define functions that happen
            # on instantiating the class
        numeric_features = [features.pop(name) for name in self.names]
        # remove name from object
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        # make the values float32
        numeric_features = tf.stack(numeric_features, axis=-1)
        # LIFO stack values together
        features['numeric'] = numeric_features

        return features, labels

NUMERIC_FEATURES = ['age','n_siblings_spouses','parch','fare']
# map executes the packing class for each value in passed array
packed_train_set = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
packed_test_set = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))
# how does this new set look? Should be categorical columns with combined numerics
show_batch(packed_train_set)

example_batch, labels_batch = next(iter(packed_train_set))

# normalize the continuous data
# read in using the pandas csv reader
desc = pd.read_csv(train_path)[NUMERIC_FEATURES].describe()
print(desc)
# pandas gave these to us via the describe function
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

# function for normalizing input data
def normalize_numeric_data(data, mean, std):
    return (data-mean)/std
# see the normalized data, not sure that partial was needed here
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
numeric_column = tf.feature_column.numeric_column('numeric', \
        normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]

print(numeric_column)

# see the example data
print(example_batch['numeric'])

# start setting up model layers, NOTE: setting up the data this way requires
# knowing the mean of each column, either through calculation or being known
numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
print(numeric_layer(example_batch).numpy())

## Set up the categorical data
# make a dict with the options (making use of the fact we know the categories)
CATEGORIES = {
    'sex' : ['male','female'],
    'class' : ['First','Second','Third'],
    'deck' : ['A','B','C','D','E','F','G','H','I','J'],
    'embark_town' : ['Cherbourg','Southhampton','Queenstown'],
    'alone' : ['y','n']
}

# begin category packing
categorical_cols = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_cols.append(tf.feature_column.indicator_column(cat_col))

print(categorical_cols)

# set up the category layer for the model
categorical_layer = tf.keras.layers.DenseFeatures(categorical_cols)
# see how it will look with the first set
print(categorical_layer(example_batch).numpy()[0])

# combine the two preprocessing layers together
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_cols + \
            numeric_columns)
# see how it looks with the first data sample
print(preprocessing_layer(example_batch).numpy()[0])

# finish the model
model = tf.keras.Sequential([preprocessing_layer, \
        tf.keras.layers.Dense(128, activation='relu'), \
        tf.keras.layers.Dense(128, activation='relu'), \
        tf.keras.layers.Dense(1)])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',metrics=['accuracy'])

# shuffle the data
train_data = packed_train_set.shuffle(500)
test_data = packed_test_set
# fit the model
model.fit(train_data, epochs=20)
# test the model
test_loss, test_accuracy = model.evaluate(test_data)
print("Test Loss: {} \n Test Accuracy {}".format(test_loss,test_accuracy))
# predict some data
predictions = model.predict(test_data)
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    prediction = tf.sigmoid(prediction).numpy()
    print("Predicted Survival: {:.2%}".format(prediction[0]), "| Actual: ",
        ("SURVIVED" if bool(survived) else 'DIED'))
