#! usr/bin/python3

# code from the tutorial found at tensorflow.org. Tutorial is the boosted trees
# which uses the Titanic data set.

# Jon Parsons
# 7-4-2020

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

# get the data
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval  =dfeval.pop('survived')

# set a seed for training - keeping tutorial seed for reproducibility
tf.random.set_seed(123)

# apparently you can plot csv data from pandas real easy
dftrain.sex.value_counts().plot(kind='barh')
plt.savefig("passenger_sexes.png",bbox_inches='tight')

# make input functions and data inputs
CATEGORICAL_COLUMNS = ['sex','n_siblings_spouses','parch','class','deck', \
            'embark_town','alone']
NUMERIC_COLUMNS = ['age','fare']

def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(feature_name, \
            vocab))

# combines the class and numeric columns into a single list
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name,vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, \
        dtype=tf.float32))

NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        dataset = dataset.repeat(n_epochs)
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    return input_fn

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

# start making the model
# simple linear model with results
linear_est = tf.estimator.LinearClassifier(feature_columns)
linear_est.train(train_input_fn, max_steps=100)
result = linear_est.evaluate(eval_input_fn)
print("Simple model results:")
print(pd.Series(result))

# now for the boosted trees model
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns, \
        n_batches_per_layer=n_batches)

est.train(train_input_fn, max_steps=100)

result = est.evaluate(eval_input_fn)
print("Boosted trees model:")
print(pd.Series(result))
