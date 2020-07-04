#! usr/bin/python3

# code from tutorial at tensorflow.org. This tutorial is on building a linear
# model from the prebuilt estimators using the titanic dataset.

# Jon Parsons
# 7-3-2020

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

# Get the data
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# see the first bit
print(dftrain.head())
print(dftrain.describe())
# get some graphs of the data
# age distibution
ages = plt.figure(1)
plt.title('Ages of the Passengers')
plt.ylabel('Number')
plt.xlabel('Age')
plt.hist(dftrain.age,bins=20)
ages.savefig("ages.png",bbox_inches='tight')

# begin model things
CATEGORICAL_COLUMNS = ['sex','n_siblings_spouses','parch','class','deck', \
        'embark_town','alone']
NUMERIC_COLUMNS = ['age','fare']

feature_columns = []

# create the dict
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list( \
        feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))

# create the input function here
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

ds = make_input_fn(dftrain, y_train, batch_size=10)()
## Skipping data inspection parts
# simple estimator
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)
print("Simple Estimator")
print(result)

# more complex estimator
# cross these two columns together
age_x_gender = tf.feature_column.crossed_column(['age','sex'], hash_bucket_size=100)

derived_feature_columns = [age_x_gender]
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+ \
        derived_feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print("With Crossed Columns")
print(result)

# make predictions with a distribution of the stuff
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

preds = plt.figure(2)
plt.title('Prediction Distribution')
plt.ylabel('Number')
plt.xlabel('Prediction of Survival')
plt.hist(probs,bins=20)
preds.savefig("preds.png",bbox_inches='tight')
