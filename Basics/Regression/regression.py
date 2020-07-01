#! usr/bin/python3

# Program that will utilize Tensorflow to predict the MPG of cars from the
# 70's and 80's. Taken from a tutorial at tensorflow.org, regression tutorial.

# Jon Parsons
# 6-14-2020

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

################################################################################
# function that builds ad compiles the model
def model_maker(shp):
    model = keras.Sequential([ \
        layers.Dense(64, activation='relu', input_shape=[shp]), \
        layers.Dense(64, activation='relu'), \
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    # learning rate of 0.001
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])
    # mae = mean absolute error := 1/n SUM(y_j - yavg_j)
    return model

################################################################################

# get data from archive
dataset_path = keras.utils.get_file("auto-mpg.data", \
 "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

# import to python
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight', \
                'Acceleration','Model Year','Origin']
raw_set = pd.read_csv(dataset_path, names=column_names, na_values="?", \
                        comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_set.copy()

# Show some of the data
print(dataset.tail())

# clean data of unknowns
dataset = dataset.dropna()
# Convert origin from category to numeric (USA - 1, Europe - 2, Japan - 3)
dataset['Origin'] = dataset['Origin'].map({1:'USA',2:'Europe',3:'Japan'})
# get the dummy values
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

# Split into training and validation
train_set = dataset.sample(frac=0.8, random_state=0)
test_set = dataset.drop(train_set.index)

# Lets make some pictures to visualise the data
dat_vis = sns.pairplot(train_set[["MPG","Cylinders","Displacement","Weight"]], \
            diag_kind="kde")
## NOTE: seaborn does not like saving as jpg ##
plt.savefig("data_vis.png",bbox_inches='tight')
plt.clf()

# Display some statistics about the data
train_stats = train_set.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# remove the desired value from the set
train_labels = train_set.pop('MPG')
test_labels= test_set.pop('MPG')

# normalize the data
def norm(x):
    return (x - train_stats['mean'])/train_stats['std']
normed_train_data = norm(train_set)
normed_test_data = norm(test_set)

# build the model
model = model_maker(len(train_set.keys()))
# did it work?
print(model.summary())

# train model
epochs = 1000

history = model.fit(normed_train_data, train_labels, epochs=epochs, \
                    validation_split=0.2, verbose=0, \
                    callbacks=[tfdocs.modeling.EpochDots()])

# training visualizations
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
mae_fig = plotter.plot({'Basic':history}, metric='mae')
plt.ylim([0,10])
plt.ylabel('MAE [MPG]')
plt.savefig("mae.png",bbox_inches='tight')
plt.clf()

mse_fig = plotter.plot({'Basic':history}, metric='mse')
plt.ylim([0,20])
plt.ylabel('MSE [MPG^2]')
plt.savefig('mse.png',bbox_inches='tight')
plt.clf()
# reset the model and train with a condition to stop training when a training
# condition stops showing improvement
model = model_maker(len(train_set.keys()))
# patience is number of epochs to wait for improvement
stopper = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
epochs = 1000
stopper_history = model.fit(normed_train_data, train_labels, epochs=epochs, \
                    validation_split=0.2, verbose=0, \
                    callbacks=[stopper, tfdocs.modeling.EpochDots()])

# see how it did this time
num_epochs = len(stopper_history.history['loss'])
print("Number of Epochs ran with early stop: ", num_epochs)

stp_fig = plotter.plot({'Early Stopping':stopper_history}, metric='mae')
plt.ylim([0,10])
plt.ylabel('MAE [MPG]')
plt.savefig('stopped_mae.png',bbox_inches='tight')
plt.clf()
# see how the model stacks up with the testing set
test_pred = model.predict(normed_test_data).flatten()

pre = plt.axes(aspect='equal')
plt.scatter(test_labels, test_pred)
plt.xlabel("True MPG")
plt.ylabel("Predicted MPG")
plt.xlim(0,50)
plt.ylim(0,50)
plt.savefig('pred.png',bbox_inches='tight')
plt.clf()
# look at error distribution
error = test_pred - test_labels
er_plt = plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.savefig('err_bins.png',bbox_inches='tight')
plt.clf()
