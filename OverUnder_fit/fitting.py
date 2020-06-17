#! usr/bin/python3

# Program using the tutorial Overfitting and Underfitting from
# Tensorflow.org.
# Uses the Higgs dataset

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from matplotlib import pyplot as plt
import numpy as np

import pathlib
import shutil
import tempfile

# set up logs
logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

# get the Higgs data set
gz = tf.keras.utils.get_file('HIGGS.csv.gz', \
                        'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

# there are 28 features in the dataset
FEATURES = 28
# extract records
ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), \
                            compression_type='GZIP')

# repack data into feature, label pair. asterisk denotes unknown number of input
# parameters
def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:],1)
    return features, label

# creates batches of 10000 as individual records
packed_ds = ds.batch(10000).map(pack_row).unbatch()

# examine some of the data
for features, label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins = 101)
    plt.savefig('data.png',bbox_inches='tight')
    plt.clf()

# split into testing and validation data
N_val = int(1e3)
N_trn = int(1e4)
buff_sz = int(1e4)
Batch_sz = 500
Steps_per_Epoch = N_trn//Batch_sz

validate_ds = packed_ds.take(N_val).cache()
train_ds = packed_ds.skip(N_val).take(N_trn).cache()
# see what the train set looks like
print(train_ds)

# set the datasets into batches of appropriate sizes and shuffle the data
validate_ds = validate_ds.batch(Batch_sz)
train_ds = train_ds.shuffle(buff_sz).repeat().batch(Batch_sz)

# Show overfitting first
# also graudally reduce the learning rate at a function of number of epochs
# passed
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, \
                    decay_steps = Steps_per_Epoch*1000,
                    decay_rate=1, staircase=False)

# function returns opimization function
def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)

# make a graph of the learning rate by epoch
step = np.linspace(0,10000)
lr = lr_schedule(step)
lr_plt = plt.figure(1)
plt.plot(step/Steps_per_Epoch, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
lr_plt.savefig('learn_rate.png',bbox_inches='tight')

# function for early stopping and logging for each 100 epocsh
def get_callbacks(name):
    return [ tfdocs.modeling.EpochDots(), \
            tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', \
            patience=200),
            tf.keras.callbacks.TensorBoard(logdir/name),]

# function for compiling and fitting model
def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer, \
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), \
        metrics = [tf.keras.losses.BinaryCrossentropy(from_logits=True, \
        name='binary_crossentropy'),'accuracy'])
    print(model.summary())

    history = model.fit( \
        train_ds, steps_per_epoch = Steps_per_Epoch, epochs = max_epochs, \
        validation_data = validate_ds, callbacks = get_callbacks(name),
        verbose=0 )

    return history

# very small model first
tiny_model = tf.keras.Sequential([ \
    # activation function is exp. linear := a*(exp(x)-1) if x < 0, x else
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)), \
    layers.Dense(1)])

# set up a dict for tracking
size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

# see how tiny did
plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', \
        smoothing_std=10)

tiny_plt = plt.figure(2)
plotter.plot(size_histories)
tiny_plt.savefig('tiny_plt.png',bbox_inches='tight')

# try a two layer model
small_model = tf.keras.Sequential([ \
        layers.Dense(16, activation='elu', input_shape=(FEATURES,)), \
        layers.Dense(16, activation='elu'), \
        layers.Dense(1)])

size_histories['small'] = compile_and_fit(small_model,'sizes/Small')

# three layers with more neurons
medium_model = tf.keras.Sequential([ \
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)), \
    layers.Dense(64, activation='elu'),
    layers.Dense(1)])

size_histories['medium'] = compile_and_fit(medium_model, "sizes/Medium")

# This one is definitely too large
large_model = tf.keras.Sequential([ \
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)), \
    layers.Dense(512, activation='elu'), \
    layers.Dense(512, activation='elu'), \
    layers.Dense(512, activation='elu'), \
    layers.Dense(1)])

size_histories['large'] = compile_and_fit(large_model, "sizes/Large")

# visualize them all
all_plt = plt.figure(4)
plotter.plot(size_histories)
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5,0.7])
plt.xscale('log')
plt.xlabel('Epochs [log]')
all_plt.savefig('all_plts.png',bbox_inches='tight')

##### Now for strategies to prevent overfitting ####
# copy the tiny model logs for comparison

shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')

# weight regularization - force weights to be within a range of each other
# L1 - cost added is proportional to abs(weight coefficient) - L1 norm
#   pushes weights to zero encouraging sparseness
# L2 - cost added is proportional to weight^2, aka weight decay - L2 norm
#   pushes to small weights, but not sparse
regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']
# set up model using L2 regularization
l2_model = tf.keras.Sequential([ \
    layers.Dense(512, activation='elu', \
        # 0.001 is the loss coefficient := 0.001 * weight**2
        kernel_regularizer=regularizers.l2(0.001), input_shape=(FEATURES,)), \
    layers.Dense(512, activation='elu', \
        kernel_regularizer=regularizers.l2(0.001)), \
    layers.Dense(512, activation='elu', \
        kernel_regularizer=regularizers.l2(0.001)), \
    layers.Dense(1)])

regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")

# see how it compares
l2_plt = plt.figure(5)
plotter.plot(regularizer_histories)
plt.ylim([0.5,0.7])
l2_plt.savefig('l2_com.png',bbox_inches='tight')

# this version adds the weight penalties to total loss. another version adds to
# raw loss and has a weight decay, 'decoupled weight decay'
### Now for Dropout ###

dropout_model = tf.keras.Sequential([ \
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)), \
    layers.Dropout(0.5), \
    layers.Dense(512, activation='elu'), \
    layers.Dropout(0.5), \
    layers.Dense(512, activation='elu'), \
    layers.Dropout(0.5), \
    layers.Dense(512, activation='elu'), \
    layers.Dropout(0.5), \
    layers.Dense(1) ])

regularizer_histories['dropout'] = compile_and_fit(dropout_model, \
                            "regularizers/dropout")

# See them all
drp_plt = plt.figure(6)
plotter.plot(regularizer_histories)
plt.ylim([0.5,0.7])
drp_plt.savefig('drp_com.png',bbox_inches='tight')

# both strategies together
combo_model = tf.keras.Sequential([ \
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), \
        activation='elu', input_shape=(FEATURES,)), \
    layers.Dropout(0.5), \
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), \
        activation='elu'), \
    layers.Dropout(0.5), \
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), \
        activation='elu'), \
    layers.Dropout(0.5), \
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), \
        activation='elu'), \
    layers.Dropout(0.5), \
    layers.Dense(1)])

regularizer_histories['combined'] = compile_and_fit(combo_model, \
                            "regularizers/combined")

# Now see all three
cmb_plt = plt.figure(7)
plotter.plot(regularizer_histories)
plt.ylim([0.5,0.7])
cmb_plt.savefig('drp_com.png',bbox_inches='tight')
