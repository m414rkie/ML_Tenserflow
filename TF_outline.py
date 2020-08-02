#! usr/bin/python

# This is an outline for Tensorflow projects. Includes basic implementations of
# functions that create, save, and recover models as well as creating plots of
# accuracy and loss. This outline is generalized for Tensorflow with Keras.

# Jon Parsons
# 8-1-2020

################################################################################
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

################################################################################
def make_model(***,loss,compile=False):
# creates and compiles the model.
# loss should be something like
# tf.keras.losses.sparse_categorical_crossentropy(from_logits=True)
    model = ([

        ])

    if compile:
        model.compile(optmizer='adam',loss=loss,metrics=['accuracy'])

    return model

################################################################################
def save_model(model,model_name):
# saves the model.
# model - model variable name
# model_name - save model as

    model.save(model_name)

################################################################################
def exist_model(model_name):
# Checks if a model exists by that name
# model_name - saved name of model

    exist = os.path.exists(model_name)

    return exist

################################################################################
def load_model(saved_name):
# loads in a saved model
# saved_name - name model is saved under

    recovered_model = tf.keras.models.load_model(saved_name)

    return recovered_model

################################################################################
def fit_model(model,epochs,dataset,callback):
# fits the model
# model - model to fit
# epochs - number of epochs to train for
# dataset - training dataset, expects tuples of (data, label)
# callback - callback function. Should be of form
# tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True)

    history = model.fit(dataset,epochs=epochs,callbacks=[callback])

    return history

################################################################################
def model_evaluate(model,test_data):
# evaluates the model
# model - model to evaluate
# test_data - data to use

    loss, acc = model.evaluate(test_data)

    return loss, acc

################################################################################
def hist_plots(history,metric,name,validation=False):
# creates and saves plots of how the model behaves during training
# history - contains the values
# metric - value to plot
# name - name of output
# validation - mark true if you want validation data plotted

    plt.plot(history.history[metric])
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    if validation:
        plt.plot(history.history['val_'+metric])
        plt.legend([metric, 'Validation '+metric])
    else:
        plt.legend([metric])

    plt.savefig(name, bbox_inches='tight')
    plt.clf()
