#! usr/bin/python

# Tutorial on text generation using an RNN from Tensorflow.org. Generates text
# using Shakspeare as a base.

# Jon Parsons
# 8-1-2020

################################################################################
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

def loss_comp(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True)


################################################################################
def make_model(vocab_size,embedding_dim,rnn_units,batch_size,compile=False):
# creates and compiles the model.

    model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size,embedding_dim, \
                    batch_input_shape=[batch_size,None]), \
                tf.keras.layers.GRU(rnn_units,return_sequences=True, \
                    stateful=True, recurrent_initializer='glorot_uniform'), \
                tf.keras.layers.Dense(vocab_size)
        ])

    if compile:
        model.compile(optimizer='adam', \
        loss=loss_comp,\
        metrics=['accuracy'])

    return model

################################################################################
def save_model(model,history,model_name):
# saves the model.
# model - model variable name
# model_name - save model as

    model.save(model_name)
    pd.DataFrame.from_dict(history.history).to_csv('history.csv',index=False)

################################################################################
def exist_model(model_name):
# Checks if a model exists by that name
# model_name - saved name of model
# save_dir - directory of save

    exist = os.path.exists(model_name)

    return exist

################################################################################
def load_model(saved_name):
# loads in a saved model
# save_dir - directory model is saved in
# saved_name - name model is saved under

    recovered_model = tf.keras.models.load_model(saved_name, compile=False)
    history = pd.read_csv('history.csv')

    return recovered_model, history

################################################################################
def fit_model(model,epochs,dataset,callback):
# fits the model
# model - model to fit
# epochs - number of epochs to train for
# dataset - training dataset, expects tuples of (data, label)
# callback - checkpoint saving
    history = model.fit(dataset,epochs=epochs,callbacks=[callback],)

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

################################################################################
################################################################################

# get the dataset
data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
path_to_file = tf.keras.utils.get_file('shakespeare.txt', data_url)

# examine the data a bit
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print("Length of text: {} characters\n".format(len(text)))

# print first 250 characters
print("First portion of the text:\n")
print(text[:250])

# get number of unique characters
vocab = sorted(set(text))
print("\nNumber of unique characters: {}".format(len(vocab)))

# begin text processing
# map strings to numerical vectors
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# see how things were translated
print("\nConversion map:\n")
print('{')
for char, _ in zip(char2idx, range(20)):
    print(' {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('}')

# create the dataset and batch the text into trainable sequences
seq_length = 100 # batch length
examples_per_epoch = len(text)//(seq_length + 1)

# make the dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# quick function to create a target text from the initial text
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# see how the model should be training
print("Based on the input, we should expect:\n")
for i, (input_idx, target_idx) in enumerate(dataset.take(5)):
    print("Step {:4d}".format(i))
    print(" input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print(" expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

# Hyperparams
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)

print("Dataset shape:\n")
print(dataset)

# we will compile in a bit
model = make_model(len(vocab),256,1024,BATCH_SIZE,"none")

# check that output matches expectations
print("Output shape:\n")
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape,"|| (batch_size, sequence_length, vocab_size)")

print("\nModel Summary\n")
print(model.summary())

model = make_model(len(vocab),256,1024,BATCH_SIZE,True)

# set up checkpoint saving
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( \
        filepath=checkpoint_prefix,save_weights_only=True)

model_name = "text_gen_trained"
loaded = False
if exist_model(model_name):
    loaded = True
    model, history = load_model(model_name)
else:
    history = fit_model(model,10,dataset,checkpoint_callback)
    save_model(model,history,model_name)
if not loaded:
    hist_plots(history,'accuracy','acc.png')
    hist_plots(history,'loss','lss.png')

# model needs to be rebuilt when batch size is changed
tf.train.latest_checkpoint(checkpoint_dir)
# recover the training weights so we dont have to do it again
model = make_model(len(vocab),256,1024,1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1,None]))

# Text generation portion
def generate_text(model, start_string):
# loops model output into text generation
    num_generate = 1000 # amount of characters to make
    # vectorize the starting string
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # output string initialization
    text_generated = []

    # temperature determines predictability of outputs.
    # Low temp is more predictable
    temperature = 0.8

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) # removes batch dimension

        # use a categorical distribution to predict
        predictions = predictions/temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # pass predicted as next input
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return(start_string + ''.join(text_generated))

# lets see some predictions
print("Prediction\n")
print(generate_text(model, start_string=u"ROMEO: "))

# make the model a bit better using a gradient tape

model = make_model(len(vocab),256,1024,64,"None")

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean( \
            tf.keras.losses.sparse_categorical_crossentropy( \
                target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))

    return loss

# new training
EPOCHS = 10
for epoch in range(EPOCHS):
    start = time.time()

    hidden = model.reset_states()

    for (batch_n, (inp,target)) in enumerate(dataset):
        lss = train_step(inp, target)

        if batch_n%100 == 0:
            template = 'Epoch {} | Batch {} | Loss {}'
            print(template.format(epoch+1, batch_n, lss))

    # save every 5 epochs
    if (epoch+1)%5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print('Epoch {} | Loss {:.4f}'.format(epoch+1, lss))
    print('Time for 1 epoch {} sec\n'.format(time.time()-start))

# save the final weights
model.save_weights(checkpoint_prefix.format(epoch=epoch))
