#! usr/bin/python3

# Tutorial from tensorflow.org. Uses an RNN to train sentiment analysis on the
# IMDB dataset

# Jon Parsons
# 7-26-2020

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

# quick function for graphing
def plot_graphs(history, metric, name):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.savefig(name, bbox_inches='tight')
    plt.clf()

# get dataset and split
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, \
                                                            as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

# get the encoder
encoder = info.features['text'].encoder
print("Encoder vocabulary size: {}".format(encoder.vocab_size))

# play with encoder a bit
sample_string = 'Hello TensorFlow'

encoded_string = encoder.encode(sample_string)
print("Encoded String: {}".format(encoded_string))
print("Original String: {}".format(sample_string))

print("Encoding translation Example")
for index in encoded_string:
    print('{} --> {}'.format(index, encoder.decode([index])))

# training parameters
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)

test_dataset = test_dataset.padded_batch(BATCH_SIZE)

# model making
model = tf.keras.Sequential([ \
    tf.keras.layers.Embedding(encoder.vocab_size, 64), \
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)), \
    tf.keras.layers.Dense(64,activation='relu'), \
    tf.keras.layers.Dense(1)
    ])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), \
    optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy']
)

# training
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, \
                                                            validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

# quick results
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# test some effects of padding
def pad_to_size(vec, size):
    zeros = [0]*(size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sample_pred_text, pad):
    encoded_sample_pred_text = encoder.encode(sample_pred_text)

    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return (predictions)

sample_text = ('The movie was so cool. The animations were awesome and '
                'were out of this world. Would recommend this movie.')

predictions = sample_predict(sample_text, pad=False)
print("Sample prediction with no padding. Should return positive")
print(predictions)
predictions = sample_predict(sample_text, pad=True)
print("Sample prediction with padding. Should return positive")

# get some plots of the training
plot_graphs(history, 'accuracy', 'acc.png')
plot_graphs(history, 'loss', 'loss.png')

# modify the model to be better. Includes another biderectional layer for
# further training of longer sequences
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64), \
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)), \
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)), \
    tf.keras.layers.Dense(64, activation='relu'), \
    tf.keras.layers.Dropout(0.5), \
    tf.keras.layers.Dense(1)
    ])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), \
                optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, \
                                                            validation_steps=30)


test_loss, test_acc = model.evaluate(test_dataset)

print("Modified model results")
print("Test Loss: {}".format(test_loss))
print("Test Accuracy: {}".format(test_acc))

sample_text = ('What a terrible movie. Every second was torture. Do not see.')

predictions = sample_predict(sample_text, pad=False)
print("Sample prediction with no padding. Shoud return negative.")
print(predictions)

predictions = sample_predict(sample_text, pad=True)
print("Sample prediction with padding. Shoud return negative.")
print(predictions)

plot_graphs(history, 'accuracy', 'new_acc.png')
plot_graphs(history, 'loss', 'new_loss.png')
