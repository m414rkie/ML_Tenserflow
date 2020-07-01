#! usr/bin/python3

# Program uses the preprocessed version of the IMDB dataset from MNIST to
# determine review sentiment. Follows a tutorial from tensorflow.org.
# Includes the text loading tutorial.

# Jon Parsons
# 6-11-2020

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences

################################################################################

# get dataset - pre-processed version
(train_data, test_data), info = tfds.load('imdb_reviews/subwords8k', \
                                split=(tfds.Split.TRAIN, tfds.Split.TEST), \
                                as_supervised=True, with_info=True)

# set encoder
encoder = info.features['text'].encoder
# how many words does it know?
print("Encoder knows {} words".format(encoder.vocab_size))

# encode and recover char string for practice
sample = "I tried this before and it was hard."
sample_encoded = encoder.encode(sample)
print("String to change \n",sample)
print("Encoded version: {}".format(sample_encoded))
sample_decoded = encoder.decode(sample_encoded)
print("Recovered version: {}".format(sample_decoded))
# See the breakdown of encoding
for vv in sample_encoded:
    print('{} was {}'.format(vv,encoder.decode([vv])))

## Skipping data exploration. See tutorial program tf_imdb.py for that ##

# batch the data and buffer the data. Ensures each datum is of the same length
BUFFER = 1000
train_batches = (train_data.shuffle(BUFFER).padded_batch(32))
test_batches = (test_data.padded_batch(32))
## Note that the padding is used as part of the input. Therefore review length
## may impact learning

# build the model
model = keras.Sequential([
        keras.layers.Embedding(encoder.vocab_size, 16),  # Takes index and
                # searchs for embedding vector.
        keras.layers.GlobalAveragePooling1D(),  # Averages over input vector,
                # Simplest method of handling multi-variate inputs to the next
                # layer
        keras.layers.Dense(1)  # Single output for bad/good review classes
])

# what does the model look like?
print(model.summary())

# compile the model
model.compile(optimizer='adam', \
            loss=tf.losses.BinaryCrossentropy(from_logits=True), \
            metrics=['accuracy']
)

# train the model
history = model.fit(train_batches, epochs=10, \
                    validation_data=test_batches, validation_steps=30)

# how does it do?
loss, accuracy = model.evaluate(test_batches)
print("Overall Loss: ", loss)
print("Overall Accuracy: ", accuracy)

# Make some graphs of training history
history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc)+1)

acu = plt.figure(1)
plt.plot(epochs, acc, 'b',label="Accuracy")
plt.plot(epochs, val_acc, 'r', label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
acu.savefig('accuracy.jpg')

lss = plt.figure(2)
plt.plot(epochs, loss, 'b',label="Loss")
plt.plot(epochs, val_loss, 'r', label="Validation Loss")
plt.title("Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
lss.savefig('loss.jpg')

# Lets see how this does with new strings.
good = "I love this great, good, movie."
bad = "I hate this terrible, aweful, movie."

good_encoded = encoder.encode(good)
bad_encoded = encoder.encode(bad)

# make a list of lists so that we can pad
gd_enc = []
gd_enc.append(good_encoded)
bd_enc = []
bd_enc.append(bad_encoded)

print("Encoded good")
print(gd_enc)
print("Encoded bad")
print(bd_enc)

good_encoded = pad_sequences(gd_enc, maxlen=32)
bad_encoded = pad_sequences(bd_enc, maxlen=32)

good_pred = model.predict_classes(good_encoded)
bad_pred = model.predict_classes(bad_encoded)

print("Model predicted good to be: ", good_pred)
print("Model predicted bad to be: ", bad_pred)
