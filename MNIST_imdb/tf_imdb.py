#! usr/bin/python3

# Third tutorial from tensorflow.org
# Classifies short movie reviews as positive or negative using the
# IMDB data set from MNIST

# Jon Parsons
# 6-5-2020

import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
# library to enable transfer learning
import nltk as nltk
import tensorflow_hub as hub
import tensorflow_datasets as tfds

################################################################################

# import data, split dataset S.T. 60% is training, 40% is validation
train_data, valid_data, test_data = tfds.load(name="imdb_reviews", \
                                    split=('train[:60%]','train[:60%]','test'), \
                                    as_supervised=True)

# display some of the data
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch,"\n", train_labels_batch)

# model will take advantage of a pre-trained embedding model that represents
# sentances as integer vectors
embeder = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
embed = hub.load(embeder)

hub_layer = hub.KerasLayer(embeder, input_shape = [], dtype=tf.string, \
                           trainable=True)

# build model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
print(model.summary())
# compile model
model.compile(optimizer='adam', \
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), \
              metrics=['accuracy'])


# training model
history = model.fit(train_data.shuffle(10000).batch(512),epochs=20, \
                    validation_data=valid_data.batch(512),
                    verbose=1)

# images of accuracy and loss
acu = plt.figure(1)
acc = model.history.history['accuracy']
epochs = range(1, len(acc)+1)
val_acc = model.history.history['val_accuracy']
plt.plot(epochs,acc, 'b', label="Training Accuracy")
plt.plot(epochs,val_acc, 'r', label="Validation Accuracy")
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
acu.savefig("accu.jpg")

lss = plt.figure(2)
loss = model.history.history['loss']
val_loss = model.history.history["val_loss"]
plt.plot(epochs,loss, 'b', label="Loss")
plt.plot(epochs,val_loss, 'r', label="Validation Loss")
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
lss.savefig("loss.jpg")

# see how it does with numbers
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names,results):
    print("%s: %.3f" % (name, value))
