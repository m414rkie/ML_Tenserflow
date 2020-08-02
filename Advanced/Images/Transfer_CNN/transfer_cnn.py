#! usr/bin/python3

# Code from tutorial on transfer learning using a CNN from tensorflow.org.
# Dataset is cats and dogs.

# Jon Parsons
# 7-14-2020

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# get the data and split into training, validation, and test sets
(raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs', \
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], \
        with_info=True, as_supervised=True,)

# see shape of each set
print("Data shapes:\n")
print(raw_train)
print(raw_validation)
print(raw_test)

# show first couple images
get_label_name = metadata.features['label'].int2str

i = 0
for image, label in raw_train.take(2):
    i += 1
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.savefig('data{}.png'.format(i), bbox_inches='tight')
    plt.clf()

# we are going to resize all images to 160X160
IM_SZ = 160
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1 # this rescale is from -1 to 1 this time
    image = tf.image.resize(image, (IM_SZ, IM_SZ))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# shuffle and batch
BATCH_SIZE = 32
SHUFFLE_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# see the resulting shapes
for image_batch, label_batch in train_batches.take(1):
    pass

print("Shape of batched data")
print(image_batch.shape)

# The pretrained model is the MobileNet V2 model. We will use the last layer
# before the flatten layer, which is also called a bottleneck layer. This layer
# retains more generality and is commonly used as the last layer when adapting
# pretrained models

# set input shape
IM_SHAPE = (IM_SZ, IM_SZ, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IM_SHAPE, \
    include_top=False, weights='imagenet')

# see the transformation that an image goes through
feature_batch = base_model(image_batch)
print("Shape of data in the model:\n")
print(feature_batch.shape)

# freeze the base
base_model.trainable = False
# see the model
print(base_model.summary())

# add a classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
# see the shape
print("Pooling layer shape")
print(feature_batch_average.shape)

# add the top layer with the pooling layer
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)

model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])

# set the learning rate and compile the model
base_lr = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_lr), \
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), \
        metrics=['accuracy'])

# see the final shape of the model
print("Final model shape")
print(model.summary())

# training variables
init_epochs = 10
val_steps = 20
# initial values
loss0, acc0 = model.evaluate(validation_batches, steps=val_steps)

print("Initial Loss: {}".format(loss0))
print("Initial Accuracy: {}".format(acc0))

# training
history = model.fit(train_batches, epochs=init_epochs, \
        validation_data=validation_batches)

# make some plots of what has been happening
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.subplot(2,1,2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,0.5])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.savefig('acc_loss.png', bbox_inches='tight')
plt.clf()

# Note: batch normalization and dropout are turned off during validation and
# thus validation will have 'better' metrics

# Fine Tuning: Initial layers learn generic features, and higher layers learn
# set-specific things. Retraining a model should generally be restricted to
# a small number of top level layers.

# unfreeze the top layer.
base_model.trainable = True
# see how many layers exist
print("Base model has {} layers".format(len(base_model.layers)))

# we want to tune starting at the 100th layer
tune_at = 100
for layer in base_model.layers[:tune_at]:
    layer.trainable = False

# compile
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), \
        optimizer=tf.keras.optimizers.RMSprop(lr=base_lr/10), \
        metrics=['accuracy'])

print(model.summary())

# training
tune_epochs = 10
total_epochs = init_epochs + tune_epochs

history_fine = model.fit(train_batches, epochs=total_epochs, \
        initial_epoch=history.epoch[-1], \
        validation_data=validation_batches)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

# new plots! Now with fine tuning
plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8,1])
plt.plot([init_epochs-1,init_epochs-1],plt.ylim(),label='Start of Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(2,1,2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0,0.5])
plt.plot([init_epochs-1,init_epochs-1],plt.ylim(),label="Start of Fine Tuning")
plt.legend(loc='upper right')
plt.title("Traning and Validation Loss")
plt.xlabel('Epochs')
plt.savefig("fine_tuning_params.png", bbox_inches='tight')
