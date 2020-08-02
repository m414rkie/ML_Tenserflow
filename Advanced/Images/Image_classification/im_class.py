#! usr/bin/python3

# Tutorial from tensorflow.org. Tutorial is on classification of images and
# prevention of overfitting. Data set is images of cats and dogs from Kaggle

# Jon Parsons
# 7-10-2020

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

# get the data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

# structure of the data is known so split into training and testing data
train_dir = os.path.join(PATH, 'train')
val_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
val_cats_dir = os.path.join(val_dir, 'cats')
val_dogs_dir = os.path.join(val_dir, 'dogs')

# explore some of the data
num_cats_trn = len(os.listdir(train_cats_dir))
num_dogs_trn = len(os.listdir(train_dogs_dir))
num_cats_val = len(os.listdir(val_cats_dir))
num_dogs_val = len(os.listdir(val_dogs_dir))

tot_trn = num_cats_trn + num_dogs_trn
tot_val = num_cats_val + num_dogs_val

print("Training Images")
print("Cats: {}".format(num_cats_trn))
print("Dogs: {}".format(num_dogs_trn))
print("Validation Images")
print("Cats: {}".format(num_cats_val))
print("Dogs: {}".format(num_dogs_val))
print("Total Training {} | Validation {}".format(tot_trn,tot_val))

# set up some training variables
batch_size = 128
epochs = 15
IMG_HT = 150 # known already
IMG_WD = 150

# prepare the images for training. Uses the keras image generator
train_image_generator = ImageDataGenerator(rescale=1./255)
val_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, \
        directory=train_dir, shuffle=True, target_size=(IMG_HT,IMG_WD), \
        class_mode='binary')

val_data_gen = val_image_generator.flow_from_directory(batch_size=batch_size, \
        directory=val_dir, target_size=(IMG_HT,IMG_WD), class_mode='binary')

# plot some of the images
sample_training_images, _ = next(train_data_gen)

# function to plot some images
def plotImages(images_arr,title):
    fig, axes = plt.subplots(1,5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    title = title + ".png"
    plt.savefig(title, bbox_inches='tight')
    plt.clf()

plotImages(sample_training_images[:5],"images")

# make the model
# padding option tries to evenly pad to the left and right (odd column right)
model = Sequential([ \
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HT,IMG_WD,3)), \
    MaxPooling2D(), \
    Conv2D(32, 3, padding='same', activation='relu'), \
    MaxPooling2D(), \
    Conv2D(64, 3, padding='same', activation='relu'), \
    MaxPooling2D(), \
    Flatten(), \
    Dense(512, activation='relu'), \
    Dense(1) \
    ])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), \
        metrics=['accuracy'])

# see the model
print("Model:")
print(model.summary())

# train the model with the fit generator from the imagedatagenerator class
history = model.fit_generator(train_data_gen, steps_per_epoch=tot_trn // batch_size, \
    epochs=epochs, validation_data=val_data_gen, validation_steps=tot_val // batch_size)

# visualization of the process
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# make the plot of training and validation
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title("Accuracy")
plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title("Loss")
plt.savefig("training_metrics.png", bbox_inches='tight')
plt.clf()

# oh my, the high training accuracy compared to validation indicates overtraining
# how to fix?
# data augmentation

# we can apply horizontal flip
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size, \
        directory=train_dir, shuffle=True, target_size=(IMG_HT,IMG_WD))

aug_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(aug_images, 'horizontal_flip')

# we can also rotate the image, this one rotates by 45deg randomly
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size, \
        directory=train_dir, shuffle=True, target_size=(IMG_HT,IMG_WD))

aug_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(aug_images, 'rotated')

# also a zoom
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size, \
        directory=train_dir, shuffle=True, target_size=(IMG_HT,IMG_WD))

aug_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(aug_images, 'zoom')

# all of them together for the training
image_gen_train = ImageDataGenerator(rescale=1./255, rotation_range=45, \
        width_shift_range=0.15, height_shift_range=0.15, horizontal_flip=True, \
        zoom_range=0.5)

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size, \
        directory=train_dir, shuffle=True, target_size=(IMG_HT,IMG_WD), \
        class_mode='binary')

# see what it all looks like together
aug_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(aug_images, 'alltogether')

# rescale the validation images
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size, \
        directory=val_dir, target_size=(IMG_HT,IMG_WD), class_mode='binary')

# apply some dropout in a new model
model_new = Sequential([ \
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HT,IMG_WD,3)), \
    MaxPooling2D(), \
    Dropout(0.2), \
    Conv2D(32, 3, padding='same', activation='relu'), \
    MaxPooling2D(), \
    Conv2D(64, 3, padding='same', activation='relu'), \
    MaxPooling2D(), \
    Dropout(0.2), \
    Flatten(), \
    Dense(512, activation='relu'), \
    Dense(1)
    ])

model_new.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), \
    metrics=['accuracy'])

print("New model Summary")
print(model_new.summary())

# train the model with the data augmentation
history = model_new.fit_generator(train_data_gen, \
        steps_per_epoch=tot_trn//batch_size, \
        epochs=epochs, validation_data=val_data_gen, \
        validation_steps=tot_val//batch_size)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# visualize the model
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title("Accuracy")
plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title("Loss")
plt.savefig("training_aug.png", bbox_inches='tight')
plt.clf()
