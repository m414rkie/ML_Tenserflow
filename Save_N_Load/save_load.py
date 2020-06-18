#! usr/bin/python3

# Program to practice saving and loading of models.
# Taken from the tutorial at Tensorflow.org

# Jon Parsons
# 6-17-2020

# uses h5py for saving the model in HDF5 format

import os
import tensorflow as tf
from tensorflow import keras

# just using the MNIST sample dataset of handwritten digits, first 1000 only
(train_images, train_labels), (test_images, test_labels) = \
                tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1,28*28)/255 # linearize and normalize
test_images = test_images[:1000].reshape(-1,28*28)/255

# make a simple model
def create_model():
    model = tf.keras.models.Sequential([ \
        keras.layers.Dense(512, activation='relu', input_shape=(784,)), \
        keras.layers.Dropout(0.2), \
        keras.layers.Dense(10)])

    model.compile(optimizer='adam', \
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), \
        metrics=['accuracy'])

    return model

model = create_model()
print(model.summary())

# set the checkpoint save path
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# set up call back feature
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, \
                                save_weights_only=True, verbose=1)

# fit the model
hist = model.fit(train_images, train_labels, epochs=10, \
        validation_data=(test_images, test_labels), callbacks=[cp_callback])

loss, acc = model.evaluate(test_images, test_labels, verbose=2)

print("Fitted Accuracy: {:5.2f}%".format(100*acc))

# create a second untrained model and use the weights from the first and see
# what happens
model2 = create_model()

loss, acc = model2.evaluate(test_images, test_labels, verbose=2)
print("Untrained Accuracy: {:5.2f}%".format(100*acc))

# load the weights into it
model2.load_weights(checkpoint_path)

loss, acc = model2.evaluate(test_images, test_labels, verbose=2)
print("Recovered Accuracy: {:5.2f}%".format(100*acc))

# adjust some parameters including checkpoint names
# use epoch in filename
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# save the weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint( \
    filepath=checkpoint_path, verbose=1, \
    save_weights_only=True, period=5)

# Another model
model3 = create_model()
model3.save_weights(checkpoint_path.format(epoch=0))
# train new model
hist2 = model3.fit(train_images, train_labels, epochs=50, \
        callbacks=[cp_callback], validation_data=(test_images,test_labels), \
        verbose=0)

# save last set of weights
latest = tf.train.latest_checkpoint(checkpoint_dir)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)

print("Second Trained Model accuracy: {:5.2f}%".format(100*acc))

# Another new model
model4 = create_model()

model4.load_weights(latest)
loss, acc = model4.evaluate(test_images, test_labels, verbose=2)
print("Second recovered weights accuracy {:5.2f}%".format(100*acc))

# How to manually save weights
model4.save_weights('./checkpoints/my_checkpoint')

# yet another new model
model5 = create_model()
# recover weights
model5.load_weights('./checkpoints/my_checkpoint')
# evaluate
loss, acc = model5.evaluate(test_images, test_labels, verbose=2)
print("Third recovered weights accuracy {:5.2f}%".format(100*acc))

# saving the entire model, as SavedModel format
model6 = create_model()
model6.fit(train_images, train_labels, epochs=5)

if not os.path.exists('saved_model/my_model'):
    os.makedirs('saved_model/my_model')

model.save('saved_model/my_model')
# recover saved model as new model
new_model = tf.keras.models.load_model('saved_model/my_model')
print(new_model.summary())
# see how it does
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Restored model accuracy: {:5.2f}%".format(100*acc))

# Now saving with HDF5 format
model7 = create_model()
model7.fit(train_images, train_labels, epochs=5)

model7.save('my_model.h5')

# recover and test
model8 = tf.keras.models.load_model('my_model.h5')

print(model8.summary())
loss, acc = model8.evaluate(test_images, test_labels, verbose=2)
print("HDF5 recovered accuray: {:5.2f}%".format(100*acc))
