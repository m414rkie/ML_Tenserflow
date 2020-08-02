#! usr/bin/python3

# Tutorial on image classification using a CNN. Tutorial from tensorflow.org.
# dataset is the CIFAR10 dataset. 60000 images in 10 classifications, including
# vehicles and animals. Images are 32X32 color.

# CNN's are good for images since they take spatial relationships into account

# Jon Parsons
# 7-8-2020

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# get data set
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalize images
train_images, test_images = train_images/255.0 , test_images/255.0

print(train_images[1].shape)
print(test_images[1].shape)

# set class names
class_names = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog', \
    'Horse','Ship','Truck']

# plot some of the data for inspection
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # labels are arrays, hence index
    plt.xlabel(class_names[train_labels[i][0]])
plt.savefig("in_data.png", bbox_inches='tight')
plt.clf()

# start the model
# convolutional part
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2))) # downsamples by taking max of inputs for
        # each dimension
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2))) # downsamples by taking max of inputs for
        # each dimension
model.add(layers.Conv2D(64, (3,3), activation='relu'))
# dense part
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
# Complete architecture output
print("Model Summary\n")
print(model.summary())
print("\n\n")

# compile and train
model.compile(optimizer='adam', \
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
    metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, \
    validation_data=(test_images, test_labels))

# see some of the model training results
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png', bbox_inches='tight')
plt.clf()

# see how some predictions do
ims = [1,4,40,1600] # just some numbers to see what happens
for i in range(4):          # reshape of the inputs because keras expects a batch
    prediction = model.predict(test_images[i].reshape(1,32,32,3))
    pred_name = class_names[prediction.argmax()]
    plt.subplot(2,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Prediction: '{}'".format(pred_name))

plt.savefig("prediction.png", bbox_inches='tight')
