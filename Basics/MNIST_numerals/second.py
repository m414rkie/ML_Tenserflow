#!usr/bin/python3
# First introductory tutorial from tensorflow.org

# Jon Parsons
# 5/29/20

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

# keras is a high-level API for tensorflow.
# mnist is a training data set consisting of 70k grayscale images of handwritten
# digits
mnist = tf.keras.datasets.mnist

# loads data to tuples, presplit into training and testing sets
# x corresponds to image, y to which digit (named)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# data sets are pixel data with value [0,255] (grayscale). This normalizes the data
x_train, x_test = x_train/255.0, x_test/255.0

# See examples of the images
fig = plt.figure(1)
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(x_train[i],cmap='gray',interpolation='none')
    plt.title("Label: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])
fig.savefig("data.png",bbox_inches='tight')

# number of pixels per row, column. Assumes consistent size
img = x_train[1]
p_rows, p_cols = img.shape

# sequential model appropriate for plain stacks of layers, each layer 1 input
# and one output. not appropriate for MIMO, layer sharing, non-linear
# sequential defines a sequence of layers
# Flatten reduces matrix to vector
# dense adds new layer of neurons
# this set gives a model with flattened inputs, a first hidden layer of
# 128 neurons with relu activation, a dropout to prevent overfitting,
# a second layer of 10 neurons for output, 10 because there are 10 possible
# outputs (is this a standard thing?) with standard activation
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(p_rows,p_cols)), \
    # set output to vectors of [:128], relu is rectified linear unit
    # a piecewise linear function that generates output with a positive input,
    # zero else. Often the default due to speed and ease of training
    tf.keras.layers.Dense(128, activation = 'relu'), \
    # dropout helps prevent overfit by randomly setting neurons to zero
    # rate = 0.2. keeps sum unchanged by scaling un-zeroed inputs by 1/(1-rate)
    tf.keras.layers.Dropout(0.2), \
    # set outputs to [:10], default activator of f(x) = x
    tf.keras.layers.Dense(10)
])

# output of model
model.summary()

predictions = model(x_train[:1]).numpy()

# softmax takes array and returns array of same size zero except where largest
# value was where 1 is now
tf.nn.softmax(predictions).numpy()

# comput quantity that the model should minimize
# this one finds loss between the labels and predictions
# expects labels as integers
# categorical crossentropy is useful for when predicting classes of objects
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# compare labels (y_train) to predictions set
loss_fn(y_train[:1], predictions).numpy()

# configure model for training
# implements stochastic gradient descent method (adam)
model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

# run the model and fit
# epochs is number of times to run through data set
model.fit(x_train,y_train, epochs=5, validation_split=0.1)

# save model
model.save('./models/mnist')

# show training and validation accuracies
# loss refers to a regression or classification error, a larger number is bad
# a typical example would be the MSE or quadratic loss
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
epochs = range(1, len(loss) + 1)
lss = plt.figure(2)
plt.plot(epochs,loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
lss.savefig("loss.jpg")
# accuracy plot
acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
acu = plt.figure(3)
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
acu.savefig("acc.jpg")

# validate model with test data
# the next one would be predict()
model.evaluate(x_test, y_test, verbose=2)
# wrap model with softmax and makes a model that returns a probability
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
probability_model(x_test[:5])

# some predictions
pre = plt.figure(4)
ims = np.random.randint(1,500,9)
for i, im in enumerate(ims):
    prediction = model.predict(x_test[im].reshape(1,p_rows,p_cols,1))
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(x_test[im],cmap='gray',interpolation='none')
    plt.title("Prediction: {}\n Actual: {}".format(prediction.argmax(),y_test[im]))

pre.savefig("pred.jpg")

# recover and check model
saved_model = tf.keras.models.load_model("./models/mnist")
pre = plt.figure(5)
ims = np.random.randint(1,500,9)
for i, im in enumerate(ims):
    prediction = saved_model.predict(x_test[im].reshape(1,p_rows,p_cols,1))
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(x_test[im],cmap='gray',interpolation='none')
    plt.title("Prediction: {}\n Actual: {}".format(prediction.argmax(),y_test[im]))

pre.savefig("saved_pred.jpg")
