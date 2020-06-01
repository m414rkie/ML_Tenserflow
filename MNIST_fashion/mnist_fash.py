# !usr/bin/python3

# Second tutorial from tensorflow
# fashion MNIST dataset
# 5/30/20

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

# import dataset
# data is composed of 60k images of 28x28 in each training and 10k in test
fash_mnist = keras.datasets.fashion_mnist
# assign to arrays
(train_images, train_labels), (test_images, test_labels) = fash_mnist.load_data()

# classification on 10 categorie. Store names as indice value
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt',
                'Sneaker','Bag','Ankle Boot']

# get number of rows and cols
img = train_images[1]
p_rows, p_cols = img.shape

# normalize pixel values
train_images = train_images/255.0
test_images = test_images/255.0

# plot some of the images
ini_disp = plt.figure(1)
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
ini_disp.savefig('ini_disp.jpg')

# build the model
fash_model = keras.models.Sequential([
    # flatten
    keras.layers.Flatten(input_shape=(p_rows,p_cols)),
    # layer one
    keras.layers.Dense(128,activation='relu'),
    # dropout layer
    keras.layers.Dropout(0.2),
    # output layer
    keras.layers.Dense(10)
])

# set loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile the model
fash_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# fit the data
fash_model.fit(train_images,train_labels,epochs=10,validation_split=0.2)

print(fash_model.history.history.keys())

# save model
fash_model.save('./fashion_mnist')

# plot loss and accuracy by epoch
acc = fash_model.history.history['accuracy']
val_acc = fash_model.history.history['val_accuracy']
epochs = range(1,len(acc)+1)
acu = plt.figure(2)
plt.plot(epochs, acc, 'b', label="Training Accuracy")
plt.plot(epochs, val_acc, 'r', label="Validation Accuracy")
plt.title('Training and Validation Accuracy')
plt.legend()
acu.savefig("acc.jpg")

loss = fash_model.history.history['loss']
val_loss = fash_model.history.history['val_loss']
lss = plt.figure(3)
plt.plot(epochs, loss, 'b', label="Training Loss")
plt.plot(epochs, val_loss, 'r', label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()
lss.savefig("loss.jpg")

# test the data - the evaluate function removes the history attributes
# save them first
test_loss, test_acc = fash_model.evaluate(test_images, test_labels, verbose=2)
print("\nTest Accuracy:",test_acc,"\nTest Loss:",test_loss)

# some predictions
pre = plt.figure(4)
ims = np.random.randint(1,1000,16)
for i, im in enumerate(ims):
    print(i)
    prediction = fash_model.predict(test_images[im].reshape(1,p_rows,p_cols,1))
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()
    plt.imshow(test_images[im],cmap='gray')
    title = "Prediction: " + class_names[prediction.argmax()] + "\nActual:" \
    + class_names[test_labels[im]]
    plt.xlabel(title)
pre.savefig("pred.jpg")

# test recovery of model
saved_model = tf.keras.models.load_model("./fashion_mnist")
spre = plt.figure(5)
ims = np.random.randint(1,1000,16)
for i, im in enumerate(ims):
    prediction = saved_model.predict(test_images[im].reshape(1,p_rows,p_cols,1))
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()
    plt.imshow(test_images[im],cmap='gray')
    #plt.subplots_adjust(hspace = 2.1)
    title = "Prediction: " + class_names[prediction.argmax()] + "\nActual: " \
    + class_names[test_labels[im]]
    plt.xlabel(title)
spre.savefig("saved_pred.jpg")
