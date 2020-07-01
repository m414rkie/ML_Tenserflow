#! usr/bin/python3

# code from the tutorial on how to load image datasets into Tensorflow found at
# tensorflow.org

# Jon Parsons
# 6-28-2020

import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pathlib
import time
import math as mt

# set autotuner
AUTOTUNE = tf.data.experimental.AUTOTUNE

# get the dataset. Consists of a set of directories with a class of image per
# directory
data_dir = tf.keras.utils.get_file(origin= \
'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    fname='flower_photos',untar=True)

data_dir = pathlib.Path(data_dir)

# examine the data a bit
# how many images?
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Number of images: {}".format(image_count))
# what kind of flowers?
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != \
        "LICENSE.txt"])
print(CLASS_NAMES)
# look at some flowers
roses = list(data_dir.glob('roses/*'))

rose = plt.figure(1)
for i in range(9):
    image = mpimg.imread(roses[i])
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(image)
    plt.title("A Rose")
    plt.xticks([])
    plt.yticks([])
rose.savefig("roses.png", bbox_inches='tight')

# begin preprocessing the images using a keras preprocessor. This method is
# slow and lackes fine-grain control
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# includes normalization
BATCH_SIZE = 512
IMG_HT = 224 # known values
IMG_WD = 224
STEP_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir), \
    batch_size=BATCH_SIZE, shuffle=True, target_size=(IMG_HT,IMG_WD), \
    classes=list(CLASS_NAMES))

# examine some examples of the classes
# function for showing images
def show_batch(image_batch, label_batch,flg):
    batches = plt.figure(2)
    for n in range(25):
        plt.subplot(5,5,n+1)
        plt.tight_layout()
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
    if flg == 1:
        name = 'examples.png'
    else:
        name = 'ex_dataapi.png'
    batches.savefig(name, bbox_inches='tight')
    plt.clf()


image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch,1)

# now try using the dataset method
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

# these series of functions are pure tensorflow that convert a path to a tuple
# of (img, label)
def get_label(file_path):
    # converts to path components
    parts = tf.strings.split(file_path, os.path.sep)
    # we know the second to last component is the class directory
    return parts[-2] == CLASS_NAMES

def decode_img(img):
    # converts the string to 3d tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # convert to normalized floats
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_HT,IMG_WD])

def process_path(file_path):
    # get the label
    label = get_label(file_path)
    # load image
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

# map the pairs
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# see the stats of a piece of data
for image, label in labeled_ds.take(3):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

# this section will shuffle, batch, and make available batches for training
# using the tf.data api
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # keeps a small dataset in memory
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # keep on repeat
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)

    # allow for fetching to happen while the current batch is still training
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

train_ds = prepare_for_training(labeled_ds)

image_batch, label_batch = next(iter(train_ds))
# see some of the newly labeled things
show_batch(image_batch.numpy(), label_batch.numpy(),2)

# Timings
default_timeit_steps = 1000

def timeit(ds, steps=default_timeit_steps):
    start = time.time()
    it = iter(ds)
    for i in range(steps):
        batch = next(it)
        if i%10 == 0:
            print('.',end='')
    print()
    end = time.time()

    duration = end-start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))

# Yes, I am commenting for flow control.
#print("\n Preprocessing speed \n")
#timeit(train_data_gen)
#print("\n Data api \n")
#timeit(train_ds)

# compare with and without cache
#print("\n Uncached Data api \n")
#uncached_ds = prepare_for_training(labeled_ds, cache=False)
#timeit(uncached_ds)

# use a cache file instead of memory
#print("\n File cache \n")
#filecache_ds = prepare_for_training(labeled_ds, cache="./flowers.tfcache")
#timeit(filecache_ds)

# option 2 is fastest.
# making my own model here
# make the model, simple model without real experimentation
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(IMG_HT,IMG_WD,3)),
    keras.layers.Dense(256, activation='elu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='elu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(5)]) # five kinds of flowers

# the sparse categorical crossentropy did not give the correct value for
# the labels batches
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(image_batch, label_batch, epochs=15, validation_split=0.2)

print(model.history.history.keys())

# see some accuracy
acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
epochs = range(1,len(acc)+1)
acu = plt.figure(3)
plt.plot(epochs, acc, 'b', label="Training Accuracy")
plt.plot(epochs, val_acc, 'r', label = "Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
acu.savefig("accuracy.png", bbox_inches='tight')

# try to do a prediction
pre = plt.figure(4)
flwrs = np.random.randint(1,500,9)
for i, flwr in enumerate(flwrs):
    imgf = image_batch[flwr]
    prediction = model.predict(imgf)
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()
    plt.imshow(image_batch[flwr])
    title = "Prediction: " + CLASS_NAMES[prediction.argmax()] + "\nActual: " \
    + CLASS_NAMES[label_batch[im]]
    plt.xlabel(title)
pre.savefig("predictions.png", bbox_inches='tight')
