#! usr/bin/python3

# Tutorial from tensorflow.org. How to tune the hyper parameters in tensorflow.
# Uses the MNIST fashion dataset

# Jon Parsons
# 6-19-2020

import tensorflow as tf
from tensorflow import keras
import kerastuner as kt

# get data
(img_train, label_train), (img_test, label_test) = \
    keras.datasets.fashion_mnist.load_data()

# normalize
img_train = img_train.astype('float32')/255
img_test = img_test.astype('float32')/255

# define function that builds the model using hyperparameters
def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    # This line will restrict the number of neurons in the first layer
    # between 32 and 512 in steps of 32
    hp_units = hp.Int('units',min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))
    # This line tunes the learning rate between a set of three
    hp_lr = hp.Choice('learning_rate', values=[1e-2,1e-3,1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_lr), \
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model

# prepare the tuner of the hyperparameters. There are four available
tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10, \
                factor=3, directory='my_dir', project_name='intro_to_kt')

# find the optimal number of parameters
tuner.search(img_train, label_train, epochs=10, validation_data= \
        (img_test,label_test))

# get those numbers
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Found optimal neurons: ", best_hps.get('units'))
print("Found optimal learning rate", best_hps.get('learning_rate'))

# retrain the model and fit
model = tuner.hypermodel.build(best_hps)
model.fit(img_train, label_train, epochs=10, validation_data=(img_test,label_test))

print(model.summary())
