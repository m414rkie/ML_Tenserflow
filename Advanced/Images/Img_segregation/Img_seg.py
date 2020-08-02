#! usr/bin/python3

# Code taken from the tutorial on image segregation found on tensorflow.org.
# The tutorial handles how to create an outline of a pet via labeling pixels.
# Dataset is the Oxford-IIIT Pet Dataset.

# Jon Parsons
# 7-16-2020

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# dataset included in the tensorflow datasets
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

# function for normalization of images, also reduces label by 1 for starting at 0
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32)/255.0
    input_mask -= 1
    return input_image, input_mask

# decorators, fancy
@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128,128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128,128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128,128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128,128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

# set training parameters
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH//BATCH_SIZE

train = dataset['train'].map(load_image_train, \
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

# function for image display
def display(display_list,name):
    plt.figure(figsize=(15,15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig(name, bbox_inches='tight')
    plt.clf()

# show sample image
for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
    display([sample_image, sample_mask], 'sample.png')

# three possible output labels
OUTPUT_CHANNELS = 3

base_model = tf.keras.applications.MobileNetV2(input_shape=[128,128,3], \
    include_top=False)

# extract the activations
layer_names = [ \
    'block_1_expand_relu', \
    'block_3_expand_relu', \
    'block_6_expand_relu', \
    'block_13_expand_relu', \
    'block_16_project' \
    ]

layers = [base_model.get_layer(name).output for name in layer_names]
# freeze the base layers
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
down_stack.trainable = False

# encoder uses the pix2pix pretrained model
up_stack = [ \
    pix2pix.upsample(512,3), \
    pix2pix.upsample(256,3), \
    pix2pix.upsample(128,3), \
    pix2pix.upsample(64,3),
    ]

# function for making a U-net model
def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128,128,3])
    x = inputs

    # downsampling
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])
    # upsampling
    concat = tf.keras.layers.Concatenate()
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    # form the last layer of the model
    last = tf.keras.layers.Conv2DTranspose( \
        output_channels, 3, strides=2, padding='same')

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

# compile and train
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', \
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
    metrics=['accuracy'])

# see the shape of the model
#tf.keras.utils.plot_model(model, to_file='model_shape.png', show_shapes=True)
print(model.summary())
# see before the training
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[ ..., tf.newaxis]
    return pred_mask[0]

def show_predictions(i, dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)], \
            'prediction{}.png'.format(i))
            i += 1
        else:
            display([sample_image, sample_mask, \
                create_mask(model.predict(sample_image[tf.newaxis, ...]))], \
                'pretrained.png')

show_predictions(1)

# call back for seeing how the model behaves during training
class DisplayCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.i = 1
    def on_epoch_end(self, epoch, logs=None):
        show_predictions(self.i)
        self.i += 1

# training params
EPOCHS = 20
VAL_SUBSPLITS = 5
VAL_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS, \
    steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VAL_STEPS, \
    validation_data=test_dataset, callbacks=[DisplayCallBack()])

# plot the loss
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0,1])
plt.legend()
plt.savefig('loss.png', bbox_inches='tight')
plt.clf()

# make a couple predictions
show_predictions(10, test_dataset, 3)
