#! usr/bin/python3

# Program taken from the tutorial on transfer learning at tensorflow.org.
# Will utilize TensorFlow Hub to train an image classifier.

# Jon Parsons
# 7-12-2020

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image
from tensorflow.keras import layers

# get pretrained classifier
classifier_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2'
# we will be setting all images to a specific size
IM_SHAPE = (224, 224)

classifier = tf.keras.Sequential([hub.KerasLayer(classifier_url, \
        input_shape=IM_SHAPE+(3,))])

# get a single image to test with
grace_hopper = tf.keras.utils.get_file('image.jpg', \
    'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IM_SHAPE)
plt.imshow(grace_hopper)
plt.savefig('g_hopper.png', bbox_inches='tight')
plt.clf()

# normalize the image
grace_hopper = np.array(grace_hopper)/255.0
# include a batch dimension and see what the classifier thinks
print("Shape before batch dim: ", grace_hopper.shape)
result = classifier.predict(grace_hopper[np.newaxis, ...])
print("Shape of output array: ", result.shape)
pred_class = np.argmax(result[0], axis=-1)
print("Class prediction: {}".format(pred_class))

# get the class labels
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', \
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

imagenet_labels = np.array(open(labels_path).read().splitlines())

# see image prediction with the label
plt.imshow(grace_hopper)
plt.axis('off')
pred_name = imagenet_labels[pred_class]
_ = plt.title("Prediction: " + pred_name.title())
plt.savefig('w_pred.png', bbox_inches='tight')
plt.clf()

# get the flowers dataset, which the classifier has not been trained for
data_root = tf.keras.utils.get_file('flower_photos', \
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', \
    untar=True)

# set the image generator to handle normalization
# results in iterator that returns data, label pairs
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.0)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IM_SHAPE)

for image_batch, label_batch in image_data:
    print("Image Batch Shape: ", image_batch.shape)
    print("Label Batch Shape: ", label_batch.shape)
    break

result_batch = classifier.predict(image_batch)

# predict some classes
pred_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
# see the predicted names
print('Class Names:\n', pred_class_names)

# pair predictions with the images
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    plt.title(pred_class_names[n])
    plt.axis='off'
_ = plt.suptitle("ImageNet Predictions")
plt.savefig("predictions.png", bbox_inches='tight')

# get the headless model so we can retrain
feature_extractor_url = \
    'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2'

feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224,224,3))
feature_batch = feature_extractor_layer(image_batch)
# see the shape
print("Feature Shape:", feature_batch.shape)

# ensure that onle the classifier layer can be trained
feature_extractor_layer.trainable = False

# attach a new head
model = tf.keras.Sequential([feature_extractor_layer, \
        layers.Dense(image_data.num_classes)])

print(model.summary())

# see how predictions are shaped per batch
pred = model(image_batch)
print("New shape: ", pred.shape)

# train the new model
model.compile(optimizer=tf.keras.optimizers.Adam(), \
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), \
        metrics=['acc'])

# this class logs the accuracy and loss for each epoch
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)
batch_stats_callback = CollectBatchStats()

history = model.fit_generator(image_data, epochs=2, steps_per_epoch=steps_per_epoch, \
        callbacks = [batch_stats_callback])

# see some of the training metrics
plt.figure(4)
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_losses)
plt.savefig('training_loss.png', bbox_inches='tight')
plt.clf()

plt.figure(5)
plt.ylabel("Accuracy")
plt.xlabel('Training Steps')
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)
plt.savefig('training_acc.png', bbox_inches='tight')
plt.clf()

# see some of the newly trained predictions
class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
print("class Names:", class_names)

pred_batch = model.predict(image_batch)
pred_id = np.argmax(pred_batch, axis=-1)
pred_label_batch = class_names[pred_id]

label_id = np.argmax(label_batch, axis=-1)

# see the predictions
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    color = 'green' if pred_id[n] == label_id[n] else 'red'
    plt.title(pred_label_batch[n].title(), color=color)
_ = plt.suptitle("Predictions")
plt.savefig("retrain_preds.png", bbox_inches='tight')

# export the model
import time
t = time.time()

export_path = "/tmp/saved_models/{}".format(int(t))
model.save(export_path, save_format='tf')

print("Saved at: ", export_path)

# confirm save and reload
reload = tf.keras.models.load_model(export_path)

result_batch = model.predict(image_batch)
reload_batch = reload.predict(image_batch)

diff = abs(reload_batch - result_batch).max()
print("Difference between saved and not saved")
print(diff)
