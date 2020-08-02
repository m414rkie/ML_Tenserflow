#! usr/bin/python3

# Tutorial on word embeddings from tensorflow.org. Includes visualizations.
# Will train a sentiment classifier on the IMDB database

# Jon Parsons
# 7-18-2020

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import io

# inspect the embedding layer args: (input_dim, output_dim)
embedding_layer = layers.Embedding(1000,5)
# lower dimensionality reduces the complexity of captured relationships

# passing an integer to the embedding layer returns the vector from the table
result = embedding_layer(tf.constant([1,2,3]))
print('Three vectors from the embedder')
print(result.numpy())

# for sequential problems, the input is (samples, sequence_length)
# input shape of (2,3) returns (2,3,N)
result = embedding_layer(tf.constant([[1,2,3],[3,4,5]]))
print("Shape of returned vector with input shape of (2,3)")
print(result.shape)

# Get the IMDB dataset
(train_data, test_data), info = tfds.load('imdb_reviews/subwords8k', \
        split = (tfds.Split.TRAIN, tfds.Split.TEST), with_info=True, \
        as_supervised=True)

# inspect existing encodings for the dataset. Includes partial words and '_' as
# spaces
encoder = info.features['text'].encoder
print("Portion of Encoder Vocabulary")
print(encoder.subwords[:20])

# pad the data to a standard length, shuffle
train_batches = train_data.shuffle(1000).padded_batch(10)
test_batches = test_data.shuffle(1000).padded_batch(10)

# see a padded batch
train_batch, train_labels = next(iter(train_batches))

print('Padded example')
print(train_batch.numpy())

# make a simple model
# NOTE: masking is not used, padding length may affect results
embedding_dim = 16
model = keras.Sequential([ \
    layers.Embedding(encoder.vocab_size, embedding_dim), \
    layers.GlobalAveragePooling1D(), \
    layers.Dense(16, activation='relu'), \
    layers.Dense(1)
    ])

print(model.summary())

# compile and train
model.compile(optimizer='adam', \
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), \
    metrics=['accuracy'])

history = model.fit(train_batches, epochs=10, validation_data=test_batches, \
    validation_steps=20)

# some plots
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("lss.png", bbox_inches='tight')
plt.clf()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("acc.png", bbox_inches='tight')
plt.clf()

# retrieve learned embedding
e = model.layers[0]
weights = e.get_weights()[0]

print("Shape of weights vector")
print(weights.shape)

# write weights to a TSV
encoder = info.features['text'].encoder

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(encoder.subwords):
    vec = weights[num+1] # 0 is padding
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

# embedding visualizer on tensorflow.org
