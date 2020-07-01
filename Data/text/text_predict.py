#! usr/bin/python3

# program from the tutorial on loading data from text files at Tensorflow.org

# Jon Parsons
# 6-29-2020

import tensorflow as tf
import tensorflow_datasets as tfds
import os

# get data files. Consists of three text files of three different translations
# of the Illiad
DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt','derby.txt','butler.txt']

for name in FILE_NAMES:
    text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL+name)

parent_dir = os.path.dirname(text_dir)
print(parent_dir)

# load the text into datasets

# individually label sets
def labeler(example, index):
    return example, tf.cast(index, tf.int64)

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
    # loads in line by line
    lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir,file_name))
    # maps with class
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

# combine and shuffle
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0] # assign first bit
for labeled_dataset in labeled_data_sets[1:]: # continues adding
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, \
                reshuffle_each_iteration=False)

# see some of the data with labels
for ex in all_labeled_data.take(5):
    print(ex)

# process the words using tokenizer
tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)
# how many words were used?
vocab_size = len(vocabulary_set)
print(vocab_size)

# encode some stuff with the vocab
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
# see an example
example_text = next(iter(all_labeled_data))[0].numpy()
encoded_text = encoder.encode(example_text)
print("Original: ", example_text)
print("Encoded: ", encoded_text)

# encode everythin with a py_function
def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label

def encode_map_fn(text, label):
    # set the shape of the return
    encoded_text, label = tf.py_function(encode, inp=[text, label],
                            Tout=(tf.int64, tf.int64))
    encoded_text.set_shape=([None])
    label.set_shape([])

    return encoded_text, label

# encode it all
all_encoded_data = all_labeled_data.map(encode_map_fn)

# split dataset
### NOTE: The tutorial as found at the website does not indicate that the
# padded_shapes option needs to be set which will result in an error ###
data_shape = ([None], [])
train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, \
                padded_shapes=data_shape) # pad the sizes so they're the
                # same
test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, \
                padded_shapes=data_shape)

# test/train data is batches of data, label pairs
# the padding fills with zeroes and we need to increase vocab size by 1
vocab_size += 1

# making the model
model = tf.keras.Sequential()
# first layer makes vectors of the word lists
model.add(tf.keras.layers.Embedding(vocab_size, 64))
# add a contextual layer, each word is affected by the surrounding words
# LSTM - long short term memory
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
# add a couple layers of sizes in the list
for units in [64,64]:
    model.add(tf.keras.layers.Dense(units, activation='relu'))

model.add(tf.keras.layers.Dense(3))

model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
    metrics=['accuracy'])

# training
model.fit(train_data, epochs=5, validation_data=test_data)
# how does it do?
eval_loss, eval_acc = model.evaluate(test_data)
print("\nEval Loss: {:.3f}, Eval Accuracy: {:.3f}".format(eval_loss,eval_acc))
