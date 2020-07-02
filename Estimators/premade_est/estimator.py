#! usr/bin/python3

# This code from the tutorials at Tensorflow.org. This tutorial is an introduction
# to the premade estimators. The data set is a set of iris measurements and
# their species

# Jon Parsons
# 7-1-2020

import tensorflow as tf
import pandas as pd

# set column and species names (known)
CSV_COL_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# get the data and read using the pandas csv reader
train_path = tf.keras.utils.get_file("iris_training.csv", \
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", \
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COL_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COL_NAMES, header=0)

# inspect some of the data
print(train.head())

# seperate the species (label) from the rest
train_label = train.pop('Species')
test_label = test.pop('Species')

# the estimator needs input functions, which output a tuple of features and label
# a quick example (known values)
def input_evaluation_set():
    features = {'SepalLength': np.array([6.4,5.0]), \
                'SepalWidth': np.array([2.8,2.3]), \
                'PetalLength': np.array([5.6,3.3]), \
                'PetalWidth': np.array([2.2,1.0])}
    labels = np.array([2,1])
    return features, labels

# now the one for use
def input_fn(features, labels, training=True, batch_size=256):
    # make the dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# define feature columns for the model
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# implementing the estimator here. This tutorial uses a DNN
classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, \
        # two hidden layers, 3 classes to predict
        hidden_units=[30,10],n_classes=3)

# train the model
classifier.train(input_fn=lambda: input_fn(train, train_label, training=True),
    steps=5000)

# evaluate the model
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_label, \
    training=False))

# making predictions
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1]
}

def input_fn_p(features, batch_size=256):
    # no labels in this one
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predictions = classifier.predict(input_fn=lambda: input_fn_p(predict_x))

for pred_dict, expect in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    prob = pred_dict['probabilities'][class_id]

    print('Predicted "{}" ({:.1f}%) | Expected "{}"'.format(SPECIES[class_id], \
        100*prob, expect))
