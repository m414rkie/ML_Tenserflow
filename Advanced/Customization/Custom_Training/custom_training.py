#! usr/bin/python3

# tutorial from tensorflow.org. Tutorial is on how to select and train models
# using the iris dataset.

# Jon Parsons
# 7-7-2020

import os
import matplotlib.pyplot as plt
import tensorflow as tf

# get dataset
train_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_file_path = tf.keras.utils.get_file(fname=os.path.basename(train_url), \
        origin=train_url)
print("Dataset is found locally at\n{}".format(train_file_path))

# begin priming for inputting data into tensorflow
# column order is known
col_names = ['sepal_length','sepal_width','petal_length','petal_width','species']

feature_names = col_names[:-1]
label_name = col_names[-1]
# known
classes = ['Iris Setosa','Iris Versicolor','Iris Virginica']

# load dataset into memory
batch_size = 32
train_set = tf.data.experimental.make_csv_dataset(train_file_path, batch_size, \
        column_names=col_names, label_name=label_name, num_epochs=1)

# examine a batch of the data
features, labels = next(iter(train_set))
print("Examining Data")
print(features)

# visualize some data
sepal = plt.figure(1)
plt.scatter(features['petal_length'],features['sepal_length'],c=labels, \
        cmap='viridis')
plt.xlabel("Petal Length")
plt.ylabel("Sepal Length")
sepal.savefig("petal_sepal.png", bbox_inches='tight')

# function that packes features with the labels
def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

# set data
train_dataset = train_set.map(pack_features_vector)
features, labels = next(iter(train_dataset))

print(features[:5])

model = tf.keras.Sequential([ \
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)), \
    tf.keras.layers.Dense(10, activation=tf.nn.relu), \
    tf.keras.layers.Dense(3)])

# see some of the pretraining predictions
predictions = model(features)
print(predictions[:5])
# and class predictions
print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))

# begin setting up the training
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss function
def loss(model, x, y, training):
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)

l = loss(model, features, labels, training=False)
# test the loss
print("Loss test: {}".format(l))

# set up gradient
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# define optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# first set of training
loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(), \
    loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables))
print("Step: {},         Loss:{}".format(optimizer.iterations.numpy(), \
    loss(model, features, labels, training=True).numpy()))

# define the training loop
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg.update_state(loss_value)
        epoch_accuracy.update_state(y, model(x, training=True))

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    # print the ongoing training
    if epoch % 50 == 0:
        print("Epoch {:03d} | Loss: {:.3f} | Accuracy: {:.3f}".format( \
            epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

# training visualizations
fig, axes = plt.subplots(2, sharex=True, figsize=(12,8))
fig.suptitle("Training Metrics")

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.savefig("training_results.png", bbox_inches='tight')

# define the test dataset
test_url = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv'
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),origin=test_url)

test_dataset = tf.data.experimental.make_csv_dataset(test_fp, batch_size, \
column_names=col_names, label_name='species', num_epochs=1, shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

# determine testing accuracy
test_accuracy = tf.keras.metrics.Accuracy()

for (x,y) in test_dataset:
    logits = model(x, training=False)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

print("Testing accuracy: {:.3%}".format(test_accuracy.result()))

# predictions from un-labeled data
predict_dataset = tf.convert_to_tensor([ \
    [5.1, 3.3, 1.7, 0.5,], \
    [5.9, 3.0, 4.2, 1.5,], \
    [6.9, 3.1, 5.4, 2.1,] ])

predictions = model(predict_dataset, training=False)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = classes[class_idx]
    print("Examples {} | Prediction: {} ({:4.1f}%)".format(i, name, 100*p))
