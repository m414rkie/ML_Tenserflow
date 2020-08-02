#! usr/bin/python3

# Code from the custom training tutorial at Tensorflow.org. Creates a real
# simple linear model, which when trained will have a weight of 3 and bias of 2.

# Jon Parsons
# 7-6-2020

import tensorflow as tf
import matplotlib.pyplot as plt

## skipping the assertion learning parts ##

# class holding the linear function, f(x) = Wx + b
class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0) # not initiating from random
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W*x + self.b

model = Model()
# check the implementation
assert model(3.0).numpy() == 15.0

# define the loss function
def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))

# get some data by adding Gaussian noise to vectors of the inputs
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random.normal(shape=[NUM_EXAMPLES])
noise = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs*TRUE_W + TRUE_b + noise

# visualizing the data
data_plot = plt.figure(1)
plt.scatter(inputs, outputs, c='b', label='With Noise')
plt.scatter(inputs, model(inputs), c='r', label='Untrained Model Output')
plt.title("Untrained Data")
plt.legend()
plt.xlabel("Inputs")
plt.ylabel("Outputs")
data_plot.savefig("untrained.png", bbox_inches='tight')

# untrained loss
print("Untrained loss: %1.6f" % loss(model(inputs), outputs).numpy())

# define the training loop
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(outputs, model(inputs))
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate*dW)
    model.b.assign_sub(learning_rate*db)

# begin training the model, with outputs of the parameters
model = Model()

Ws, bs = [], []
ls = []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(outputs, model(inputs))
    ls.append(current_loss.numpy())

    train(model, inputs, outputs, learning_rate=0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' % (epoch, Ws[-1], bs[-1], ls[-1]))

weights = plt.figure(2)
plt.plot(epochs, Ws, 'r', label='W')
plt.plot(epochs, bs, 'b', label='b')
plt.plot([TRUE_W]*len(epochs), 'r--', label='True W')
plt.plot([TRUE_b]*len(epochs), 'b--', label='True b')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Value")
weights.savefig("training_params.png", bbox_inches='tight')

lss = plt.figure(3)
plt.plot(epochs, ls, 'g', label="Training Loss")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
lss.savefig('loss.png', bbox_inches='tight')

# visualizing the data
trnd_plot = plt.figure(4)
plt.scatter(inputs, outputs, c='b', label='With Noise')
plt.scatter(inputs, model(inputs), c='r', label='Trained Model Output')
plt.title("Trained Data")
plt.legend()
plt.xlabel("Inputs")
plt.ylabel("Outputs")
data_plot.savefig("trained.png", bbox_inches='tight')
