#! usr/bin/python3

# Tutorial from tensorflow.org. Tutorial covers how to create custom layers.

# Jon Parsons
# 7-5-2020

import tensorflow as tf

# is the GPU available
print(tf.test.is_gpu_available())

# layers can be objects
layer = tf.keras.layers.Dense(100)
# and can be modified later
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))
# call the layer to do things
layer(tf.zeros([10,5]))
# see the variables
print("\n\nKeras Dense Layer")
print(layer.variables)
print(layer.kernel)
print(layer.bias)

# building a custom layer by expanding a keras.layer class
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
    # defining variables in build require shapes to be specified
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), \
                        self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)

# make the custom layer with 10 outputs
layer = MyDenseLayer(10)
_ = layer(tf.zeros([10,5]))
print("\n\nCustom Dense Layer")
print(layer.variables)
print([var.name for var in layer.trainable_variables])

# composing pre-existing layers together
# a resnet block is here
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernal_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1,1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernal_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters2, (1,1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor # residual part

        return tf.nn.relu(x)

# create the block
block = ResnetIdentityBlock(1, [1,2,3])
_ = block(tf.zeros([1,1,2,2])) # some discrepency between actual and tutorial
# input shapes
print("\n\nResidual Block")
print(block.variables)
print(block.layers)
print(len(block.variables))
print(block.summary())

# calling layers sequentially uses the Sequential keyword, howver you lose the
# residual properties of the above block
my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1,(1,1), input_shape= \
    (None,None,3)), \
    tf.keras.layers.BatchNormalization(), \
    tf.keras.layers.Conv2D(2,1, padding='same'), \
    tf.keras.layers.BatchNormalization(), \
    tf.keras.layers.Conv2D(3, (1,1)), \
    tf.keras.layers.BatchNormalization()])

my_seq(tf.zeros([1,2,3,3]))
print("\n\nSequential")
print(my_seq.summary())
