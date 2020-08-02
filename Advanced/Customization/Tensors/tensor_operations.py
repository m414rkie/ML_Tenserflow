#! usr/bin/python3

# tutorial from Tensorflow.org. Tutorial is on how to manipulate tensors and
# use GPU acceleration

# Jon Parsons
# 7-5-2020

import tensorflow as tf

# tensorflow has a full library of math operations, including a linalg inverse
# operation
print(tf.add(1,2))
print(tf.add([1,2],[3,4]))
print(tf.square(5))
print(tf.reduce_sum([1,2,3]))
print(tf.square(2)+tf.square(3))
print("\n")
x = tf.matmul([[1]],[[2,3]])
print(x)
print(x.shape)
print(x.dtype)
print("\n")

# conversion between tf.Tensor and np.ndarray
import numpy as np
ndarray = np.ones([3,3])
# automatically convert numpy arrays to tensors
tensor = tf.multiply(ndarray, 42)
print(tensor)
# automatically converts from tensor to numpy
print(np.add(tensor, 1))
# .numpy() converts to numpy
print(tensor.numpy())

# GPU acceleration
x = tf.random.uniform([3,3])
print("Check for GPU availability")
print(tf.config.experimental.list_physical_devices("GPU"))
print("On GPU #0?")
print(x.device.endswith('GPU:0'))
# Laptop has no GPU's :(
# Explicit device placement
import time

def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)

    result = time.time() - start
    print("10 loops: {:0.2f}ms".format(1000*result))

    return

# force CPU
print("On CPU")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# force GPU
if tf.config.experimental.list_physical_devices("GPU"):
    print("On GPU:")
    with tf.device("GPU:0"):
        x.tf.random.uniform([1000,1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)

# How to do datasets
# making a data pipeline
ds_tensors = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])
# make a csv
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 1
    Line 2
    Line 3
    """)

ds_file = tf.data.TextLineDataset(filename)
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

# iterate elements
print("Elements of tensor:")
for x in ds_tensors:
    print(x)
print("Elements of file:")
for x in ds_file:
    print(x)
