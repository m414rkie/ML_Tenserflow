#!usr/bin/python3
# checks installation of Tensorflow

# Jon Parsons
# 5-31-2020

import tensorflow as tf

print(tf.__version__)

x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

result = tf.multiply(x1,x2)

print(result)
