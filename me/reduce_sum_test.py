# #!/usr/bin/python
# coding:utf-8

# Just disables the warning, doesn't enable AVX/FMA
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# https://www.dotnetperls.com/reduce-sum-tensorflow

# Reduce_sum, TensorFlow. In machine learning, we must think in terms of numeric operations on arrays (matrices). Methods like reduce_sum are important here.

# For reduce_sum, we sum elements across dimensions. The method applies "dimensionality reduction" to do this. The result is a sum of numbers.

# An example. Here we have a constant array of integers. It is a 2-dimensional array (a matrix). Each subarray has 2 elements. We use tensors that call reduce_sum on this array.

# First argument:
# The first argument to reduce_sum is the tensor we want to sum the elements of.
# Second argument:
# We can specify the "axis" we wish to sum—this changes the values summed and the output.

import tensorflow as tf

a = tf.constant([[1, 3, 2], [2, 0, 4], [0, 1, 10]])

b = tf.reduce_sum(a)
c = tf.reduce_sum(a, 0)
d = tf.reduce_sum(a, 1)
e = tf.reduce_sum(a, 1)

tensors = [b, c, d, e]

with tf.Session() as sess:
    for tensor in tensors:
        result = sess.run(tensor)
        print(result)
        