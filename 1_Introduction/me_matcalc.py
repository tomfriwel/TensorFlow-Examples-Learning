# Just disables the warning, doesn't enable AVX/FMA
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Create a constant op that produces a 1x2 matrix.
matrix1 = tf.constant([[3., 3.]])  # 1x2
matrix2 = tf.constant([[2.], [2.]])  # 2x1
matrix3 = tf.constant([[2., 2., 5.]])  # 1x3

product = tf.matmul(matrix2, matrix1)

# with tf.Session() as sess:
result = tf.Session().run(product)
print(result)
