# Just disables the warning, doesn't enable AVX/FMA
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Set Eager API
print("Setting Eager mode...")
tfe.enable_eager_execution()

print("Define constant tensors")
a = tf.constant(2)
print("a = %i" % a)
b = tf.constant(3)
print("b = %i" % b)

# Run the operation without the need of tf.Session
c = a + b
print("a + b = %i" % c)
d = a * b
print("a * b = %i" % d)

# Full compatibility with Numpy
# Define constant tensors
a = tf.constant([[2., 1.],[1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)
b = np.array([[3., 0.],[5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)

# 2 1    3 0
# 1 0    5 1

# 11 1
# 3 0

c = a + b
print("a + b = %s", c)

d = tf.matmul(a, b)
print("a * b = %s", d)


print("Iterate through Tensor 'a':")
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])
        
    
