import tensorflow as tf

# Basic 
a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph.
with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants: %i" % sess.run(a*b))

# Basic operations with variable as graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the default graph.
with tf.Session() as sess:
    # Run every operations with variable input
    print("Addtion with variable: %i" % sess.run(add, feed_dict={a:2, b:3}))
    print("Multiplication with variable: %i" % sess.run(mul, feed_dict={a:2, b:3}))

# Create a constant op that produces a 1x2 matrix.
matrix1 = tf.constant([[3., 3.]])   #1x2
matrix2 = tf.constant([[2.], [2.]]) #2x1
matrix3 = tf.constant([[2., 2., 5.]]) #1x3

product = tf.matmul(matrix2, matrix1)

print(product)

# with tf.Session() as sess:
result = tf.Session().run(product)
print(result)
