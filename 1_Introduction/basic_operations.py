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
