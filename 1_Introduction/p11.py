import tensorflow as tf

hello = tf.constant('Hello, tomfriwel')
sess = tf.Session()
print(sess.run(hello))