# Mixed National Institute of Standards and Technology database
# Import MNIST
from tensorflow.examples.tutorials.mnist import input_data
print(input_data)
# input_data.maybe_download("train-images-idx3-ubyte.gz", "/tmp/data/")
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# # Load data
# X_train = mnist.train.images
# Y_train = mnist.train.labels
# X_test = mnist.test.images
# Y_test = mnist.test.labels

# # Get the next 64 images array and labels
# batch_X, batch_Y = mnist.train.next_batch(64)