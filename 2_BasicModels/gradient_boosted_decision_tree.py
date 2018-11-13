import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Ignore all GPUs, tf random forest does not benefit from it.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner

# Set verbosity to display errors only
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=False, source_url="http://yann.lecun.com/exdb/mnist/")

batch_size = 4096
num_classes = 10
num_features = 784
max_steps = 100

# GBDT Parameters
learning_rate = 0.1
l1_regul = 0.
l2_regul = 1.
examples_per_layer = 1000
num_trees = 10
max_depth = 16

# Fill GBDT parameters into the config proto
learner_config = gbdt_learner.LearnerConfig()
learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
learner_config.regularization.l1 = l1_regul
learner_config.regularization.l2 = l2_regul / examples_per_layer
learner_config.constraints.max_tree_depth = max_depth
growing_mode = gbdt_learner.LearnerConfig.LAYER_BY_LAYER
learner_config.growing_mode = growing_mode
run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)
learner_config.multi_class_strategy = (gbdt_learner.LearnerConfig.DIAGONAL_HESSIAN)

# Create a TensorFlor GBDT Estimator
gbdt_model = GradientBoostedDecisionTreeClassifier(
    model_dir=None, # No saved directory specified
    learner_config=learner_config,
    n_classes=num_classes,
    examples_per_layer=examples_per_layer,
    num_trees=num_trees,
    center_bias=False,
    config=run_config)

# Display TF info logs
tf.logging.set_verbosity(tf.logging.INFO)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image":mnist.train.images},
    y=mnist.train.labels,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True
)

# Train the model
gbdt_model.fit(input_fn=input_fn, max_steps=max_steps)

# Evaluate the model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"images":mnist.test.images},
    y=mnist.test.labels,
    batch_size=batch_size,
    shuffle=False)

# Use the Estimator 'evaluate' method
e = gbdt_model.evaluate(input_fn=input_fn)

print("Test Accuracy:", e["accuracy"])