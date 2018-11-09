# TensorFlow-Examples-Learning
Learning resp of TensorFlow-Examples

#### `sess.run`
```
Runs operations and evaluates tensors in fetches.

This method runs one "step" of TensorFlow computation, by running the necessary graph fragment to execute every Operation and evaluate every Tensor in fetches, substituting the values in feed_dict for the corresponding input values.

The fetches argument may be a single graph element, or an arbitrarily nested list, tuple, namedtuple, dict, or OrderedDict containing graph elements at its leaves. A graph element can be one of the following types:
```

`tf.Session().run`'s first parameter may take a graph element. I think the graph element is just like a mathematical formula.

`tf.placeholder` just like a variable in mathematical formula.