import tensorflow as tf

# Check version
print("Tensorflow version: ", tf.__version__)

# Test TensorFlow for cuda availability
print("Tensorflow is built with CUDA: ", tf.test.is_built_with_cuda())

# Check devices
print("All devices: ", tf.config.list_physical_devices(device_type=None))
print("GPU devices: ", tf.config.list_physical_devices(device_type='GPU'))

# Print a randomly generated tensor
# tf.math.reduce_sum: https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum
# tf.random.normal: https://www.tensorflow.org/api_docs/python/tf/random/normal
print(tf.math.reduce_sum(tf.random.normal([1, 10])))
