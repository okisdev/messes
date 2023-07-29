import numpy as np
import tensorflow as tf

# Tensor
print("\n --- Tensor --- \n")

tensor_1 = tf.constant(0)
print(f"Print constant tensor {tensor_1} of rank {tf.rank(tensor_1)}")
print("Tensor:", tensor_1)

print("\n")

tensor_2 = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9])
print("Tensor:", tensor_2)
print("Rank:", tf.rank(tensor_2).numpy())


# Tensor operations
print("\n --- Tensor operations --- \n")

x = tf.constant([[1, 1], [1, 1]])
y = tf.constant([[2, 4], [6, 8]])

print(tf.add(x, y), "\n")
print(tf.matmul(x, y), "\n")


# Multi-dimensional tensors
print("\n --- Multi-dimensional tensors --- \n")

tensor_3 = tf.ones(shape=[1, 2, 3], dtype=tf.float32)
print("Tensor:", tensor_3)
print("Tensor Rank: ", tf.rank(tensor_3).numpy())
print("Shape: ", tensor_3.shape)
print("Elements' type", tensor_3.dtype)
print("The size of the second axis:", tensor_3.shape[1])
print("The size of the last axis:", tensor_3.shape[-1])
print("Total number of elements: ", tf.size(tensor_3).numpy())
print("How many dimensions? ", tensor_3.ndim)


# Indexing
print("\n --- Indexing --- \n")

tensor_4 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(tensor_4[:].numpy())
print(tensor_4[0, :].numpy())
print(tensor_4[0, -1].numpy())
print(tensor_4[1:, 2:].numpy)


# Data types
print("\n --- Data types --- \n")

tensor_5 = tf.constant([1, 2, 3, 4], dtype=tf.float32)
print("Tensor:", tensor_5)
print("Tensor type:", tensor_5.dtype)

casted_tensor_5 = tf.cast(tensor_5, dtype=tf.int32)
print("Casted tensor:", casted_tensor_5)
print("Casted tensor type:", casted_tensor_5.dtype)
