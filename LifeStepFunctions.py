import numpy as np
from scipy.signal import convolve2d
import tensorflow as tf


def life_step_for_arrays(X):
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))

def life_step_for_tensors(X):
    """
    Not cyclic
    Args:
        X: 3D tensor of shape (batch_size, d1, d2).
    """
    X_4D = tf.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
    kernel = tf.Variable(tf.ones(shape=(3,3,1,1)), dtype='float32')
    kernel[1,1,0,0].assign(0)
    counts = tf.nn.conv2d(X_4D, filters=kernel, strides=1, padding='VALID')
    counts = tf.reshape(counts, (counts.shape[0], counts.shape[1], counts.shape[2]))

    X_valid = X[:,1:-1,1:-1]
    return tf.cast((counts == 3) | (tf.cast(X_valid, dtype=bool) & (counts == 2)), dtype='float32')

def life_step_for_tensors_cyclic(X):
    # TODO
    return