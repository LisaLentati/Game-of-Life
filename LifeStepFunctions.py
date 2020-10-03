import numpy as np
from scipy.signal import convolve2d
import tensorflow as tf


def life_step_for_arrays(X):
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))

def life_step_for_tensors(X):
    """
    Args:
        X: 3D tensor of shape (batch_size, d1, d2).
    """

    X = tf.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
    kernel = tf.ones(shape=(3,3,1,1))
    X = tf.nn.conv2d(X, filters=kernel, padding='VALID')
    return X

def life_step_for_tensors_cyclic(X):
    # TODO
    return