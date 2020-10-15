import tensorflow as tf
from tensorflow.python.training.tracking.data_structures import NoDependency
import torch


class ContinuousGameOfLife(tf.keras.layers.Layer):
    
    def __init__(self, ):
        super(ContinuousGameOfLife, self).__init__()
        self.flat = tf.keras.layers.Flatten()
        self.add_padding = CyclicPadding2D()

    def build(self, input_shape):
        self.k1 = tf.constant([[1,1,1],[1,0,1],[1,1,1]], dtype='float32')
        self.k1 = tf.reshape(self.k1, shape=(3,3,1,1))
        self.k2 = tf.constant([[0,0,0],[0,1,0],[0,0,0]], dtype='float32')
        self.k2 = tf.reshape(self.k2, shape=(3,3,1,1))
        super(ContinuousGameOfLife, self).build(input_shape)
        
    def call(self, inputs):
        batch_size, d1, d2 = inputs.shape
        x = self.add_padding(inputs)
        x = tf.reshape(x, shape=(batch_size, d1+2, d2+2, 1))
        cell = tf.nn.conv2d(x, filters=self.k2, strides=1, padding='VALID')
        around_cell = tf.nn.conv2d(x, filters=self.k1, strides=1, padding='VALID')

        x1 = tf.math.maximum(4-around_cell,0)
        x2 = tf.math.maximum((around_cell + cell)-2,0)
        x3 = tf.math.minimum(x1, x2)
        x4 = tf.math.minimum(x3,1)

        return tf.reshape(x4, shape=(batch_size,d1,d2))


class ContinuousGameOfLife3x3(tf.keras.layers.Layer):
    
    def __init__(self, ):
        super(ContinuousGameOfLife3x3, self).__init__()

    def build(self, input_shape):
        self.k1 = tf.constant([[1,1,1],[1,0,1],[1,1,1]], dtype='float32')
        self.k2 = tf.constant([[0,0,0],[0,1,0],[0,0,0]], dtype='float32')
        super(ContinuousGameOfLife3x3, self).build(input_shape)
        
    def call(self, inputs):
        cell = tf.tensordot(inputs, self.k2, axes=([1,2], [0,1]))
        around_cell = tf.tensordot(inputs, self.k1, axes=([1,2], [0,1]))

        x1 = tf.math.maximum(4-around_cell,0)
        x2 = tf.math.maximum((around_cell + cell)-2,0)
        x3 = tf.math.minimum(x1, x2)
        x4 = tf.math.minimum(x3,1)

        return tf.reshape(x4, shape=(-1,1,1))

