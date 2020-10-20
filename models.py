import tensorflow as tf
from tools import CyclicPadding2D

class ContinuousGameOfLife(tf.keras.layers.Layer):
    
    def __init__(self, game_function):
        super(ContinuousGameOfLife, self).__init__()
        self.forward_game = game_function
        
        self.add_padding = CyclicPadding2D()

        self.k1 = tf.constant([[1,1,1],[1,0,1],[1,1,1]], shape=(3,3,1,1), dtype='float32')
        self.k2 = tf.constant([[0,0,0],[0,1,0],[0,0,0]], shape=(3,3,1,1), dtype='float32')    
        
    def call(self, inputs):
        batch_size, d1, d2 = inputs.shape
        x = self.add_padding(inputs)
        x = tf.reshape(x, shape=(batch_size, d1+2, d2+2, 1))
        cell = tf.nn.conv2d(x, filters=self.k2, strides=1, padding='VALID')
        around_cell = tf.nn.conv2d(x, filters=self.k1, strides=1, padding='VALID')

        xx = self.forward_game(cell, around_cell)
        
        return tf.reshape(xx, shape=(batch_size,d1,d2))        


class ContinuousReverseGame(tf.keras.models.Model):
    
    def __init__(self, game_function, min_v, max_v, grid_len):
        super(ContinuousReverseGame, self).__init__()
        self.forward_game = game_function
        self.min_v = min_v
        self.max_v = max_v
        self.l = grid_len
        self.k1 = tf.constant([[1,1,1],[1,0,1],[1,1,1]], shape=(3,3,1,1), dtype='float32')
        self.k2 = tf.constant([[0,0,0],[0,1,0],[0,0,0]], shape=(3,3,1,1), dtype='float32')

        self.input_img = tf.Variable(tf.random.uniform(shape=(1,self.l+2,self.l+2), minval=self.min_v, maxval=self.max_v), trainable=True, validate_shape=True) #constraint=tf.keras.constraints.min_max_norm(0,1))

        
    def call(self, target):
        self.input_img[:,0,:].assign(self.input_img[:,-2,:])
        self.input_img[:,-1,:].assign(self.input_img[:,1,:])
        self.input_img[:,:,0].assign(self.input_img[:,:,-2])
        self.input_img[:,:,-1].assign(self.input_img[:,:,1])

            
        input_img = tf.reshape(self.input_img, shape=(1, self.l+2, self.l+2, 1))
        cell = tf.nn.conv2d(input_img, filters=self.k2, strides=1, padding='VALID')
        around_cell = tf.nn.conv2d(input_img, filters=self.k1, strides=1, padding='VALID')

        xx = self.forward_game(cell, around_cell)
        xx = tf.reshape(xx, shape=(self.l,self.l))        
        return xx
