import tensorflow as tf
from tensorflow import keras


class CyclicPadding2D(keras.layers.Layer):

    def __init__(self,):
        super(CyclicPadding2D, self).__init__()

    def build(self, input_shape):
        self.grid = tf.Variable(tf.zeros(shape=(input_shape[0], input_shape[1]+2, input_shape[2]+2), dtype=tf.float32), 
                                trainable=False, validate_shape=True)
        super(CyclicPadding2D, self).build(input_shape)

    def call(self, inputs):

        self.grid[:,1:-1, 1:-1].assign(inputs)
        self.grid[:,0,0].assign(inputs[:,-1,-1])
        self.grid[:,0,-1].assign( inputs[:,-1,0])
        self.grid[:,-1,0].assign(inputs[:,0,-1])
        self.grid[:,-1,-1].assign(inputs[:,0,0])
        self.grid[:, 1:-1, 0].assign(inputs[:,:,-1])
        self.grid[:,1 : -1 , -1].assign(inputs[:,:, 0])

        self.grid[:,0, 1:-1].assign(inputs[:,-1,:])
        self.grid[:,-1, 1:-1].assign(inputs[:,0,:])
        return self.grid

class DenseSymmetric2D(tf.keras.layers.Layer):

    def __init__(self,):
        super(DenseSymmetric2D, self).__init__()

    def __call__(self, input_shape):
        
        w1 = tf.constant(tf.keras.initializers.RandomUniform(minval=0.01, maxval=0.09), 
                         shape=(input_shape[0], input_shape[1], input_shape[2]))
        w1 = tf.Variable(tf.keras.initializers.RandomUniform(shape=(input_shape[0], input_shape[1]+2, input_shape[2]+2), dtype=tf.float32), 
                        trainable=False, validate_shape=True)
        w2 = tf.reverse(w1, axis=[0])
        w1 = w1 + w2
        w2 = tf.reverse(w1, axis=[1])
        w1 = w1 + w2
        w2 = tf.transpose(w1)
        self.W = w1 + w2

    
        def __init__(self,):

    def build(self, input_shape):
        self.grid = tf.Variable(tf.zeros(shape=(input_shape[0], input_shape[1]+2, input_shape[2]+2), dtype=tf.float32), 
                        trainable=False, validate_shape=True)
        super(CyclicPadding2D, self).build(input_shape)  

    def call(self, inputs):
        
        self.grid[:,1:-1, 1:-1].assign(inputs)
        self.grid[:,0,0].assign(inputs[:,-1,-1])
        self.grid[:,0,-1].assign( inputs[:,-1,0])
        self.grid[:,-1,0].assign(inputs[:,0,-1])
        self.grid[:,-1,-1].assign(inputs[:,0,0])

class LocallyDense(keras.layers.Layer):
    
    def __init__(self, ):
        super(LocallyDense, self).__init__()
        

    def build(self, input_shape):
        m = input_shape[-2] - 2
        n = input_shape[-1] - 2
        self.w00 = self.add_weight(name="w00", shape=(m,n), initializer="ones", trainable=True)
        self.w01 = self.add_weight(name="w01", shape=(m,n), initializer="ones", trainable=True)
        self.w02 = self.add_weight(name="w02", shape=(m,n), initializer="ones", trainable=True)
        self.w10 = self.add_weight(name="w10", shape=(m,n), initializer="ones", trainable=True)
        self.w11 = self.add_weight(name="w11", shape=(m,n), initializer="ones", trainable=True)
        self.w12 = self.add_weight(name="w12", shape=(m,n), initializer="ones", trainable=True)
        self.w20 = self.add_weight(name="w20", shape=(m,n), initializer="ones", trainable=True)
        self.w21 = self.add_weight(name="w21", shape=(m,n), initializer="ones", trainable=True)
        self.w22 = self.add_weight(name="w22", shape=(m,n), initializer="ones", trainable=True)
        self.b = self.add_weight(name="b", shape=(m,n), initializer='zeros', trainable=True)

    def call(self, padded_input):
        p00 = padded_input[:,:-2,:-2]
        p01 = padded_input[:,:-2,1:-1]
        p02 = padded_input[:,:-2,2:]
        p10 = padded_input[:,1:-1,:-2]
        p11 = padded_input[:,1:-1,1:-1]
        p12 = padded_input[:,1:-1,2:]
        p20 = padded_input[:,2:,:-2]
        p21 = padded_input[:,2:,1:-1]
        p22 = padded_input[:,2:,2:]
        
        return tf.matmul(p00, self.w00) + tf.matmul(p01, self.w01) + tf.matmul(p02, self.w02) + 
        tf.matmul(p10, self.w10) + tf.matmul(p11, self.w11) + tf.matmul(p12, self.w12) + 
        tf.matmul(p20, self.w20) + tf.matmul(p21, self.w21) + tf.matmul(p22, self.w22) + self.b

class Conv2D(keras.layers.Layer):
    
    def __init__(self,kernel):
        super(Conv2D, self).__init__()
        self.kernel = kernel 
        
    def call(self, x):
        print(x.shape)
        x = tf.nn.conv2d(x, self.kernel, strides=1, padding='VALID')
        return x