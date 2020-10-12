import tensorflow as tf
from tensorflow.python.training.tracking.data_structures import NoDependency
import torch


class GameOfLifeModel(tf.keras.models.Model):
    def __init__(self, grid_size, model_3x3):
        super(GameOfLifeModel, self).__init__()
        self.m = grid_size[0]
        self.n = grid_size[1]
        self.model_3x3 = model_3x3


    def build(self, input_shape):
        batch_size = input_shape[0]
        self.grid = tf.Variable(tf.zeros(shape=(batch_size, self.m,self.n)), dtype='float32')

        self.grids_3x3 = NoDependency(dict())
        for i in range(self.m):
            for j in range(self.n):
                self.grids_3x3[(i,j)] = tf.Variable(tf.zeros(shape=(batch_size,3,3)), dtype='float32')


    def call(self, x):
        # find values for the cells in the interior
        for i in range(1,self.m-1):
            for j in range(1,self.n-1):
                #print(self.grids_3x3[(i,j)].shape, i, j)
                #print(x[:,i-1:i+2, j-1:j+2].shape, (i-1,i+2), (j-1,j+2))
                self.grids_3x3[(i,j)].assign(x[:,i-1:i+2, j-1:j+2])
                X = self.model_3x3.call(self.grids_3x3[(i,j)])
                X = tf.reshape(X, shape=(-1,1,1))
                self.grid[:,i:i+1,j:j+1].assign(X)

        # find values for the boundary cells
        for i in range(1, self.m-1):
            self.grids_3x3[(i,0)][:,:,1:].assign(x[:,i-1:i+2,:2])
            self.grids_3x3[(i,0)][:,:,0:1].assign(x[:,i-1:i+2,-1:])
            
            X = self.model_3x3.call(self.grids_3x3[(i,0)])
            X = tf.reshape(X, shape=(-1,1,1))
            self.grid[:,i:i+1,0:1].assign(X)
        
        for i in range(1, self.m-1):
            self.grids_3x3[(i,self.n-1)][:,:,:2].assign(x[:,i-1:i+2,-2:])
            self.grids_3x3[(i,self.n-1)][:,:,-1:].assign(x[:,i-1:i+2,0:1])
            
            X = self.model_3x3.call(self.grids_3x3[(i,self.n-1)])
            X = tf.reshape(X, shape=(-1,1,1))
            self.grid[:,i:i+1,-1:].assign(X)

        for j in range(1, self.n-1):
            self.grids_3x3[(0,j)][:,1:,:].assign(x[:,:2,j-1:j+2])
            self.grids_3x3[(0,j)][:,0:1,:].assign(x[:,-1:,j-1:j+2])
            
            X = self.model_3x3.call(self.grids_3x3[(0,j)])
            X = tf.reshape(X, shape=(-1,1,1))
            self.grid[:,0:1,j:j+1].assign(X)

        for j in range(1, self.n-1):
            self.grids_3x3[(self.m-1,j)][:,2:,:].assign(x[:,:1,j-1:j+2])
            self.grids_3x3[(self.m-1,j)][:,:2,:].assign(x[:,-2:,j-1:j+2])
            
            X = self.model_3x3.call(self.grids_3x3[(self.m-1,j)])
            X = tf.reshape(X, shape=(-1,1,1))
            self.grid[:,-1:,j:j+1].assign(X)

        # point (0,0)
        self.grids_3x3[(0,0)][:,1:,1:].assign(x[:,:2,:2])
        self.grids_3x3[(0,0)][:,:1,1:].assign(x[:,-1:,0:2])
        self.grids_3x3[(0,0)][:,1:,:1].assign(x[:,:2,-1:])
        self.grids_3x3[(0,0)][:,:1,:1].assign(x[:,-1:,-1:])

        X = self.model_3x3.call(self.grids_3x3[(0,0)])
        X = tf.reshape(X, shape=(-1,1,1))
        self.grid[:,0:1,0:1].assign(X)

        # point (0,-1)
        self.grids_3x3[(0,self.n-1)][:,1:,:2].assign(x[:,:2,-2:])
        self.grids_3x3[(0,self.n-1)][:,:1,:2].assign(x[:,-1:,-2:])
        self.grids_3x3[(0,self.n-1)][:,1:,-1:].assign(x[:,:2,:1])
        self.grids_3x3[(0,self.n-1)][:,:1,-1:].assign(x[:,-1:,0:1])
        
        X = self.model_3x3.call(self.grids_3x3[(0,self.n-1)])
        X = tf.reshape(X, shape=(-1,1,1))
        self.grid[:,:1,-1:].assign(X)

        # point (-1, 0)
        self.grids_3x3[(self.m-1,0)][:,:2,-2:].assign(x[:,-2:,:2])
        self.grids_3x3[(self.m-1,0)][:,:2,:1].assign(x[:,-2:,-1:])
        self.grids_3x3[(self.m-1,0)][:,2:,-2:].assign(x[:,:1,:2])
        self.grids_3x3[(self.m-1,0)][:,-1:,:1].assign(x[:,:1,-1:])
        
        X = self.model_3x3.call(self.grids_3x3[(self.m-1,0)])
        X = tf.reshape(X, shape=(-1,1,1))
        self.grid[:,-1:,:1].assign(X)

        # point (-1,-1)
        self.grids_3x3[(self.m-1,self.n-1)][:,:2,:2].assign(x[:,-2:,-2:])
        self.grids_3x3[(self.m-1,self.n-1)][:,:2,-1:].assign(x[:,-2:,:1])
        self.grids_3x3[(self.m-1,self.n-1)][:,-1:,:2].assign(x[:,:1,-2:])
        self.grids_3x3[(self.m-1,self.n-1)][:,-1:,-1:].assign(x[:,:1,:1])
        
        X = self.model_3x3.call(self.grids_3x3[(self.m-1,self.n-1)])
        X = tf.reshape(X, shape=(-1,1,1))
        self.grid[:,-1:,-1:].assign(X)
        return self.grid


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

class ContinuousGameOfLife3x3Pytorch(torch.nn.Module):

    def __init__(self):
        super(ContinuousGameOfLife3x3Pytorch, self).__init__()
        self.k1 = torch.Tensor([[1,1,1],[1,0,1],[1,1,1]])
        self.k2 = torch.Tensor([[0,0,0],[0,1,0],[0,0,0]])
        
    def forward(self, x):
        cell = torch.tensordot(x, self.k2, ([1,2], [0,1]))
        around_cell = torch.tensordot(x, self.k1, ([1,2], [0,1]))

        x1 = torch.max(4-around_cell,torch.zeros_like(cell))
        x2 = torch.max((around_cell + cell)-2,torch.zeros_like(cell))
        x3 = torch.min(x1, x2)
        x4 = torch.min(x3,torch.ones_like(cell))

        return torch.reshape(x4, shape=(-1,1,1))
