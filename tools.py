import tensorflow as tf
import numpy as np 
from scipy.signal import convolve2d


class CyclicPadding2D(tf.keras.layers.Layer):
    """
    It adds a cyclic padding around the two last dimensions of the tensor. 
    """

    def __init__(self,):
        super(CyclicPadding2D, self).__init__()

    def build(self, input_shape):
        self.grid = tf.Variable(tf.zeros(shape=(input_shape[0], input_shape[1]+2, input_shape[2]+2), dtype=tf.float32), 
                                trainable=False, validate_shape=True)
        super(CyclicPadding2D, self).build(input_shape)

    def call(self, inputs):
        """
        Args:
            inputs: a 3D tensor of shape (batch_size, d1, d2)

        Returns:
            The padded 3D tensor of shape (batch_size, d1+2, d+2)
        """

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


def life_step_for_arrays(X):
    neighbours_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    return (neighbours_count == 3) | (X & (neighbours_count == 2))


def life_step_for_tensors(X):
    """
    X: 3D tensor of shape (batch_size, d1, d2).
    """
    pad = CyclicPadding2D()
    X_padded = pad(X)
    X_4D = tf.reshape(X_padded, (X_padded.shape[0], X_padded.shape[1], X_padded.shape[2], 1))

    kernel = tf.Variable(tf.ones(shape=(3,3,1,1)), dtype='float32')
    kernel[1,1,0,0].assign(0)
    
    neighbours_count_4D = tf.nn.conv2d(X_4D, filters=kernel, strides=1, padding='VALID')
    neighbours_count_3D = tf.reshape(neighbours_count_4D, (neighbours_count_4D.shape[0], neighbours_count_4D.shape[1], 
                                                     neighbours_count_4D.shape[2]))
    
    return tf.cast((neighbours_count_3D == 3) | (tf.cast(X, dtype=bool) & (neighbours_count_3D == 2)), dtype='float32')


def generate_input(trials, grid_shape=(25,25), v_min=0.1, v_max=0.9, steps_before_generating_output=2):
    """ Generates random grids for the Game of Life. Each cell in the grid has probability p to be alive,
    where p is chosen uniformly between v_min and v_max. Only the non-zero grids are returned. 

    Args:
        trials (int): Number of times the codes tries to generate a valid grid.
        grid_shape (tuple): Shape of the grid. Defaults to (25,25).
        v_min (float): Lower bound for the probability of a cell being alive. Defaults to 0.1.
        v_max (float): Upper bound for the probability of a cell being alive. Defaults to 0.9.
        steps_before_generating_output (int): Defaults to 2.

    Returns:
        samples (3D tensor): the generated grids 
        samples_1_step (3D tensor): the state of the generated grids after one step
    """

    samples = []

    probs = np.random.uniform(v_min, v_max, trials)

    for prob in probs:
        grid = np.random.binomial(n=1, p=prob, size=grid_shape)
        for _ in range(steps_before_generating_output):
            grid = life_step_for_arrays(grid)
            
        if grid.sum() > 0:
            samples.append(grid)
            
    samples = tf.constant(np.array(samples).astype(float), dtype="float32")
    samples_1_step = life_step_for_tensors(samples)

    return samples, samples_1_step

    
def create_masks(size):
    blank_line = [0.]*size

    m1 = [0.,1.] * int(np.ceil(size/2))
    m1 =m1[:size]

    m2 = [1., 0.] * int(np.ceil(size/2))
    m2 =m2[:size]
    
    M1 = [m1,blank_line]*int(np.ceil(size/2))
    M2 = [m2,blank_line]*int(np.ceil(size/2))
    M3 = [blank_line, m1]*int(np.ceil(size/2))
    M4 = [blank_line, m2]*int(np.ceil(size/2))
    
    M1 = tf.constant(M1)[:size]
    M2 = tf.constant(M2)[:size]
    M3 = tf.constant(M3)[:size]
    M4 = tf.constant(M4)[:size]
    return M1, M2, M3, M4
