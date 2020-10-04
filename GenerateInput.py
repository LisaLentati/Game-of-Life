import numpy as np
import tensorflow as tf

from LifeStepFunctions import life_step_for_arrays

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
    samples_1_step = []

    probs = np.random.uniform(v_min, v_max, trials)

    for prob in probs:
        grid = np.random.binomial(n=1, p=prob, size=grid_shape)
        for _ in range(steps_before_generating_output):
            grid = life_step_for_arrays(grid)
            
        if grid.sum() > 0:
            samples.append(grid)
            grid_1_step = life_step_for_arrays(grid)
            samples_1_step.append(grid_1_step)

    samples = tf.constant(np.array(samples).astype(float), dtype="float32")
    samples_1_step = tf.constant(np.array(samples_1_step).astype(float), dtype="float32")

    return samples, samples_1_step