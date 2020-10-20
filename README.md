# Conway's reverse game of life
Code created for the Kaggle competition Conway's reverse game of life.

**Link of the competition**: <https://www.kaggle.com/c/conways-reverse-game-of-life-2020>

**The task** : given a 25x25 cyclic grid, can we predict what was the state of the grid one step before? 


**The approach**:
- We create a function _f_ which extends the game of life to all values between 0 and 1. 

- We then look for an input grid which minimizes the loss between the target grid and _f_(input grid). 


In the jupyter notebooks we explore different ways to define _f_ and different ways to train the model.


## Defining _f_
In the Game of Life, the next state of a cell depends on:
- the current state of the cell,
- the current number of alive cells surrounding the cell we are considering.


