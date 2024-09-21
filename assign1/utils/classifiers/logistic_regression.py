"""
Implementations of logistic regression. 
"""

import numpy as np


def logistic_regression_loss_naive(w, X, y, reg):
    """
    Logistic regression loss function, naive implementation (with loops over N samples)

    NOTE:
    In this function, you are NOT supposed to use functions like:
    - np.dot
    - np.matmul (or operator @)
    - np.linalg.norm
    You can (not necessarily) use functions like:
    - np.sum
    - np.log
    - np.exp

    Inputs have dimension D, there are K classes, and we operate on minibatches
    of N examples.

    Inputs:
    - w (float): a numpy array of shape (D + 1,) containing weights.
    - X (float): a numpy array of shape (N, D + 1) containing a minibatch of data.
    - y (uint8): a numpy array of shape (N,) containing training labels; y[i] = k means 
        that X[i] has label k, where k can be either 0 or 1.
    - reg: (float) regularization strength. For regularization, we use L2 norm.

    Returns a tuple of:
    - loss (float): the mean value of loss functions over N examples in minibatch.
    - gradient (float): gradient wrt w, an array of same shape as w
    """

    loss = 0
    dw = np.zeros_like(w)

    ############################################################################
    # TODO:                                                                    #
    # Compute the softmax loss and its gradient using explicit loops.          #
    # Store the results in loss and dW. If you are not careful here, it is     #
    # easy to run into numeric instability.                                    #
    # Don't forget the regularization!                                         #
    # NOTE: You may want to convert y to float for computations. For numpy     #
    # dtypes, see https://numpy.org/doc/stable/reference/arrays.dtypes.html    #
    ############################################################################
    ############################################################################
    #                              START OF YOUR CODE                          #
    ############################################################################
    
    raise NotImplementedError

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################

    return loss, dw


def sigmoid(x):
    """
    Sigmoid function.

    Inputs:
    - x: (float) a numpy array of shape (N,)

    Returns:
    - h: (float) a numpy array of shape (N,), containing the element-wise sigmoid of x
    """

    h = np.zeros_like(x)

    ############################################################################
    # TODO:                                                                    #
    # Implement sigmoid function.                                              #         
    ############################################################################
    ############################################################################
    #                          START OF YOUR CODE                              #
    ############################################################################
    
    raise NotImplementedError

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return h 


def logistic_regression_loss_vectorized(w, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    NOTE:
    In this function, you CAN (not necessarily) use functions like:
    - np.dot (unrecommanded)
    - np.matmul (operator @)
    - np.linalg.norm
    You SHOULD use the function you wrote above:
    - sigmoid

    Inputs and outputs are the same as logistic_regression_loss_naive.
    """

    loss = 0
    dw = np.zeros_like(w)

    ############################################################################
    # TODO:                                                                    #
    # Compute the logistic regression loss and its gradient without using      # 
    # explicit loops.                                                          #
    # Store the results in loss and dW. If you are not careful here, it is     #
    # easy to run into numeric instability. Don't forget the regularization!   #
    # NOTE: For multiplication bewteen vectors/matrices, np.matmul(A, B) is    #
    # recommanded (i.e. A @ B) over np.dot see                                 #
    # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html       #
    # Again, pay attention to the data types!                                  #
    ############################################################################
    ############################################################################
    #                          START OF YOUR CODE                              #
    ############################################################################
    
    raise NotImplementedError

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dw
