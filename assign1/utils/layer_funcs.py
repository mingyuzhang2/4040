"""
Implementation of layer functions.
"""

import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine transformation function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """

    ############################################################################
    # TODO:                                                                    #
    # Implement the affine forward pass. Store the result in 'out'.            #
    # You will need to reshape the input into rows.                            #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    raise NotImplementedError

    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine transformation function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - x: input data, of shape (N, d_1, ... d_k)
    - w: weights, of shape (D, M)
    - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """

    ############################################################################
    # TODO: Implement the affine backward pass.                                #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    raise NotImplementedError

    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs) activation function.

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """

    ############################################################################
    # TODO: Implement the ReLU forward pass.                                   #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    raise NotImplementedError

    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs) activation function.

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """

    ############################################################################
    # TODO: Implement the ReLU backward pass.                                  #
    # NOTE: You may want to use np.where, see                                  #
    # https://numpy.org/doc/stable/reference/generated/numpy.where.html        #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    raise NotImplementedError

    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return dx


def tanh_forward(x):
    """
    Computes the forward pass for the tanh activation function.

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """

    ############################################################################
    # TODO: Implement the tanh forward pass.                                   #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    raise NotImplementedError

    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return out


def tanh_backward(dout, x):
    """
    Computes the backward pass for the tanh activation function.

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """

    ############################################################################
    # TODO: Implement the tanh backward pass.                                  #
    # NOTE: You may want to use the derivative of tanh                         #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    raise NotImplementedError

    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    This adjusts the weights to minimize loss.
    y_prediction = argmax(softmax(x))

    Inputs:
    - x: (float) a tensor of shape (N, #classes)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradient of loss wrt input x
    """

    # Initialize the loss.
    #loss = 0.0
    #dx = np.zeros_like(x)

    # When calculating the cross entropy,
    # you may meet another problem about numerical stability, log(0).
    # To avoid this, you can add a small number to it, log(0+epsilon).
    epsilon = 1e-15

    from .classifiers.softmax import softmax, onehot, cross_entropy

    ############################################################################
    # TODO:                                                                    #
    # You can use the previous softmax loss function here.                     #
    # Hint:                                                                    #
    #   * Be careful of overflow problem.                                      #
    #   * You may use the functions you wrote in task1                         #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################

    raise NotImplementedError

    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return loss, dx


def check_accuracy(preds, labels):
    """
    Return the classification accuracy of input data.

    Inputs:
    - preds: (float) a tensor of shape (N,)
    - y: (int) an array of length N. ground truth label 
    Returns: 
    - acc: (float) between 0 and 1
    """

    return np.mean(np.equal(preds, labels))
