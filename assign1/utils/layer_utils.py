"""
Implementation of layer utilities. 
"""

import numpy as np


class AffineLayer:
    """ An affine hidden layer performs an affine transform """

    def __init__(self, input_dim, output_dim=100, weight_scale=1e-2):
        """
        Initialize weight W with random value and bias b with zero.

        Inputs:
        - input_dim: (int) the number of input neurons, 
            like D or D1xD2x...xDn.
        - output_dim: (int) the number of hidden neurons in this layer

        Variables to cache:
        - W: weights
        - b: bias
        """

        self.input_dim = input_dim
        self.output_dim = output_dim

        W = np.random.rand(input_dim, output_dim) * weight_scale
        b = np.zeros(output_dim)

        # parameter cache
        self.params = {'W': W, 'b': b}
        self.cache = {}
        self.gradients = {}

    def feedforward(self, X):
        """
        Feed forward function
        Inputs:
        - X: (float) a tensor of shape (N,D) or (N, D1, D2, ..., Dn)

        Variables to cache:
        - X: input

        Returns:
        - out: output of shape (N, output_dim)
        """

        W, b = self.params['W'], self.params['b']

        from .layer_funcs import affine_forward

        ############################################################################
        # TODO:                                                                    #
        # - Affine transform forward pass                                          #
        #   Use functions in layer_funcs.py                                        #
        # - Cache the layer outputs in self.cache                                  #
        #   NOTE: Here the answer is provided as an example                        #
        ############################################################################
        ############################################################################
        #                           START OF YOUR CODE                             #
        ############################################################################

        out = affine_forward(X, W, b)

        raise NotImplementedError
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Cache the output X into self.cache for later back propogation
        # NOTE: You MUST make sure the name here matches the one in backward()
        self.cache['X'] = X

        return out

    def backward(self, dout):
        """
        Backpropagation function
        Inputs:
        - dout: (float) a tensor with shape (N, hidden_dim)
            Here hidden_dim denotes the number of hidden neurons

        Variables to cache:
        - dW: gradients wrt W
        - db: gradients wrt b

        Returns:
        - dX: gradients wrt intput X, shape (N, D)
        """

        # NOTE: You're trying to fetch the outputs from self.cache with name 'X',
        # so in the previous function, you should store your outputs with the name 'X'
        X = self.cache['X']
        W, b = self.params['W'], self.params['b']

        dX = np.zeros_like(X)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)

        from .layer_funcs import affine_backward

        ############################################################################
        # TODO:                                                                    #
        # - Derive the gradients wrt to X                                          #
        #   Use layer_funcs.py                                                     #
        # - Cache the gradients for optimizing in self.gradients                   #
        ############################################################################
        ############################################################################
        #                         START OF YOUR CODE                               #
        ############################################################################

        raise NotImplementedError
    
        ############################################################################
        #                          END OF YOUR CODE                                #
        ############################################################################

        return dX

    def update_layer(self, params):
        """ Update layer parameters """
        self.params = params


class DenseLayer(AffineLayer):
    """
    A dense hidden layer performs an affine transform followed by activation.
    Here we use ReLU as default activation function.

    For simplicity, we will implement the DenseLayer on top of AffineLayer.
    """

    def __init__(self, input_dim, output_dim=100, weight_scale=1e-2, activation='relu'):
        """
        Initialize weight W with random value and 
        bias b with zero.

        Inputs:
        - input_dim: (int) the number of input neurons, 
            like D or D1xD2x...xDn.
        - output_dim: (int) the number of hidden neurons 
            in this layer

        Variables to cache:
        - W: weights
        - b: bias
        """

        # Initialize the underlying affine layer
        super().__init__(input_dim, output_dim, weight_scale)

        # Specify the activation
        if activation == 'relu':
            from .layer_funcs import relu_forward, relu_backward
            self.activation_forward = relu_forward
            self.activation_backward = relu_backward
        else:
            raise ValueError('invalid activation function:', activation)

    def feedforward(self, X):
        """
        Feed forward function
        Inputs:
        - X: (float) a tensor of shape (N,D) or (N, D1, D2, ..., Dn)

        Variables to cache:
        - A: affine output
        - X: input

        Returns:
        - out: output of shape (N, output_dim)
        """

        out = np.zeros((X.shape[0], self.output_dim))

        ############################################################################
        #                             START OF YOUR CODE                           #
        ############################################################################
        ############################################################################
        # TODO:                                                                    #
        # - Affine transform forward pass: use parent class method                 #
        #   NOTE: Use super().f() to call a parent class function                  #
        # - Cache intermediate affine result                                       #
        ############################################################################

        raise NotImplementedError
        
        ############################################################################
        # TODO: Activation forward pass                                            #
        # NOTE: Use self.activation_forward function                               #
        ############################################################################

        raise NotImplementedError
        
        ############################################################################
        #                              END OF YOUR CODE                            #
        ############################################################################

        return out

    def backward(self, dout):
        """
        Backpropagation function
        Inputs:
        - dout: (float) a tensor with shape (N, hidden_dim)

        Variables to cache:
        - dW: gradients wrt W
        - db: gradients wrt b

        Returns:
        - dX: gradients wrt intput X, shape (N, D)
        """

        X = self.cache['X']  # cached input data
        A = self.cache['A']  # cached intermediate affine result

        dA = np.zeros_like(A)
        dX = np.zeros_like(X)

        ############################################################################
        #                             START OF YOUR CODE                           #
        ############################################################################
        ############################################################################
        # TODO:                                                                    #
        # - Activation backward                                                    #
        #   remember to use functions in class                                     #
        ############################################################################

        raise NotImplementedError
        
        ############################################################################
        # TODO:                                                                    #
        # - Affine backward                                                        #
        #   remember to use parent class functions                                 #
        ############################################################################

        raise NotImplementedError
        
        ############################################################################
        #                              END OF YOUR CODE                            #
        ############################################################################

        return dX

