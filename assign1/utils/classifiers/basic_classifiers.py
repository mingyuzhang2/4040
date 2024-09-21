"""
Implementation of basic classifiers. 
"""

import numpy as np


class BasicClassifier:
    """ A basic classfier """

    def __init__(self):
        self.W = None
        self.velocity = None

    def w_init(self, dim, num_classes):
        """
        Model weights initializer.
        Subclasses will override this, so no content needed for this function.

        Inputs:
        - dim: the dimension of the model
        - num_classes: the total number of classes

        Returns a numpy array:
        - W: containing model weights of shape 
            (D,) for logistic regression
            (D, C) for softmax classifier
        """
        raise NotImplementedError

    def train(
        self, X, y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        optim="SGD",
        momentum=0.5,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent (SGD).
        Batch size is set to 200, learning rate to 0.001, regularization rate to 0.00001.

        Inputs:
        - X: a numpy array of shape (N, D) containing training data; there are N
            training samples each of dimension D.
        - y: a numpy array of shape (N,) containing training labels; y[i] = c
            means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) L2 regularization strength.
        - num_iters: (integer) number of steps to take when optimizing.
        - batch_size: (integer) number of training examples to use at each step.
        - optim: the optimization method, the default optimizer is 'SGD' and
            feel free to add other optimizers.
        - verbose: (boolean) if true, print progress during optimization.

        Returns:
        - loss_history: a list containing the value of the loss function of each iteration.
        """

        num_train, dim = X.shape
        # assume y takes values 0...K-1 where K is the number of classes
        num_classes = np.max(y) + 1

        # Initialize W and velocity(for SGD with momentum)
        if self.W is None:
            self.W = 0.001 * self.w_init(dim, num_classes)

        if self.velocity is None:
            self.velocity = np.zeros_like(self.W)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):

            ########################################################################
            # TODO:                                                                #
            # Sample batch_size elements from the training data and their          #
            # corresponding labels to use in this round of gradient descent.       #
            # Store the data in X_batch and their corresponding labels in          #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)  #
            # and y_batch should have shape (batch_size,)                          #
            #                                                                      #
            # Hint: Use np.random.choice to generate indices. Sometimes, random    #
            # choice will be better than training in order.                        #
            ########################################################################
            ########################################################################
            #                     START OF YOUR CODE                               #
            ########################################################################

            raise NotImplementedError
        
            ########################################################################
            #                       END OF YOUR CODE                               #
            ########################################################################

            # compute the loss and gradients
            loss, grad = self.loss(X_batch, y_batch, reg)
            # log the training process
            loss_history.append(loss)

            ########################################################################
            # TODO:                                                                #
            # Update the weights using the gradient and the learning rate.         #
            ########################################################################
            ########################################################################
            #                     START OF YOUR CODE                               #
            ########################################################################

            raise NotImplementedError
        
            ########################################################################
            #                    END OF YOUR CODE                                  #
            ########################################################################

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Subclasses will override this, so no content needed for this function.

        Inputs:
        - X: a numpy array of shape (N, D) containing training data; there are N
            training samples each of dimension D.

        Returns:
        - y_pred: predicted labels for the data in X. y_pred is a 1-dimensional
            array of length N, and each element is an integer giving the predicted
            class.
        """
        raise NotImplementedError

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this, so no content needed for this function.

        Inputs:
        - X_batch: a numpy array of shape (N, D) containing a minibatch of N
            data points; each point has dimension D.
        - y_batch: a numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns:
        - loss:  a single float
        - gradient:  gradients wst W, an array of the same shape as W
        """
        raise NotImplementedError


class LogisticRegression(BasicClassifier):
    """ A subclass that uses the Logistic Regression loss function """

    def loss(self, X_batch, y_batch, reg):
        from .logistic_regression import logistic_regression_loss_vectorized
        return logistic_regression_loss_vectorized(self.W, X_batch, y_batch, reg)

    def w_init(self, dim, num_classes):
        return np.random.randn(dim)

    def predict(self, X):

        y_pred = np.zeros(X.shape[0])

        ########################################################################
        # TODO:                                                                #
        # Implement this method. Store the predicted labels in y_pred.         #
        # NOTE:                                                                #
        # To make your life easier, consider using np.where                    #
        # https://numpy.org/doc/stable/reference/generated/numpy.where.html    #
        ########################################################################
        ########################################################################
        #                     START OF YOUR CODE                               #
        ########################################################################

        raise NotImplementedError
    
        ########################################################################
        #                    END OF YOUR CODE                                  #
        ########################################################################

        return y_pred


class Softmax(BasicClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        from .softmax import softmax_loss_vectorized
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

    def w_init(self, dim, num_classes):
        return np.random.randn(dim, num_classes)

    def predict(self, X):

        y_pred = np.zeros(X.shape[0])

        ########################################################################
        # TODO:                                                                #
        # Implement this method. Store the predicted labels in y_pred.         #
        # NOTE:                                                                #
        # Think about which np function is suitable here.                      #
        ########################################################################
        ########################################################################
        #                     START OF YOUR CODE                               #
        ########################################################################

        raise NotImplementedError
    
        ########################################################################
        #                    END OF YOUR CODE                                  #
        ########################################################################

        return y_pred
