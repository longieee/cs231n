from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    for i in range(num_train):
        f = X[i] @ W  # shape (1,C)
        f -= -np.max(f)  # shift the values of f so that the highest number is 0
        softmax = np.exp(f) / np.sum(np.exp(f))  # shape (1,C)
        loss -= np.log(softmax[y[i]])  # cross-entropy loss

        # Adjust softmax for gradient calculation
        softmax[y[i]] -= 1
        dW += np.outer(X[i], softmax)

    # Regularization
    loss = loss / num_train + reg * np.sum(W**2)
    dW = dW / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]

    Z = X @ W  # shape (N,C)
    softmax = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Stabilization
    A = softmax / softmax.sum(axis=1, keepdims=True)  # Shape (N,C)
    loss = -np.log(A[range(N), y]).sum()

    # Adjust softmax for gradient calculation
    A[range(N), y] -= 1

    # Regularization
    loss = loss / N + reg * np.sum(W**2)
    dW = X.T @ A / N + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
