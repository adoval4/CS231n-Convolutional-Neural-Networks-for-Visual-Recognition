import numpy as np
import math
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

  num_samples = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_samples):
    scores = X[i].dot(W)

    probs = np.exp(scores)
    probs /= np.sum(probs)

    sum_exps = 0
    for j in xrange(num_classes):
      dW[:, j] += probs[j] * X[i]

      if j == y[i]:
        loss -= scores[j]
        dW[:, j] -= 1 * X[i]

      sum_exps += math.exp(scores[j])

    loss += math.log(sum_exps)

  loss /= num_samples
  dW /= num_samples

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_samples = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  probs = np.exp(scores)
  probs /= np.sum(probs, axis=1, keepdims=True)

  # print(probs.shape)
  # print(probs[range(num_samples), y].shape)

  corect_logprobs = -np.log(probs[range(num_samples),y])
  loss = np.sum(corect_logprobs)
  loss /= num_samples
  loss += 0.5 * reg * np.sum(W * W)

  dscores = probs
  dscores[range(num_samples), y] -= 1 

  dW = X.T.dot(dscores)
  dW /= num_samples
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

