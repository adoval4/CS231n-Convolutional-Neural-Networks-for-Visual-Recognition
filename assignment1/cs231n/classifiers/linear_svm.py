# -*- coding: utf-8 -*-

import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    indicator = (scores - correct_class_score + 1) > 0
    # print(indicator)
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] -= np.sum(np.delete(indicator,j)) * X[i].T
        continue

      dW[:, j] += indicator[j] * X[i].T
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin



  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_train = X.shape[0]

  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train), y]

  # hinge loss
  margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1) 

  # make the margins for correct classes equal to 0 cause they are not considered in the final sum 
  margins[np.arange(num_train), y] = 0

  # sum all margins
  loss = np.sum(margins)

  # get mean
  loss /= num_train

  # add regularization
  loss += 0.5 * reg * np.sum(W * W)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  indicators = np.zeros(margins.shape)
  indicators[margins > 0] = 1

  # for each samples (column)
  # count the number of classes that didn’t meet the desired margin ( margin > 0 or corect_class_score - wrong_class_score < delta )
  num_classes_not_enough_margin = np.sum(indicators, axis=1)

  # set minus sum of mistaken class scores on indactors values that will multiply the correct class weigth
  indicators[np.arange(num_train), y] = - num_classes_not_enough_margin  

  dW = X.T.dot(indicators)

  # get mean
  dW /= num_train

  # add normalization gradient
  dW += reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
