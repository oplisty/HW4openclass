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
    - W: A numpy array of shape (D, num_class) containing weights.
    - X: A numpy array of shape (num_train, vector_length) containing a minibatch of data.
    - y: A numpy array of shape (num_train,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)# (1,vector_length) *(vecntor_length,numclass)-> (1,num_class)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        for c in range(num_classes):
            if c == y[i]:
                dW[:, c] += (p[c] - 1.0) * X[i]
            else:
                dW[:, c] += p[c] * X[i]

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW= dW/num_train +2 *reg *W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train=X.shape[0]
    score=X.dot(W) #numtrain,numclass
    score -= np.max(score, axis=1, keepdims=True)
    prob=-np.log(np.exp(score)/np.sum(np.exp(score),axis=1,keepdims=True))
    loss=np.sum(prob[np.arange(num_train),y])
    loss=loss/num_train+ reg *np.sum(W*W)
    
    dprob_score=np.exp(score)/np.sum(np.exp(score),axis=1,keepdims=True)
    dprob_score[np.arange(num_train),y]-=1
    dW=X.T @ dprob_score
    dW=dW/num_train+2 *reg * W 


    return loss, dW
