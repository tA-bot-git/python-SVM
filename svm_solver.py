import numpy as np
import time
import random
from random import shuffle
from svm_loss import svm_loss
from svm_gradient import svm_gradient


def svm_solver(X, y, C, num_iter=50000, num_per_batch=32):
    """
    Pegasos SVM solver.

    Args:
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train]
        C: SVM hyper-parameter
        num_iter: Number of iterations
        num_per_batch: Number of samples per mini-batch

    Returns:
        w: The learnt weights vector
        b: The learnt intercept

    """

    w = np.zeros(X.shape[1])
    b = 0.
    for t in range(1, num_iter):
        start = time.time()        
        
        # 1st Step: Sample a random mini-batch of size num_per_batch (done)                               
        samp = np.random.choice(np.arange(X.shape[0]), size = num_per_batch)
        X_ = X[samp]
        Y_ = y[samp]
        
        # 2nd Step: Compute the learning-rate n_t=1/(lambda*t) where lambda=1/C (done)            
        lambda_ = 1/C
        n_t = 1 / (lambda_ * (t+1))

        # 3rd Step: Compute the gradients and update the parameters as 
        # w:=w-n_t*grad_w and b:=b-n_t*grad_b        
        grad_w, grad_b = svm_gradient(w, b, X_, Y_, C)
        w = w - n_t * grad_w       
        b = b - n_t * grad_b
        
        # http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf        
        
        if t % 5000 == 0:
            exec_time = time.time() - start
            loss = svm_loss(w, b, X, y, C)
            print('Iter {}/{}: cost = {}  ({}s)'.format(t, num_iter, loss, exec_time))

    return w, b
