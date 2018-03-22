import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import normalize

def svm_loss(w, b, X, y, C):
    """
    Computes the loss of a linear SVM w.r.t. the given data and parameters

    Args:
        w: Parameters of shape [num_features]
        b: Bias (a scalar)
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]
        C: SVM hyper-parameter

    Returns:
        l: The value of the objective for a linear SVM

    """

    l = 0
    #######################################################################
    # TODO:                                                               #
    # Compute and return the value of the unconstrained SVM objective     #
    #                                                                     #
    #######################################################################
    
    #var_1 frac{\lambda}{2}||w||^2 
    lambda_ = (1/C) / 2      
    normwSq = np.square(np.linalg.norm(w, ord=2))   
               
    w_t = np.transpose(w)
    wTranspX = np.dot(X, w_t) + b
    fx = wTranspX
    fx_t = np.transpose(fx)
    var = 1 - np.multiply(fx_t, y)      
    var2 = np.maximum(0, var)                 
    m = X.size  
    var_2 = np.sum(var2)
       
        
    first = np.dot(normwSq,lambda_)
    second = var_2/m        
    
    l =  first + second
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return l
