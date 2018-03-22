import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import normalize

def svm_gradient(w, b, x, y, C):
    """
    Compute gradient for SVM w.r.t. to the parameters w and b on a mini-batch (x, y)

    Args:
        w: Parameters of shape [num_features]
        b: Bias (a scalar)
        x: A mini-batch of training example [k, num_features]
        y: Labels corresponding to x of size [k]

    Returns:
        grad_w: The gradient of the SVM objective w.r.t. w of shape [k, num_features]
        grad_b: The gradient of the SVM objective w.r.t. b of shape [k, 1]

    """

    grad_w = 0 # gradient
    grad_b = 0 # loss        
    

    #######################################################################
    # TODO:                                                               #
    # Compute the gradient for a particular choice of w and b.            #
    # Compute the partial derivatives and set grad_w and grad_b to the    #
    # partial derivatives of the cost w.r.t. both parameters              #
    #                                                                     #
    #######################################################################
            
    lambda_  = 1/C        
    N = x.size      
    normw = np.linalg.norm(w)   
    w_t = np.transpose(w)
    x_t = np.transpose(x)
    Xdotw = np.dot(x, w)+b
    f_x = np.add(Xdotw, b) #scores                       
    var = np.multiply(f_x, y)      
    var_ = np.transpose((np.multiply(-y, x_t)))
    threshold  = 1.
    preds = np.where(var >= threshold,1,1)    
    var_[preds] = np.zeros(var_.shape[1])   
    grad_w = w + C * var_.sum(axis=0)
        
    var2 = -y
    var2[preds] = 0
    grad_b = C * var2.sum()
    
  
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return grad_w, grad_b