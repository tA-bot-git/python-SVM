import numpy as np


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
        grad_v: The gradient of the SVM objective w.r.t. b of shape [k, 1]

    """

    grad_w = 0
    grad_b = 0


    #######################################################################
    # TODO:                                                               #
    # Compute the gradient for a particular choice of w and b.            #
    # Compute the partial derivatives and set grad_w and grad_b to the    #
    # partial derivatives of the cost w.r.t. both parameters              #
    #                                                                     #
    #######################################################################


    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return grad_w, grad_b