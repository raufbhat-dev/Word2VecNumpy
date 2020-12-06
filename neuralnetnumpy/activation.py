import numpy as np

class PassThrough:
    def __init__(self, **kwargs):
        self.activation = None
    
    def __call__(self, y):
        self.activation_derivative = np.ones(y.shape)
        return y

class Sigmoid:
    def __init__(self, **kwargs):
        self.activation = 'sigmoid'
        self.smooth = kwargs.get('beta',1)

    def __call__(self, y):
        y_ret = 1/(1+np.exp(-self.smooth*y))
        self.activation_derivative = np.multiply(y_ret, np.ones(y_ret.shape[-1])- y_ret)
        return y_ret

class Relu():
    def __init__(self, activation, **kwargs):
        self.activation = activation
        self.alpha = kwargs.get('alpha', 0.03)

    def __call__(self,y):
        if self.activation.lower() == 'relu':
            y_ret = np.where(y<0, 0, y)
            self.activation_derivative = np.matrix(np.where(y>0, 1, 0))
            return y_ret
        elif self.activation.lower() == 'leakyrelu':
            y_ret = np.where(y > 0, y, y*self.alpha) 
            self.activation_derivative = np.matrix(np.where(y>0, 1, self.alpha))
            return y_ret
        elif self.activation.lower() == 'elu':
            y_ret = np.where(y > 0, y, self.alpha*np.exp(y)) 
            self.activation_derivative = np.matrix(np.where(y>0, 1, self.alpha*np.exp(y)))
            return y_ret

class Softmax:
    def __init__(self):
        self.activation = 'softmax'

    def __call__(self, y):
        exps = np.exp(y)
        if y.shape[1]>1: 
            y_ret = exps/np.sum(exps,axis=1)[:,None]
        else:
            y_ret = exps/np.sum(exps)
        if y.shape[1]>1:
            softmax_i_i = np.eye(y_ret.shape[1]) * y_ret[:,np.newaxis,:]
            softmax_der = -np.einsum('ki,kj->kij', y_ret, y_ret) + softmax_i_i
        else:
            softmax_der = -np.outer(y_ret, y_ret) + np.diag(y_ret.flatten())
        self.activation_derivative = softmax_der
        return y_ret

class Softplus:
    def __init__(self):
        self.activation = 'softplus'

    def __call__(self, y):
        y_ret = np.log(1 + (np.exp(y)))
        self.activation_derivative = 1 / (1 + np.exp(-y))
        return y_ret

class Tanh:
    def __init__(self):
        self.activation = 'tanh'

    def __call__(self, y):
        y_ret = np.tanh(y)
        self.activation_derivative = (1 - np.power(y_ret, 2))
        return y_ret
        