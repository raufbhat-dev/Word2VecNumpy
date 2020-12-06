import numpy as np

class AdaGrad:
    def __init__(self, learning_rate = 0.01):
        self.optimiser_func = 'AdaGrad'
        self.learning_rate = learning_rate
        self.epsilon = 1e-9
    
    def __call__(self, layer, upstream_gradient_w, upstream_gradient_b):
        gradient_tensor_weight = np.einsum('ki,kj->kij', upstream_gradient_w.T, upstream_gradient_w.T)
        gradient_tensor_diag_weight = np.einsum('ijj->ij', gradient_tensor_weight).T
        
        layer.w_parameter  = layer.w_parameter + gradient_tensor_diag_weight             
        new_learning_rate_weights = self.learning_rate*(layer.w_parameter+self.epsilon)**-0.5

        gradient_tensor_bias = np.outer(upstream_gradient_b, upstream_gradient_b)
        gradient_tensor_diag_bias = np.diag(gradient_tensor_bias)
        
        layer.b_parameter  = layer.b_parameter + gradient_tensor_diag_bias             
        new_learning_rate_bias = self.learning_rate*np.power(layer.b_parameter+self.epsilon,-1/2)
        
        layer.w_delta = -1*np.multiply(new_learning_rate_weights,upstream_gradient_w) 
        layer.b_delta = -1*np.multiply(new_learning_rate_bias,upstream_gradient_b) 
        
class RMSprop(AdaGrad):
    def __init__(self, learning_rate = 0.01, beta = 0.9):
        super().__init__(learning_rate)
        self.optimiser_func = 'RMSprop'
        self.beta = beta

    def __call__(self, layer, upstream_gradient_w, upstream_gradient_b):
        gradient_tensor_weight = np.einsum('ki,kj->kij', upstream_gradient_w.T, upstream_gradient_w.T)
        gradient_tensor_diag_weight = np.einsum('ijj->ij', gradient_tensor_weight).T

        layer.w_parameter  = self.beta*layer.w_parameter + gradient_tensor_diag_weight             
        new_learning_rate_weights = self.learning_rate*(layer.w_parameter+self.epsilon)**-0.5

        gradient_tensor_bias = np.outer(upstream_gradient_b, upstream_gradient_b)
        gradient_tensor_diag_bias = np.diag(gradient_tensor_bias)
        
        layer.b_parameter  = self.beta*layer.b_parameter + gradient_tensor_diag_bias             
        new_learning_rate_bias = self.learning_rate*np.power(layer.b_parameter+self.epsilon,-1/2)
        
        layer.w_delta = -1*np.multiply(new_learning_rate_weights,upstream_gradient_w) 
        layer.b_delta = -1*np.multiply(new_learning_rate_bias,upstream_gradient_b) 
