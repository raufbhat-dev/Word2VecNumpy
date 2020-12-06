import numpy as np
import importlib

from neuralnetnumpy  import activation

Activation = importlib.reload(activation)

class Layer:
    def __init__(self, input_count, neruon_count, activation_func, layer_type):
        self.layer_type = layer_type
        self.dropout_layer = False
        self.activation_method = getattr(Activation, activation_func)()
        self.w = np.random.randn(input_count,neruon_count) / np.sqrt(neruon_count) 
        self.w_delta = np.zeros((input_count,neruon_count))
        self.b = np.ones(neruon_count)
        self.b_delta = np.zeros(neruon_count)
        self.y = np.random.uniform(low = 0, high = 1, size = (neruon_count, 1))
        self.y_activation = np.random.uniform(low = 0, high = 1, size = (neruon_count, 1))
        self.activation_derivative = np.random.uniform(low = 0, high = 1, size = (neruon_count, 1))
        self.w_parameter = np.zeros((input_count,neruon_count))  # Gradient accumulator for Adaptive learning rate
        self.b_parameter = np.zeros(neruon_count) # Gradient accumulator for Adaptive learning rate
  
    def __call__(self,inputs):
        self.y = np.matmul(inputs,self.w)+ self.b
        self.y_activation = self.activation_method(self.y)
        self.activation_derivative = self.activation_method.activation_derivative
        return self.y_activation
    
    def getLocalGradient(self):
        return np.copy(self.w_delta), np.copy(self.b_delta)
    
    def getWeightBias(self):
        return np.copy(self.w), np.copy(self.b)

class Dropout(Layer):
    def __init__(self, input_count, neruon_count, activation_func, layer_type, percentage):
        super().__init__(input_count, neruon_count, activation_func, layer_type)
        self.percentage = percentage 
        self.dropout_layer = True

    def __call__(self,inputs):
        self.drop_index = np.random.choice(self.w.shape[-1], int(np.ceil(self.w.shape[-1]*self.percentage)), replace=False)
        weights, bias = self.setWeightBias(self.w, self.b)
        self.y = np.matmul(inputs,weights)+ bias
        y_activation = self.activation_method(self.y)
        y_activation.T[self.drop_index] = 0
        self.y_activation = y_activation
        activation_derivative = self.activation_method.activation_derivative
        activation_derivative.T[self.drop_index] = 0
        self.activation_derivative = activation_derivative
        return self.y_activation

    def getLocalGradient(self):
        weights_grad, bias_grad = self.setWeightBias(self.w_delta, self.b_delta)
        return weights_grad, bias_grad
    
    def setWeightBias(self,weight, bais):
        drop_node_weights = np.copy(weight)
        drop_node_bias = np.copy(bais)
        drop_node_weights.T[self.drop_index] = 0
        drop_node_bias[self.drop_index] = 0
        drop_node_weights = np.multiply(weight, drop_node_weights)
        drop_node_bias = np.multiply(bais, drop_node_bias)
        return drop_node_weights, drop_node_bias

    def getWeightBias(self):
        return self.setWeightBias(self.w, self.b)
    
    def getDropIndex(self):
        return np.copy(self.drop_index)