import numpy as np

class Regularisation:
    def __init__(self, regularisation_type = 'L2', gamma = 0.5):
        self.regularisation_type = regularisation_type
        self.gamma = gamma
        self.reg_loss = 0.0
    
    def accumulateRegLoss(self, parameter):
        pass

    def regLossGradient(self, layer, parameter):
        pass
    
    def resetRegLoss(self):
        self.reg_loss = 0.0

class L1(Regularisation):
    def __init__(self, gamma):
        super().__init__('l1',gamma)
    
    def accumulateRegLoss(self, parameter):
        self.reg_loss = self.reg_loss + self.gamma*np.nansum(parameter)
        
    def regLossGradient(self, layer, parameter):
        layer.w_delta = layer.w_delta - 1*(self.gamma/parameter.shape[-1])*np.ones(parameter.shape)
    
    def resetRegLoss(self):
        self.reg_loss = 0.0
        
class L2(Regularisation):
    def __init__(self, gamma):
        super().__init__('l2',gamma)
    
    def accumulateRegLoss(self, parameter):
        self.reg_loss = self.reg_loss + self.gamma*np.nansum(np.power(parameter, 2))
        
    def regLossGradient(self, layer, parameter):
        layer.w_delta = layer.w_delta - (self.gamma/parameter.shape[-1])*parameter

    def resetRegLoss(self):
        self.reg_loss = 0.0

class KLdivergence(Regularisation):
    def __init__(self, gamma):
        super().__init__('kldivergence',gamma)
    
    def accumulateRegLoss(self, distribution_a, distribution_b):
        self.reg_loss = self.gamma*np.sum((distribution_a*np.log(np.divide(distribution_a/distribution_b)) + (1-distribution_a)*np.log(np.divide((1-distribution_a)/(1-distribution_b)))))
        
    def regLossGradient(self, distribution_a, distribution_b):
        self.reg_derivative = self.gamma*(-1*distribution_a/(distribution_b)) + ((1-distribution_a)/(1-distribution_b))

    def resetRegLoss(self):
        self.reg_loss = 0.0
        