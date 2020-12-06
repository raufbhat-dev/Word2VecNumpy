import numpy as np

class Loss:
    def __init__(self, loss_func, clip_grad, norm_grad, epsilon):
        self.loss_func = loss_func
        self.clip_grad = clip_grad
        self.norm_grad = norm_grad
        self.loss =  0.0
        self.epsilon = epsilon

class MeanSquared(Loss):
    def __init__(self, clip_grad = True, norm_grad = True):
        self.epsilon = 1e4
        super().__init__('meansquared', clip_grad, norm_grad, self.epsilon)
    
    def getLoss(self, y_pred, y):
        error =  y - y_pred
        self.loss = np.nansum(np.diag(np.matmul(error,error.T))) / y.shape[-1]
        loss_grad = -1*error
        if self.clip_grad:
            loss_grad = np.clip(loss_grad, -1*self.epsilon, self.epsilon)
        
        if self.norm_grad:
            loss_grad_max = np.max(np.abs(loss_grad),axis=1)
            loss_grad = loss_grad/loss_grad_max[:,None]

        self.loss_derivative = loss_grad
            

class BinaryCrossEntropy(Loss):
    def __init__(self, clip_grad = True, norm_grad = True):
        self.theta = 1e-9
        self.epsilon = 1e4
        super().__init__('crossentropy', clip_grad, norm_grad, self.epsilon)
    
    def getLoss(self, y_pred, y):
        self.loss = -1*(np.sum(np.multiply(y,np.log(y_pred)) + np.multiply((1-y),np.log(1- y_pred))))/y_pred.shape[0]
        loss_grad = -1*np.divide(y,y_pred)+ np.divide((1-y),(1 - y_pred)) 
        if self.clip_grad:
            loss_grad = np.clip(loss_grad, -1*self.epsilon, self.epsilon)
        
        if self.norm_grad:
            loss_grad_max = np.max(np.abs(loss_grad),axis=1)
            loss_grad = loss_grad/loss_grad_max[:,None]
        self.loss_derivative = loss_grad
