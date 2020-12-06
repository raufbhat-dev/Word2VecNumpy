class Adam:
    def __init__(self,RMSProp = False, learning_rate = 0.01, **kwargs):
        if RMSProp:
            self.optimiser_func = 'adam'
            self.beat = kwargs['beta']
            self.epsilon = kwargs['epsilon']
            self.learning_rate = learning_rate
        else:
            self.optimiser_func = 'nadam'
    
    def __call__(self,gradient):
        if self.optimiser_func.lower() == 'adam':
            self.running_grad = self.running_grad + np.matmul(gradient,gradient.T)
        elif self.optimiser_func == 'nadam':
            self.running_grad = self.beta*self.running_grad + np.matmul(gradient,gradient.T)
        update = -1*learning_rate*np.multiply(gradient,np.linalg.pinv(np.diag(self.running_grad)))
        return update         
