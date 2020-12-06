class StochasticGradientDescent:
    def __init__(self, learning_rate = 0.01):
        self.optimiser_func = 'StochasticGradientDescent'
        self.learning_rate = learning_rate

    def __call__(self, layer, upstream_gradient_w, upstream_gradient_b):
        layer.w_delta = -1*self.learning_rate*upstream_gradient_w 
        layer.b_delta = -1*self.learning_rate*upstream_gradient_b

class Momentum(StochasticGradientDescent):
    def __init__(self, learning_rate = 0.01, beta = 0.9):
        super().__init__(learning_rate)
        self.optimiser_func = 'Momentum'
        self.beta = beta
    
    def __call__(self, layer, upstream_gradient_w, upstream_gradient_b):
        layer.w_delta =  self.beta*layer.w_delta -1*self.learning_rate*upstream_gradient_w
        layer.b_delta = self.beta*layer.b_delta -1*self.learning_rate*upstream_gradient_b
