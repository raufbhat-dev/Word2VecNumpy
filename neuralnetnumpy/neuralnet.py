import importlib
import numpy as np

from neuralnetnumpy  import layer, regularisation 

Layer = importlib.reload(layer)
Regularisation = importlib.reload(regularisation)

class NeuralNet:
    def __init__(self, network_arch, loss_method, optimiser_method, epoch, batch_size, mode, **kwargs):
        self.loss = loss_method
        self.optimiser = optimiser_method
        self.regularisation = kwargs.get('regularisation_method',Regularisation.Regularisation())
        self.epoch_count = epoch
        self.batch_size = batch_size 
        self.mode = mode
        self.layers_list = []
        self.loss  = loss_method
        self.network_arch = network_arch
                    
    def createNetwork(self):
        network_layers = []
        for index, _layer in  enumerate(self.network_arch):
                if _layer.get('layer_type') != 'input':
                    if _layer.get('dropout', 0):
                        self.layers_list.append(layer.Dropout(self.network_arch[index-1].get('size',2), _layer.get('size',2), _layer.get('activation','sigmoid') , _layer.get('layer_type'), _layer.get('dropout')))
                    else:
                        self.layers_list.append(layer.Layer(self.network_arch[index-1].get('size',2), _layer.get('size',2), _layer.get('activation','sigmoid') , _layer.get('layer_type')))

    def forwardPass(self, inputs, outputs = None):
        layer_out = inputs
        drop_index = []
        for _layer in self.layers_list:
            layer_out = _layer(inputs)
            inputs = layer_out
            self.regularisation.accumulateRegLoss(_layer.w)
        if self.mode.lower() == 'train':
            self.loss.getLoss(layer_out,outputs)
        elif self.mode.lower() == 'test':
            return layer_out
    
    def backProp(self, inputs):
        upstream_gradient = self.loss.loss_derivative
        for index, _layer in enumerate(reversed(self.layers_list)):
            if _layer.layer_type == 'output':
                if _layer.activation_method.activation == 'softmax':
                    if inputs.shape[1]>1: 
                        upstream_gradient =  np.einsum('ij,ijk->ik', upstream_gradient, _layer.activation_derivative)
                    else:
                        upstream_gradient =  np.matmul(upstream_gradient, _layer.activation_derivative)
                else:
                    upstream_gradient = np.multiply(upstream_gradient,_layer.activation_derivative)
                upstream_gradient_w =  np.matmul(self.layers_list[len(self.layers_list)-2].y_activation.T, upstream_gradient) 
            if _layer.layer_type == 'hidden':
                upstream_gradient =  np.matmul(upstream_gradient, self.layers_list[len(self.layers_list) -index].w.T)
                upstream_gradient = np.multiply(upstream_gradient,_layer.activation_derivative)
                if (len(self.layers_list)-index-1) != 0:
                    upstream_gradient_w = np.matmul(self.layers_list[len(self.layers_list) -index -2].y_activation.T, upstream_gradient)
                else:
                    upstream_gradient_w = np.matmul(inputs.T, upstream_gradient)
            upstream_gradient_b = np.sum(upstream_gradient,axis=0)
            self.optimiser(_layer, upstream_gradient_w, upstream_gradient_b)
            weights = _layer.getWeightBias()[0]
            if self.layers_list[len(self.layers_list) -index -2].dropout_layer:
                drop_index = self.layers_list[len(self.layers_list) -index -2].drop_index
                weights[drop_index,] = 0
            self.regularisation.regLossGradient(_layer, weights)
    
        for _layer_ in self.layers_list:
            w_delta, b_delta = _layer_.getLocalGradient()
            _layer_.w = _layer_.w + w_delta
            _layer_.b = _layer_.b + b_delta
    
    def train(self, inputs, outputs):
        for i in range(self.epoch_count):
            batch_count = int(np.ceil(len(outputs)/self.batch_size))
            for index in range(int(np.ceil(len(outputs)/self.batch_size))):
                inp_batch = inputs[self.batch_size*index:self.batch_size*index+self.batch_size,]
                out_batch = outputs[self.batch_size*index:self.batch_size*index+self.batch_size,]
                inp_batch = inp_batch.reshape(self.batch_size, inp_batch.shape[0])
                out_batch = out_batch.reshape(self.batch_size, out_batch.shape[0])
                self.regularisation.resetRegLoss()
                self.forwardPass(inp_batch, out_batch)
                self.backProp(inp_batch)
                print('Epoch:{}  Batch: {}/{}  Reg Loss: {:.3f}  Loss : {:.3f}'.format(i+1, index+1, batch_count, self.regularisation.reg_loss, self.loss.loss), end='\r', flush=True)
            print("",end="\n")
        
        