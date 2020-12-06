import numpy as np
import importlib

from word2vecnumpy import skipgram

SkipGram = importlib.reload(skipgram)

class NegativeSampling(SkipGram.SkipGram):
    def __init__(self, network_arch, loss_method, optimiser_method, epoch, batch_size, frame_size, number_of_negative_samples, mode, **kwargs):
        super().__init__(network_arch, loss_method, optimiser_method, epoch, batch_size, frame_size, mode, **kwargs)
        self.number_of_negative_samples = number_of_negative_samples
        
    def generateNegativeSample(self, word_freq, vocabulary):
        normalised_word_freq = np.power(word_freq/np.sum(word_freq), 0.75)
        normalised_word_freq = normalised_word_freq/sum(normalised_word_freq)
        negative_sample_bag_of_index = np.random.choice(len(vocabulary), self.number_of_negative_samples, p = normalised_word_freq)
        negative_sample_bag_of_words = [word for index, word in enumerate(vocabulary) if index in negative_sample_bag_of_index]
        return negative_sample_bag_of_words, negative_sample_bag_of_index

    def generateTrainingData(self, inputs, vocabulary, start_index, batch_index):
        inp_batch = inputs[start_index: start_index+batch_index]
        input_data_matrix = output_data_matrix = None
        for word_index in range(self.batch_size):
            start_index = start_index + 1  
            input_word_index = word_index + int(np.floor(self.frame_size/2))
            input_vector = output_vector = np.zeros((len(vocabulary)))
            input_vector[vocabulary.index(inp_batch[input_word_index])] = 1 
            for contex_index, context_word in enumerate(inp_batch[word_index: word_index+self.frame_size]):
                if not contex_index == int(np.floor(self.frame_size/2)):
                    output_vector[vocabulary.index(context_word)] = 1
            if output_data_matrix is None:
                output_data_matrix = output_vector
            else:
                output_data_matrix = np.vstack((output_data_matrix, output_vector))
            if input_data_matrix is None:
                input_data_matrix = input_vector
            else:
                input_data_matrix = np.vstack((input_data_matrix, input_vector))
        return input_data_matrix, output_data_matrix, start_index     

    def backProp(self, inputs, word_freq, vocabulary):
        upstream_gradient = self.loss.loss_derivative
        for row_index in range(upstream_gradient.shape[0]):
            negative_sample_bag_of_index = self.generateNegativeSample(word_freq, vocabulary)[1]
            remove_gradient_index = [index for index in range(upstream_gradient.shape[1]) if index not in negative_sample_bag_of_index]
            upstream_gradient[row_index, remove_gradient_index] = 0
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
    
    def train(self, inputs, vocabulary, word_freq):
        for i in range(self.epoch_count):
            batch_index = self.batch_size + self.frame_size-1
            start_index =0
            batch_count = int(np.floor((len(inputs)-self.frame_size)/self.batch_size))
            for index in range(batch_count):
                input_data_matrix, output_data_matrix, start_index = self.generateTrainingData(inputs, vocabulary, start_index, batch_index)
                self.regularisation.resetRegLoss()
                input_data_matrix = input_data_matrix.reshape(self.batch_size, input_data_matrix.shape[0])
                output_data_matrix = output_data_matrix.reshape(self.batch_size, output_data_matrix.shape[0])
                self.forwardPass(input_data_matrix, output_data_matrix)
                self.backProp(input_data_matrix, word_freq, vocabulary)
                print('Epoch:{}  Batch: {}/{}  Reg Loss: {:.2f}  Loss : {:.2f}'.format(i+1, index+1, batch_count, self.regularisation.reg_loss, self.loss.loss), end='\r', flush=True)
            print("",end="\n")
        
