import numpy as np
import importlib

from neuralnetnumpy import neuralnet

NeuralNet = importlib.reload(neuralnet)

class Cbow(NeuralNet.NeuralNet):
    def __init__(self, network_arch, loss_method, optimiser_method, epoch, batch_size, frame_size, mode, **kwargs):
        super().__init__(network_arch, loss_method, optimiser_method, epoch, batch_size, mode, **kwargs)
        self.frame_size = frame_size

    def train(self, inputs, vocabulary):
        for i in range(self.epoch_count):
            batch_index = self.batch_size + self.frame_size-1
            start_index =0
            batch_count = int(np.floor((len(inputs)-self.frame_size)/self.batch_size))
            for index in range(batch_count):
                input_data_matrix, output_data_matrix, start_index = self.generateTrainingData(inputs, vocabulary, start_index, batch_index)
                input_data_matrix = input_data_matrix.reshape(self.batch_size, input_data_matrix.shape[0])
                output_data_matrix = output_data_matrix.reshape(self.batch_size, output_data_matrix.shape[0])
                self.regularisation.resetRegLoss()
                self.forwardPass(input_data_matrix, output_data_matrix)
                self.backProp(input_data_matrix)
                print('Epoch:{}  Batch: {}/{}  Reg Loss: {:.2f}  Loss : {:.2f}'.format(i+1, index+1, batch_count, self.regularisation.reg_loss, self.loss.loss), end='\r', flush=True)
            print("",end="\n")
    
    def generateTrainingData(self, inputs, vocabulary, start_index, batch_index):
        inp_batch = inputs[start_index: start_index+batch_index]
        input_data_matrix = output_data_matrix = None
        for word_index in range(self.batch_size):
            start_index = start_index + 1  
            output_word_index = word_index + int(np.floor(self.frame_size/2))
            input_vector = output_vector = np.zeros((len(vocabulary)))
            output_vector[vocabulary.index(inp_batch[output_word_index])] = 1 
            if output_data_matrix is None:
                output_data_matrix = output_vector
            else:
                output_data_matrix = np.vstack((output_data_matrix, output_vector))
            for contex_index, context_word in enumerate(inp_batch[word_index: word_index+self.frame_size]):
                if not contex_index == int(np.floor(self.frame_size/2)):
                    input_vector[vocabulary.index(context_word)] = 1
            if input_data_matrix is None:
                input_data_matrix = input_vector
            else:
                input_data_matrix = np.vstack((input_data_matrix, input_vector))
        return input_data_matrix, output_data_matrix, start_index
                   
    def getSimilarWords(self, word, vocabulary, number_of_words):
        weights = np.copy(self.layers_list[0].w + self.layers_list[1].w.T)
        word_index = vocabulary.index(word)
        squared_difference = (weights - weights[word_index])**2
        squared_distance = np.nansum(squared_difference, axis=1)
        min_word_index = squared_distance.argsort()[:number_of_words]
        return [vocabulary[index] for index in min_word_index]
    
