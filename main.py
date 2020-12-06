import json
import re, string
import numpy as np
import importlib
import codecs

from neuralnetnumpy.Optimiser import GradientDescent, AdaptiveLearningRate
from neuralnetnumpy  import loss, regularisation 
from word2vecnumpy import cbow, skipgram, negativesampling

#reload modules
GradientDescent = importlib.reload(GradientDescent)
AdaptiveLearningRate = importlib.reload(AdaptiveLearningRate)
Loss = importlib.reload(loss)
Regularisation = importlib.reload(regularisation)
Cbow = importlib.reload(cbow)
SkipGram = importlib.reload(skipgram)
NegativeSampling = importlib.reload(negativesampling)

RAW_DATA_PATH = '/mnt/ebs-1/rauf_bhat/git_repo_rauf/Word2VecNumpy/raw_data/HarryPotter/'
file_list = ['Harry Potter 1 - Sorcerer Stone.txt',
              'Harry Potter 2 - Chamber of Secrets.txt',
              'Harry Potter 3 - The Prisoner Of Azkaban.txt',
              'Harry Potter 4 - The Goblet Of Fire.txt',
              'Harry Potter 5 - Order of the Phoenix.txt',
              'Harry Potter 6 - The Half Blood Prince.txt',
              'Harry Potter 7 - Deathly Hollows.txt']

STOP_WORDS_FILE = 'stopwords.list'

with open(RAW_DATA_PATH+STOP_WORDS_FILE) as stop_list:
    stop_words = json.load(stop_list)
    
pattern = re.compile('[^a-zA-Z]')

context_data = []
for file in file_list:
    with codecs.open(RAW_DATA_PATH+file, 'r', encoding='utf-8', errors='ignore') as fdata:
        for line in fdata:
            words = pattern.sub(' ', line).lower().split()
            words = [w for w in words if w not in stop_words and len(w)!=1]
            context_data.extend(words)     
            
vocabulary = list(set(context_data))

vocabulary_freq = []
for word in vocabulary :
    vocabulary_freq.append(context_data.count(word))
    
new_vocabulary = [word for index, word in enumerate(vocabulary) if vocabulary_freq[index]>5]   

print('old_vocabulary',len(vocabulary))
print('new_vocabulary',len(new_vocabulary))

new_context_data = [word for word in context_data if vocabulary_freq[vocabulary.index(word)] >5]

print('old_context_data',len(context_data))
print('new_context_data',len(new_context_data))

new_vocabulary_freq = []
for word in new_vocabulary :
    new_vocabulary_freq.append(new_context_data.count(word))
    
#optimiser
learning_rate = 0.01
optimiser_method = GradientDescent.StochasticGradientDescent(learning_rate)
#optimiser_method = GradientDescent.Momentum(learning_rate,beta=0.9)
#optimiser_method = AdaptiveLearningRate.AdaGrad(learning_rate)
#optimiser_method = AdaptiveLearningRate.RMSprop(learning_rate,beta=0.9)

#loss
clip_grad = False
norm_grad = False
#loss_method = Loss.MeanSquared(clip_grad, norm_grad)
loss_method = Loss.BinaryCrossEntropy(clip_grad, norm_grad)

#Regularisation
gamma = 0.0005
regularisation_method = Regularisation.L2(gamma)

epoch = 10
batch_size = 1
frame_size = 7
mode = 'train'

network_arch = [{'layer_type':'input', 'size':len(new_vocabulary)},
                {'layer_type':'hidden', 'size':100, 'activation':'PassThrough'},
                {'layer_type':'output', 'size':len(new_vocabulary), 'activation':'Softmax'}]

word_to_vec_cbow = Cbow.Cbow(network_arch, loss_method, optimiser_method, epoch, batch_size, frame_size, mode)#, regularisation_method = regularisation_method)

word_to_vec_cbow.createNetwork()

word_to_vec_cbow.train(new_context_data, new_vocabulary)

word_to_vec_skip_gram = SkipGram.SkipGram(network_arch, loss_method, optimiser_method, epoch, batch_size, frame_size, mode)#, regularisation_method = regularisation_method)

word_to_vec_skip_gram.createNetwork()

word_to_vec_skip_gram.train(new_context_data, new_vocabulary)

number_of_negative_samples = 7*(frame_size-1)

network_arch = [{'layer_type':'input', 'size':len(new_vocabulary)},
                {'layer_type':'hidden', 'size':100, 'activation':'PassThrough'},
                {'layer_type':'output', 'size':len(new_vocabulary), 'activation':'Sigmoid'}]


word_to_vec_neg = NegativeSampling.NegativeSampling(network_arch, loss_method, optimiser_method, epoch, batch_size, frame_size, number_of_negative_samples, mode)#, regularisation_method = regularisation_method)

word_to_vec_neg.createNetwork()

word_to_vec_neg.train(new_context_data, new_vocabulary, new_vocabulary_freq)

top_x_words = 20
for word_index in np.argsort(new_vocabulary_freq)[-top_x_words:]:
    word = new_vocabulary[word_index]
    print('word:                  ', word)
    print('word_to_vec_cbow:      ',word_to_vec_cbow.getSimilarWords(word, new_vocabulary, number_of_words))
    print('word_to_vec_skip_gram: ',word_to_vec_skip_gram.getSimilarWords(word, new_vocabulary, number_of_words))
    print('word_to_vec_neg:       ',word_to_vec_neg.getSimilarWords(word, new_vocabulary, number_of_words))
