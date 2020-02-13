# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:07:48 2020

@author: 이충섭
"""

import hmm
from hmm import Model

states = ('rainy', 'sunny')
symbols = ('walk', 'shop', 'clean')

start_prob = {
    'rainy' : 0.5,
    'sunny' : 0.5
}

trans_prob = {
    'rainy': { 'rainy' : 0.7, 'sunny' : 0.3 },
    'sunny': { 'rainy' : 0.4, 'sunny' : 0.6 }
}

emit_prob = {
    'rainy': { 'walk' : 0.1, 'shop' : 0.4, 'clean' : 0.5 },
    'sunny': { 'walk' : 0.6, 'shop' : 0.3, 'clean' : 0.1 }
}

sequence = ['walk', 'shop', 'clean', 'clean', 'walk', 'walk', 'walk', 'clean']
sequence2 = ['walk', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk']
model = Model(states, symbols, start_prob, trans_prob, emit_prob)

print(model.evaluate(sequence))

for i in range(0, len(sequence)) :
     print(i)
     print(model._forward(sequence)[i]['sunny'] * model._backward(sequence)[i]['sunny'] + model._forward(sequence)[i]['rainy'] * model._backward(sequence)[i]['rainy'])

            
print(model.decode(sequence))
print(model.decode(sequence2))

print(model._forward(sequence))

# from tensorflow.keras.datasets import mnist
# import numpy as np
# (train_x, train_y), (test_x, test_y) = mnist.load_data()

# X = train_x[0:100]

# X = np.reshape(X, (-1, 28*28))
# Y = train_y[0:100]

# Z = []

# for y in Y :
#     Z.append(np.full((784), y))
    
# Y = Z

# Z = list(zip(X, Y))

# model = hmm.train(Z)