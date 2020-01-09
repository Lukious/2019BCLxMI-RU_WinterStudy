# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:49:50 2020
@author: Lukious
"""

import numpy as np
import glob
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
import os
from keras import objectives
import tensorflow as tf

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean =0, std=1.0
    epsilon = K.random_normal(shape=(batch,dim))
    return z_mean +K.exp(0.5*z_log_var)*epsilon

def main():
    
    leaky_relu = tf.nn.leaky_relu

    folder_path = './pokemon-dataset/images/MatrixPreprocessed'
    counter = 0

    for filename in os.listdir(folder_path):
        counter = counter + 1
        print(filename)
        temp = np.load(folder_path+'/'+filename)
        temp = temp.reshape((1,56,56))
        if counter == 1:
            Pokemon_data = temp
            Pokemon_data = Pokemon_data.reshape((1,56,56))
        else:
            Pokemon_data = np.concatenate((Pokemon_data, temp),axis = 0)

    print("Data Load is DONE. \nTotal Data Number : "+ str(counter))

    #(X_train,Y_train), (X_test,Y_test) = mnist.load_data()
    X_train = Pokemon_data
    row = 56
    col = 56
    dim = row * col
    X_train = np.reshape(X_train,[-1,dim]).astype('float32')/255
        
    input_shape = (dim,)
    intermediate_dim = 512
    batch_size = 64
    latent_dim = 2 # mean and standard deviation!
    epochs = 128
    
    # VAE model = autoencoder (encoder + decoder)
    inputs = Input(shape=input_shape,name='encoder_input')
    # train q(z|x) -> approximation
    x = Dense(intermediate_dim,activation=leaky_relu)(inputs)
    x = Dense(intermediate_dim,activation=leaky_relu)(x)
    z_mean = Dense(latent_dim,name='z_mean')(x)
    z_log_var = Dense(latent_dim,name='z_log_var')(x)   
    
    # use reparameterization trick to push the sampling out as input
    # z_mean+sqrt(var)*eps , Adding zero-mean Gaussian noise
    z = Lambda(sampling,output_shape=(latent_dim,),name='z')([z_mean,z_log_var])
    
    encoder = Model(inputs,[z_mean,z_log_var,z],name='encoder')
    encoder.summary()
    plot_model(encoder,to_file='vae_pokemon_encoder.jpg',show_shapes=True)
    
    # decoder
    # p(x|z)
    latent_inputs = Input(shape=(latent_dim,),name='z_sampling')
    x = Dense(intermediate_dim,activation=leaky_relu)(latent_inputs)
    x = Dense(intermediate_dim,activation=leaky_relu)(x)
    outputs = Dense(dim,activation='sigmoid')(x) # 0~1
    
    decoder = Model(latent_inputs,outputs,name='decoder')
    decoder.summary()
    plot_model(decoder,to_file='vae_pokemon_decoder.jpg',show_shapes=True)
    
    # VAE
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs,outputs,name='vae_pokemon')

    models = (encoder,decoder)
    data = (X_train)
    
    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        loss = xent_loss + kl_loss
        return loss
    
    vae.compile(optimizer='adam',loss=vae_loss)
    vae.summary()
    plot_model(vae,to_file='vae_mlp_pokemon.jpg',show_shapes=True)
    vae.fit(X_train,X_train,epochs=epochs,batch_size=batch_size)
    vae.save_weights('vae_mlp_pokemon.h5')
    
    plot_results(models,data,batch_size=batch_size,model_name='vae_mlp')

def plot_results(models,
                 data,
                 batch_size=64,
                 model_name="vae_pokemon"):

    encoder, decoder = models
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "pokemon_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 36
    digit_size = 56
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    main()
