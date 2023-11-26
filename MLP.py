#Goals:
#Get the MLP to run forward prop on small subset of data
import numpy as np
import pandas as pd
import math
import random
from scipy.io import loadmat


#Lets start by grabbing out data (CSC89.mat must be in your file path and scipy.io must be installed)
chan89 = loadmat(f'CSC89.mat')
rawdata = chan89['data'][0]

#Now lets just grab a very small portion of the data for testing.
#NOTE: Think you gotta normalize this for sigmoid btw 0 and 1
rawdata = (rawdata - np.min(rawdata)) / (np.max(rawdata) - np.min(rawdata))
sec_dat = rawdata[800000:800005]

encoder_sizes = [5,3]
latent_space_size = 2
decoder_sizes = [3,5]

def make_net(encoder_sizes,latent_space_size,decoder_sizes):  
    #Initilize the network to have random weights between 0 and 1.
    #Hidden state ahead determines the number of sets and the current state determines the number of weights per set.
    
    mlp_init = []   #The network
    
    #Build the weights for the encoder (before sigma and mu)
    for k in range(len(encoder_sizes)-1): #There will be 1 less set of weights than labels
        random_numbers = [[random.random() for _ in range(encoder_sizes[k])]for _ in range(encoder_sizes[k+1])]
        mlp_init.append(random_numbers)
        #print('hello')
    
    #Build the weights that go to sigma and mu respecively
    #First layer here will be sigma and next will be mu
    random_numbers = [[[random.random() for _ in range(encoder_sizes[-1])]for _ in range(latent_space_size)]for _ in range(2)] #Is 2 since we are using sigma and mu as parameters
    mlp_init.append(random_numbers)
    #Then here we will sample and that will reduce the above layer from 2x2x1 to 2x1
    #Here we will need a special layer as well I think
    random_numbers = [[random.random() for _ in range(latent_space_size)]for _ in range(decoder_sizes[0])]
    mlp_init.append(random_numbers) 
    
    #Build the weights for the decoder (after sampling layer)
    for k in range(len(decoder_sizes)-1): #There will be 1 less set of weights than labels
        random_numbers = [[random.random() for _ in range(decoder_sizes[k])]for _ in range(decoder_sizes[k+1])]
        mlp_init.append(random_numbers)
    
    
    #print(mlp_init)
    return mlp_init    
    
mlp_weights = make_net(encoder_sizes,latent_space_size,decoder_sizes)



def sampler(sig_mu,latent_space_size):
    
    #Lets pick the first set as sigma and the second set as mu. Don't think this matters
    
    sample = []
    for k in range(latent_space_size):
        #Not 100% sure this is okay to do? I think the network will acount for it. Sigma has to be positive though
        sigma = abs(sig_mu[0][k])
        mean = sig_mu[1][k]
        sample.append(np.random.normal(mean, sigma))
        
    return sample




#Do fowrads propegation to put values in the mlp
def forward(sec_dat, mlp_weights,encoder_sizes,latent_space_size,decoder_sizes):
    #Just start with linear activation function for dense layers... you can always apply a sigmoid or somethin afterwards
    #The input will just be datapoints
    #This is a 2000x1 array
    
    #Going to store all of the values that the nodes in the network take on in an array called xi
    
    xi = []
    
    input_vals = sec_dat.tolist()
    xi.append(input_vals)
    
    #All of the encoder layers that are not special.
    for k in range(len(encoder_sizes)-1):
        elayer = mlp_weights[k]
        values = []
        for m in range(len(elayer)):
            #raw weights
            wis = np.array(elayer[m])
            xis = np.array(xi[-1])
            #For now just linear activation -- Not anymore... going to do sigma for delta rule
            value2 = sum(wis*xis)
            
            value = 1/(1+math.e**(-value2))    #sigmoid function
            
            values.append(value)
            
        xi.append(values)
    
    #Now we have our specialish layers for getting sigma and mu
    #for both sigma and mu
    sig_mu = []
    for k in range(2):
        values = []
        for m in range(latent_space_size):
            llayer = mlp_weights[len(encoder_sizes)-1]
            sig_or_mu = llayer[k]
            wis = np.array(sig_or_mu[m])
            xis = np.array(xi[-1])
            value2 = sum(wis*xis)
            
            value = 1/(1+math.e**(-value2))    #sigmoid function
            values.append(value)
        sig_mu.append(values)
    
    xi.append(sig_mu)
    
    plot_data = sampler(sig_mu,latent_space_size)
    
    xi.append(plot_data)
    
    #Decoder layers are not special here I don't think
    for k in range(len(decoder_sizes)):
        dlayer = mlp_weights[k + len(encoder_sizes)]
        values = []
        for m in range(len(dlayer)):
            #raw weights
            wis = np.array(dlayer[m])
            xis = np.array(xi[-1])
            value2 = sum(wis*xis)
            
            value = 1/(1+math.e**(-value2))    #sigmoid function
            values.append(value)
        xi.append(values)
    
    #print(xi)
    return xi
    
    
    #Now we have to sample from the distribution that we have "constructed" in order to get the start of the decoder.
    
xis = forward(sec_dat,mlp_weights,encoder_sizes,latent_space_size,decoder_sizes)


def backprop(xis,mlp_weights,encoder_sizes,latent_space_size,decoder_sizes):

    #To start I am just going to be looking at the reconstruction loss. I am not 100% sure yet if we need to incorperate the
    #other component of loss or if this is good enough.
    #Lets start by doing backprop with reconstruction error

    #learning rate
    eta = 50
    
    #print(mlp_weights)
    
    
    #Get the output deltas
    deltas = []
    output_deltas = []
    for m in range(len(xis[-1])):
        delta = (xis[-1][m] - xis[0][m])*xis[-1][m]*(1-xis[-1][m])
        output_deltas.append(delta)
    deltas.append(output_deltas)

    
    #print(deltas[-1][0])
    #print(deltas[-1][1])
        
    #Update weights and get new deltas
    for m in range(len(decoder_sizes)):
        layer = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m]
        for k in range(len(layer)):
            weight_set = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m][k]
            for z in range(len(weight_set)):
                #wji = wji-eta*delta*hi
                mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m][k][z] = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m][k][z] - eta*deltas[-1-m][k]*xis[-2-m][z]
    
        #Calculate new deltas
        #Only need to for stuff before sampling for now
        #delta current layer = sum(del_prev*weights to current delta) * hi(1-hi)
        if m<1:
            #just always 1 in front of what we are trying to get weights for
            #target_layer = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-2-m]
            layer = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m]
            hidden_deltas = []
            for q in range(len(xis[-2-m])):
                prev_dels = np.array(xis[-1-m])
                wjk = []
                for b in range(len(layer)):
                    wjk.append(mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m][b][q])
                wjk = np.array(wjk)
                
                sum_calc = sum(prev_dels*wjk)
                new_delta = sum_calc*xis[-2-m][q]
                hidden_deltas.append(new_delta)
                
                
            deltas.append(hidden_deltas)

        print(mlp_weights)
        print(deltas)

backprop(xis, mlp_weights,encoder_sizes,latent_space_size,decoder_sizes)



