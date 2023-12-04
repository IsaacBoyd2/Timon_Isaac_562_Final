#Goals:
#Get the MLP to run forward prop on small subset of data
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Get the behavior data
button = loadmat('behavioralDataPerEphysSample_samplingFrequency2000Hz_run2.mat')

#User X-pos
uxp = button['unityData'][0][0][0][0]
#User Y-Pos
uyp = button['unityData'][0][0][1][0]

#Set the number of trials that we do per epoch (size of training set)
data_points = 2000

#Start of velocity task
instruction_switch = np.where(button['unityData'][0][0][3][0] == 1)[0][0]

#Get the user position
uxp = uxp[instruction_switch:instruction_switch+data_points]
uyp = uyp[instruction_switch:instruction_switch+data_points]

#Calculate instantaneous velocity
dxp = uxp[1:len(uxp)-1]-uxp[0:len(uxp)-2]
dyp = uyp[1:len(uyp)-1]-uyp[0:len(uyp)-2]

vel = np.sqrt(dxp**2 + dyp**2)*2000  #2000 is sampling rate = 1/s

#Bring in the LFP data
data_holder = []
num_channels = 5
for iiii in range(89,89+num_channels):
    chan = loadmat(f'CSC{iiii}.mat')
    data = chan['data'][0]
    data = data[instruction_switch:instruction_switch+data_points]
    #Intra channel normalization maybe try whole dataset normalization later
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data_holder.append(data)
    
data_holder = np.array(data_holder)
input_data = data_holder.T


#rawdata = (rawdata - np.min(rawdata)) / (np.max(rawdata) - np.min(rawdata))
#sec_dat = rawdata[800000:800005]
    
#plt.figure(1)
#plt.plot(rawdata[600000:800000])

#plt.figure(2)
#plt.plot(vel[600000:800000])
#plt.show()


#Lets create a mock training set.

#Inner size of encoder and decoder ###MUST MATCH
#This might be a mistake
encoder_sizes = [5,4,4,3]
latent_space_size = 2
decoder_sizes = [3,4,4,5]

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
    
#xis = forward(sec_dat,mlp_weights,encoder_sizes,latent_space_size,decoder_sizes)


def backprop(xis,mlp_weights,encoder_sizes,latent_space_size,decoder_sizes,eta):

    #To start I am just going to be looking at the reconstruction loss. I am not 100% sure yet if we need to incorperate the
    #other component of loss or if this is good enough.
    #Lets start by doing backprop with reconstruction error
    
    loss_pre = []
    for m in range(len(xis[-1])):
        loss_mini = (xis[-1][m] - xis[0][m])
        loss_pre.append(loss_mini)
        
    loss_out = sum(loss_pre)
        
    
    
    #print(mlp_weights)
    
    
    #Get the output deltas
    deltas = []
    output_deltas = []
    for m in range(len(xis[-1])):
        delta = (xis[-1][m] - xis[0][m])*xis[-1][m]*(1-xis[-1][m])
        output_deltas.append(delta)
    deltas.append(output_deltas)



#        #              #
#   #    #     #   #    #    
#   #          #   #    #
#   #    #         #    #
#        #              #

   
        
    #Update weights and get new deltas
    for m in range(len(decoder_sizes)):
        layer = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m]
        for k in range(len(layer)):
            weight_set = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m][k]
            for z in range(len(weight_set)):
                
                #wji = wji-eta*delta*hi
                mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m][k][z] = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m][k][z] - eta*deltas[-1][k]*xis[-2-m][z]
    
        #Calculate new deltas
        #Only need to for stuff before sampling for now
        #delta current layer = sum(del_prev*weights to current delta) * hi(1-hi)
         
        if m < (len(decoder_sizes)-1): 
            #just always 1 in front of what we are trying to get weights for
            #target_layer = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-2-m]
            layer = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m]
            hidden_deltas = []
            for q in range(len(xis[-2-m])):
                prev_dels = np.array(deltas[0+m])
                wjk = []
                for b in range(len(layer)):
                    wjk.append(mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m][b][q])
                wjk = np.array(wjk)
                
                sum_calc = sum(prev_dels*wjk)
                new_delta = sum_calc*xis[-2-m][q]*xis[-1-m][k]*(1-xis[-1-m][k])#*Here there should be oj(1-oj)? (Should be good now)
                hidden_deltas.append(new_delta)
                
                
            deltas.append(hidden_deltas)
        
    #print(mlp_weights)
    #print('\n\n')
    #print(deltas)
    #print('\n\n')
    #print(xis)
    
    
        
    #KL loss is really only relavent in the encoder so we are going to keep it here
    #Now we have to update the weights on the super special layers that go into the mu and sigma
    #delta of pervious*partial of KL loss*input node to mu/sigma layer
    
    #Start with sigma connected nodes
    #Might be missing a term in previous deltas??? for sigmoids...
    #print('spacer')
    #print(deltas)
    #print('spacer2')
    
    sig_layer = mlp_weights[len(encoder_sizes)-1][0]
    for m in range(len(sig_layer)):
        weight_sets = sig_layer[m]
        for k in range(len(weight_sets)):
            #print(eta*deltas[-1][k]*(xis[-len(decoder_sizes)-2][0][m]))
            mlp_weights[len(encoder_sizes)-1][0][m][k] = mlp_weights[len(encoder_sizes)-1][0][m][k] - eta*deltas[-1][k]*(xis[-len(decoder_sizes)-2][0][m]*(1-xis[-len(decoder_sizes)-2][0][m]))*((-1/xis[-len(decoder_sizes)-2][0][m])+2*xis[-len(decoder_sizes)-2][0][m])*xis[-len(decoder_sizes)-3][k]
    
    #New deltas calculated are going to follow this layer
    after_sampling_deltas_sig = []
    for k in range(len(weight_sets)):
        delta = deltas[-1][k]*(xis[-len(decoder_sizes)-2][0][m]*(1-xis[-len(decoder_sizes)-2][0][m]))*((-1/xis[-len(decoder_sizes)-2][0][m])+2*xis[-len(decoder_sizes)-2][0][m])
        after_sampling_deltas_sig.append(delta)
        
    #deltas.append(after_sampling_deltas)
    
    #Have to add mu and somehow account for that in the new deltas
    sig_layer = mlp_weights[len(encoder_sizes)-1][1]
    for m in range(len(sig_layer)):
        weight_sets = sig_layer[m]
        
        #print('hello')
        #print(weight_sets)
        #print('goodbye')
        
        for k in range(len(weight_sets)):
            mlp_weights[len(encoder_sizes)-1][1][m][k] = mlp_weights[len(encoder_sizes)-1][1][m][k] - eta*deltas[-1][k]*(xis[-len(decoder_sizes)-2][1][m]*(1-xis[-len(decoder_sizes)-2][1][m]))*((-1/xis[-len(decoder_sizes)-2][1][m])+2*xis[-len(decoder_sizes)-2][1][m])*xis[-len(decoder_sizes)-3][k]
    
    #New deltas calculated are going to follow this layer
    after_sampling_deltas_mu = []
    for k in range(len(weight_sets)):
        delta = deltas[-1][k]*(xis[-len(decoder_sizes)-2][1][m]*(1-xis[-len(decoder_sizes)-2][1][m]))*((-1/xis[-len(decoder_sizes)-2][1][m])+2*xis[-len(decoder_sizes)-2][1][m])
        after_sampling_deltas_mu.append(delta)
    
    
    after_sampling_deltas_mu = np.array(after_sampling_deltas_mu)
    after_sampling_deltas_sig= np.array(after_sampling_deltas_sig)
    
    adder_thing = (after_sampling_deltas_sig+after_sampling_deltas_mu).tolist() 
    deltas.append(adder_thing)
    
    #Now we have to take care of updating the weights in the encoder.
    #Need to calculate deltas... just in case we add more layers to encoder
    #Update weights and get new deltas
    
    #print(xis)
    
    #print(mlp_weights)
     
    
    #Hey we wanna update the encoder cus we just did the sample layer and decoder
    #How we gunna do that?
    #Well the first layer was already done
    #We need encoder size - 1 times to do this
    for m in range(len(encoder_sizes)-1):
        print('askldjfha;lsdfh;alksjdfasf')
        #Going from back to front
        layer = mlp_weights[len(encoder_sizes)-2-m]
        for k in range(len(layer)):
            
            weight_set = mlp_weights[len(encoder_sizes)-2-m][k]
            #print(len(decoder_sizes)-2-m)
            #print('hello')
            #print(weight_set)
            #print('seeya')
            #print(xis[-5-m])
            for z in range(len(weight_set)):
                
                #print(z)                
                mlp_weights[len(encoder_sizes)-2-m][k][z] = mlp_weights[len(encoder_sizes)-2-m][k][z] - eta*deltas[-1][k]*xis[-len(decoder_sizes)-4-m][z]
                
                
                
        #Calculate new deltas
        #Only need to for stuff before sampling for now
        #delta current layer = sum(del_prev*weights to current delta) * hi(1-hi)
  
        #just always 1 in front of what we are trying to get weights for
        #target_layer = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-2-m]
        #Are we missing deltas in this calculation?
        layer = mlp_weights[len(encoder_sizes)-2-m]
        hidden_deltas = []
        
        #Lets look at the xis past the last line of the encoder
        for q in range(len(xis[-len(decoder_sizes)-4-m])):  #Need to un-hardcode the 5s here soon to be length of decoder + 1 + sampling layer shenanigans at some point
            
            #print(xis[-len(encoder_sizes)-4-m])
            
            #[0,1,2]
            #[1,2,3]
            #Probably can just be deltas[-1]
            prev_dels = np.array(deltas[len(decoder_sizes)+m])  #think this should be previous deltas not previous values FYI
            
            #print('hilo')
            #print(m)
            #print(xis[-6-m])
            #print(deltas[-3-m])
            #print(mlp_weights[len(encoder_sizes)-2-m])
            
            
            #print(deltas)
            
            #print(prev_dels)
            
            
            #print(mlp_weights[len(encoder_sizes)-2-m])
          
            
            wjk = []
            for b in range(len(layer)):
                #print(m)
                #print(q)
                #print(b)
                wjk.append(mlp_weights[len(encoder_sizes)-2-m][b][q])
            wjk = np.array(wjk)
            
            #print(wjk)
            
            #Reconstruction deltas
            sum_calc = sum(prev_dels*wjk)
            #new_delta = sum_calc*xis[-2-m][q]*xis[-1-m][k]*(1-xis[-1-m][k])
            #Either 433 or 322
            
            #print(xis[-len(encoder_sizes)-4-m][q])
            
            #Recon delta, h (value we are going to), (1-prev)(prev)
            new_delta = sum_calc*xis[-len(decoder_sizes)-4-m][q]*xis[-len(decoder_sizes)-3-m][k]*(1-xis[-len(decoder_sizes)-3-m][k])#*Here there should be oj(1-oj)? (Should be good now)
            hidden_deltas.append(new_delta)
            
            
        deltas.append(hidden_deltas)
    
    
    
    #print('\n\n')
    #print(mlp_weights)
    #print('\n\n')
    #print(deltas)
    
    
    return loss_out

#backprop(xis, mlp_weights,encoder_sizes,latent_space_size,decoder_sizes)


mlp_weights = make_net(encoder_sizes,latent_space_size,decoder_sizes)

#Start training

#print('input')
#print(input_data[0])

#learning rate
eta = 0.005

#Set number of epochs
epochs = 1000

loss_track_over_epoch = []

for pocs in range(epochs):
    recon_loss = []
    for lk in range(len(data_holder)):
        e_input = input_data[lk]
        #Do forwards propegation
        xis = forward(e_input,mlp_weights,encoder_sizes,latent_space_size,decoder_sizes)
        #Do backwards propegation
        loss_out = backprop(xis, mlp_weights,encoder_sizes,latent_space_size,decoder_sizes,eta)
        #Check the loss from backprop
        recon_loss.append(loss_out)
    loss_track_over_epoch.append(sum(recon_loss))


print('Looking at the reconstruction loss as the network optimizes')
print(loss_track_over_epoch)


#Problem: Reconstruction loss does not seem to
#         be getting much better between trials
#Ideas

#1. Go back to smaller dataset         (done)
#2. Try more layers (fix hardcoding)   (probably done)
#3. Try smaller layer sizes
#4. Try larger layer sizes
#5. Try smaller etas
#6. Try larger etas
#7. Try more epochs
#8. Try less epochs
  
