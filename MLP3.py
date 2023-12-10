

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
    #Perform sampling (using reparamaterization trick) z = mu + sig*epsilon
    sample = []
    grapher = []
    epsilons = []
    for k in range(latent_space_size):
        epsilon = np.random.normal(0, 1)   
        epsilons.append(epsilon)
        sigma = sig_mu[0][k]*epsilon
        
        mean = sig_mu[1][k]
        sample_output2 = mean+sigma
        
        #print(sigma)
        
        grapher.append(sample_output2)
        #Trying out sigmoid activation on sample layer
        #going to change this to linear for now
        #sample_output = 1/(1+math.e**(-sample_output2)) 
        sample_output = sample_output2
        sample.append(sample_output)
        
    return sample,epsilons,grapher

#Do fowrads propegation to put values in the mlp
def forward(sec_dat, mlp_weights,encoder_sizes,latent_space_size,decoder_sizes):

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
    
    #print(mlp_weights)
    
    xi.append(sig_mu)
    [plot_data,epsilons,grapher] = sampler(sig_mu,latent_space_size)
    
    #print(grapher)
    
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
            if k == len(decoder_sizes)-1:
                value = value2 #Linear activation
            else:
                value = 1/(1+math.e**(-value2))    #sigmoid function
            values.append(value)
        xi.append(values)

    return xi,epsilons,grapher


def backprop(xis,mlp_weights,encoder_sizes,latent_space_size,decoder_sizes,eta,kl_lambda, epochs,epsilons,insta_vel,graph_data,lk,Y_val):
    
    #Get the output deltas
    deltas = []
    output_deltas = []
    
    ################################################################################################################ 
    
    for m in range(len(xis[-1])):
        #delta 1: target - out #* sig'
        #Trying: Linear activation Note: Gotta change forwards prop for this to be valid
        delta = (xis[0][m] - xis[-1][m])#*xis[-1][m]*(1-xis[-1][m])
        output_deltas.append(delta)
    deltas.append(output_deltas)
    
    ################################################################################################################ 
    
    #Update weights and get new deltas
    for m in range(len(decoder_sizes)):
        layer = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m]
        for k in range(len(layer)):
            weight_set = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m][k]
            for z in range(len(weight_set)):  
                #wji = wji-eta*delta*hi
                mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m][k][z] = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m][k][z] + eta*deltas[-1][k]*xis[-2-m][z]
    
        #Now do the deltas for the decoder + update the weights of the decoder ***(for all values besides first layer of decoder since we might want to change its activation)
        #This if statement tells us to skip calculating the delta for the "z layer" so that we an do a custom activation
        if m < (len(decoder_sizes)-1): 
            #delta 2+: (decoder) = sig'*sum(d1k*wjk)
            hidden_deltas = []
            #Legnth of new deltas
            for q in range(len(xis[-2-m])):   
                prev_dels = np.array(deltas[-1])
                wjk = []
                #Length of previous deltas
                for b in range(len(deltas[-1])):
                    wjk.append(mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-m][b][q])
                wjk = np.array(wjk)

                sum_calc = sum(prev_dels*wjk)
                new_delta = sum_calc*xis[-2-m][q]*(1-xis[-2-m][q])
                hidden_deltas.append(new_delta)

            deltas.append(hidden_deltas)  
    
    ################################################################################################################ 
    
    #Delta "3": z-layer (assume it has a sigmoid activation) -- sig'*sum(d1k*wjk) -- can replace sig' with linear if needed
    hidden_deltas = []
    #Legnth of new deltas
    for q in range(len(xis[-len(decoder_sizes)-1])):
        prev_dels = np.array(deltas[-1])
        wjk = []
        #Length of previous deltas
        for b in range(len(deltas[-1])):
            wjk.append(mlp_weights[-len(decoder_sizes)][b][q])
        wjk = np.array(wjk)
        sum_calc = sum(prev_dels*wjk)
        
        #Optional Actvation here (currently sigmoid)
        #Idea: Add a custom loss function here that accounts for 
        new_delta = sum_calc#*xis[-len(decoder_sizes)-1][q]*(1-xis[-len(decoder_sizes)-1][q]) 
        #Idea: Enfore ceperation based on 1. velocity and 2. distance
        
        #print('graph_data')
        #print(xis[-len(decoder_sizes)-1])
        
        [custom_loss, cus_loss]= custom_loss_calc(insta_vel,xis[-len(decoder_sizes)-1][q],lk,Y_val,q)
        
        #print(custom_loss)
        
        hidden_deltas.append(new_delta+custom_loss)
           
          
    
    deltas.append(hidden_deltas)

    ################################################################################################################ 

    #Delta "4": Pre sampling = 1*delta3 - 2mu  &   epsilon*delta3 + 1/sig - sig
    #Note: Epsilons may or may not need to be the same. ALSO! KL lambda can be put in here
    #Note2: Check the signs on mu and sigma (this could be a source of error)
    hidden_deltas = []
    #Sigmas
    for k in range(latent_space_size):
        #Recon + -DKL
        sig_del = epsilons[k]*deltas[-1][k]# + ((1/xis[len(encoder_sizes)][0][k]) - xis[len(encoder_sizes)][0][k])*kl_lambda
        hidden_deltas.append(sig_del)
    
    #This is done seperately for ordering... first 2 in this layer are sigmas and then second 2 are mus due to the way we defined them in sampling.
    for k in range(latent_space_size):
        #Recon + -DKL
        mu_del = deltas[-1][k]# - xis[len(encoder_sizes)][1][k]*kl_lambda
        hidden_deltas.append(mu_del)
        
    deltas.append(hidden_deltas)
    
    #Update weights coming out of this layer
    #Update sigma and mu seperaetely (because of the way mlp_weights is built)
    
    count = 0
    for t in range(2):
        for k in range(latent_space_size):
            weight_set = mlp_weights[len(encoder_sizes)-1][t][k]
            for z in range(len(weight_set)):  
                mlp_weights[len(encoder_sizes)-1][t][k][z] = mlp_weights[len(encoder_sizes)-1][t][k][z] + eta*deltas[-1][count]*xis[len(encoder_sizes)-1][z]
            count = count + 1
                
    ################################################################################################################ 
    #Delta "5"+: sig'sum(delta4*wjk)  
    #Already did weights coming into encoder so we do 1 less then the encoder weights.
    for m in range(len(encoder_sizes)-1):
        hidden_deltas = []
        #Length of new deltas
        for q in range(len(xis[-len(decoder_sizes)-3-m])):   
            prev_dels = np.array(deltas[-1])
            wjk = []
            #Length of previous deltas
            
            #print(mlp_weights[len(encoder_sizes)-2-m][b][q])
            
            #There was an issue here bc. We are drawing from mlp witch has a nested list that it stores sigma and mu in. That is why this condition is written in.
            if m == 0:
                for b in range(2):
                    for s in range(latent_space_size):
                        wjk.append(mlp_weights[len(encoder_sizes)-1-m][b][s][q])
                
            else:
                for b in range(len(deltas[-1])):
                    wjk.append(mlp_weights[len(encoder_sizes)-1-m][b][q])
            
            wjk = np.array(wjk)

            sum_calc = sum(prev_dels*wjk)
            new_delta = sum_calc*xis[-len(decoder_sizes)-3-m][q]*(1-xis[-len(decoder_sizes)-3-m][q])
            hidden_deltas.append(new_delta)

        deltas.append(hidden_deltas) 
            
        layer = mlp_weights[len(encoder_sizes)-2-m]
        for k in range(len(layer)):
            weight_set = mlp_weights[len(encoder_sizes)-2-m][k]
            for z in range(len(weight_set)):  
                #wji = wji-eta*delta*hi
                mlp_weights[len(encoder_sizes)-2-m][k][z] = mlp_weights[len(encoder_sizes)-2-m][k][z] + eta*deltas[-1][k]*xis[-len(decoder_sizes)-4-m][z]
    
    #print(xis) 
    return cus_loss

            
def custom_loss_calc(insta_vel,label,lk,Y_val,q):
    #Make sure insta_vel[lk] is correct
    
    #print('label')
    #print(label)
    #print('velocity')
    #print(insta_vel[lk])
    
    loss = 2*(label-insta_vel[lk])*Y_val
    track_loss = (label-insta_vel[lk])**2
    return loss,track_loss
        
    

import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Get the behavior data
button = loadmat('behavioralDataPerEphysSample_samplingFrequency2000Hz_run2.mat')

#User X-pos
uxpog = button['unityData'][0][0][0][0]
#User Y-Pos
uypog = button['unityData'][0][0][1][0]


#Start of velocity task
instruction_switch = np.where(button['unityData'][0][0][3][0] == 1)[0][0]


#Get the user position
uxp = uxpog[instruction_switch:len(uypog)]
uyp = uypog[instruction_switch:len(uypog)]

#Calculate instantaneous velocity
dxp = uxp[1:len(uxp)]-uxp[0:len(uxp)-1]
dyp = uyp[1:len(uyp)]-uyp[0:len(uyp)-1]

vel = np.sqrt(dxp**2 + dyp**2)*2000  #2000 is sampling rate = 1/s
vel = np.array(vel)

downsample_rate = 1000

#Estimate instantaneous velocity by taking midpoint between velocities
insta_vel = (vel[1:len(vel)] + vel[0:len(vel)-1])/2
insta_vel = insta_vel.tolist()
insta_vel = insta_vel[0:len(insta_vel):downsample_rate]

# Find indices of NaN values
nan_indices = np.isnan(insta_vel)

# Get the indices where the value is True (indicating NaN)
indices_of_nans = np.where(nan_indices)[0]

# Remove elements based on indices
insta_vel = [value for index, value in enumerate(insta_vel) if index not in indices_of_nans]
insta_vel = [(x - min(insta_vel)) / (max(insta_vel) - min(insta_vel)) for x in insta_vel]

#Bring in the LFP data
data_holder = []
num_channels = 5
for iiii in range(89,89+num_channels):
    chan = loadmat(f'CSC{iiii}.mat')
    data = chan['data'][0]
    #Try just taking magnitude?
    data = abs(data)
    
    #Account for the velocity calculation
    data = data[instruction_switch+1:len(data)-1]
    data = data.tolist()
    data = data[0:len(data):downsample_rate]
    data = [value for index, value in enumerate(data) if index not in indices_of_nans]  
    
    
    
    data_holder.append(data)

#Try normalizing based on everything
flat_data = [item for sublist in data_holder for item in sublist]

'''x = np.linspace(0,1,10)

plt.figure(1000)
plt.hist(flat_data,x)
plt.show()'''

min_value = min(flat_data)
max_value = max(flat_data)
data_holder = [[(x - min_value) / (max_value - min_value) for x in row] for row in data_holder]
data_holder = np.array(data_holder)
input_data = data_holder.T


#Define network shape
encoder_sizes = [num_channels,3]
latent_space_size = 2
decoder_sizes = [3,num_channels]

#Define hyperparameters

kl_lambda = 100
Y_val = 10000

#learning rate
eta = 0.0001

#Set number of epochs
epochs = 2

loss_track_over_epoch = []


#Generate network
mlp_weights = make_net(encoder_sizes,latent_space_size,decoder_sizes)

for pocs in range(epochs):

    graph_data = []
    recon_loss = []
    kl_loss = []
    custom_loss_tot = []
    for lk in range(len(data_holder[0])):
        e_input = input_data[lk]
        #Do forwards propegation
        [xis,epsilons,grapher] = forward(e_input,mlp_weights,encoder_sizes,latent_space_size,decoder_sizes)
        
        #if pocs == epochs-1:
        #graph_data.append(xis[-len(decoder_sizes)-1])
        #print(grapher)
        graph_data.append(grapher)
            
        #Do backwards propegation
        #Going to throw labels at backprop for new loss function
        cus_loss_iter = backprop(xis, mlp_weights,encoder_sizes,latent_space_size,decoder_sizes,eta,kl_lambda, epochs,epsilons,insta_vel,graph_data,lk,Y_val)
        custom_loss_tot.append(cus_loss_iter)
        #Calculate Reconstruction loss
        recon = sum((np.array(xis[-1]) - np.array(xis[0])))**2
        #Calculate KL loss
        kl = (1/2)*(1 + math.log(xis[len(encoder_sizes)][0][0]**2) - xis[len(encoder_sizes)][1][0] - xis[len(encoder_sizes)][0][0]**2)
        #Calculate total loss
        
        recon_loss.append(recon)
        kl_loss.append(kl)
        
    print('recon_loss')
    print(np.mean(recon_loss))
    print('kl_loss')
    print(np.mean(kl_loss))
    print('\n')
    print('custom_loss')
    print(sum(custom_loss_tot))
    print('\n')



graph_data = np.array(graph_data).T

colors = insta_vel
# Create a scatter plot with colormap
fig = plt.figure(1)
plt.scatter(graph_data[0], graph_data[1],s = 50, c=colors, cmap='Oranges', edgecolors='none', alpha=0.7)


# Show the plot
plt.show()



