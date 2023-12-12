

def make_net(encoder_sizes, latent_space_size, decoder_sizes):
    # Initilize the network to have random weights between 0 and 1.
    # Hidden state ahead determines the number of sets and the current state determines the number of weights per set.
    mlp_init = []  # The network
    nw_shape = np.concatenate((encoder_sizes, latent_space_size, decoder_sizes), axis=None)

    for k in range(len(nw_shape) - 1):
        random_numbers = [[random.random() for _ in range(nw_shape[k])] for _ in range(nw_shape[k + 1])]
        mlp_init.append(random_numbers)

    return mlp_init


def forward(sec_dat, mlp_weights, encoder_sizes, latent_space_size, decoder_sizes, activations):
    # Going to store all of the values that the nodes in the network take on in an array called xi
    xi = []

    #input_vals = sec_dat.tolist()
    xi.append(sec_dat)
    # All of the encoder layers that are not special.
    for k in range(len(encoder_sizes) + len(decoder_sizes)):
        elayer = mlp_weights[k]
        values = []
        for m in range(len(elayer)):
            # raw weights
            wis = np.array(elayer[m])
            xis = np.array(xi[-1])
            # For now just linear activation -- Not anymore... going to do sigma for delta rule
            value2 = sum(wis * xis)

            if activations[k] == 1:
                value = 1 / (1 + math.e ** (-value2))  # sigmoid function
            else:
                value = value2

            values.append(value)

        xi.append(values)

    return xi


def backprop(xis, mlp_weights,encoder_sizes,latent_space_size,decoder_sizes,eta,insta_vel,Y_val,activations,lk):
    deltas = []

    for k in range(len(encoder_sizes) + len(decoder_sizes)):
        # Output delta has recon loss
        hidden_deltas = []
        for m in range(len(xis[-k - 1])):
            if k == 0:
                val = (xis[0][m] - xis[-1][m])
            else:
                for q in range(len(xis[-k - 1])):
                    wjk = []
                    prev_dels = np.array(deltas[-1])
                    for b in range(len(deltas[-1])):
                        wjk.append(mlp_weights[len(encoder_sizes) + len(decoder_sizes) - k][b][q])
                    wjk = np.array(wjk)
                    val = sum(prev_dels * wjk)

            if activations[k] == 1:
                delta = val * xis[-k - 1][m] * (1 - xis[-k - 1][m])
            else:  # assuming sigmoid hidden layers (could change to like ReLu if stuff works)
                delta = val

            if k == len(decoder_sizes):
                delta = delta + 2 * (xis[-k - 1][m] - insta_vel[lk]) * Y_val
                cus_loss = 2*(xis[-k-1][m]-insta_vel[lk])*Y_val

            hidden_deltas.append(delta)

        deltas.append(hidden_deltas)

        # Update weights
        layer = mlp_weights[len(decoder_sizes) + len(encoder_sizes) - 1 - k]
        for v in range(len(layer)):
            weight_set = mlp_weights[len(decoder_sizes) + len(encoder_sizes) - 1 - k][v]
            for z in range(len(weight_set)):
                # wji = wji-eta*delta*hi
                mlp_weights[len(decoder_sizes) + len(encoder_sizes) - 1 - k][v][z] = mlp_weights[len(decoder_sizes) + len(encoder_sizes) - 1 - k][v][z] - eta * deltas[-1][v] * xis[-2 - k][z]
    
    #print(xis) 
    #return cus_loss


        
    

import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

os.chdir('C:/Users/wonde/Documents/School/BU Year 1/CSC/')
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
param2 = [89,90,91,92,93]
num_channels = len(param2)
for iiii in param2:#range(89,89+num_channels):
    chan = loadmat(f'CSC{iiii}.mat')
    data = chan['data'][0]
    data = abs(data)
    #Account for the velocity calculation
    data = data[instruction_switch+1:len(data)-1]
    data = data.tolist()
    data = data[0:len(data):downsample_rate]
    data = [value for index, value in enumerate(data) if index not in indices_of_nans]
    data_holder.append(data)

#Try normalizing based on everything
flat_data = [item for sublist in data_holder for item in sublist]


min_value = min(flat_data)
max_value = max(flat_data)
data_holder = [[(x - min_value) / (max_value - min_value) for x in row] for row in data_holder]
data_holder = np.array(data_holder)
input_data = data_holder.T


#Define network shape
encoder_sizes = [num_channels,3]
latent_space_size = 2
decoder_sizes = [3,num_channels]

activations = [1,0,1,0]

#Define hyperparameters

kl_lambda = 100
Y_val = 1

#learning rate
eta = 0.01

#Set number of epochs
epochs = 100

loss_track_over_epoch = []


#Generate network
mlp_weights = make_net(encoder_sizes,latent_space_size,decoder_sizes)

for pocs in range(epochs):

    graph_data = []
    recon_loss = []
    custom_loss_tot = []
    for lk in range(len(data_holder[0])):
        e_input = input_data[lk]
        # Do forwards propegation
        xis = forward(e_input, mlp_weights, encoder_sizes, latent_space_size, decoder_sizes, activations)
        graph_data.append(xis[-len(decoder_sizes) - 1])

        # print(xis)

        # Do backwards propegation
        backprop(xis, mlp_weights, encoder_sizes, latent_space_size, decoder_sizes, eta, insta_vel, Y_val, activations,
                 lk)

        # Calculate Reconstruction loss
        recon = sum((np.array(xis[-1]) - np.array(xis[0])) ** 2)
        # Calculate KL loss
        custom_loss = sum((np.array(xis[-len(decoder_sizes) - 1]) - (np.ones(latent_space_size) * insta_vel[lk])) ** 2)
        # Calculate total loss

        recon_loss.append(recon)
        custom_loss_tot.append(custom_loss)
        
    '''print('recon_loss')
    print(np.mean(recon_loss))
    print('kl_loss')
    print(np.mean(kl_loss))
    print('\n')
    print('custom_loss')
    print(sum(custom_loss_tot))
    print('\n')'''
graph_data = np.array(graph_data).T

colors = insta_vel
# Create a scatter plot with colormap
fig = plt.figure(1)
plt.scatter(graph_data[0], graph_data[1],s = 50, c=colors, cmap='Oranges', edgecolors='none', alpha=0.7)
plt.show()

#graph_data = np.array(graph_data).T

###############################################################################################
#Trained on 2 and graph_data contains our points
#####################################################################################
#Now we gotta forwards prop on 5
button2 = loadmat('behavioralDataPerEphysSample_samplingFrequency2000Hz_run5.mat')
#User X-pos
uxpog2 = button2['unityData'][0][0][0][0]
#User Y-Pos
uypog2 = button2['unityData'][0][0][1][0]
instruction_switch2 = np.where(button2['unityData'][0][0][3][0] == 1)[0][0]
#Get the user position
uxp2 = uxpog2[instruction_switch2:len(uxpog2)]
uyp2 = uypog2[instruction_switch2:len(uypog2)]
#Calculate instantaneous velocity
dxp2 = uxp2[1:len(uxp2)]-uxp2[0:len(uxp2)-1]
dyp2 = uyp2[1:len(uyp2)]-uyp2[0:len(uyp2)-1]
vel2 = np.sqrt(dxp2**2 + dyp2**2)*2000
vel2 = np.array(vel2)
#Estimate instantaneous velocity by taking midpoint between velocities
insta_vel2 = (vel2[1:len(vel2)] + vel2[0:len(vel2)-1])/2
insta_vel2 = insta_vel2.tolist()
insta_vel2 = insta_vel2[0:len(insta_vel2):downsample_rate]
# Find indices of NaN values
nan_indices2 = np.isnan(insta_vel2)
# Get the indices where the value is True (indicating NaN)
indices_of_nans2 = np.where(nan_indices2)[0]
# Remove elements based on indices
insta_vel2 = [value for index, value in enumerate(insta_vel2) if index not in indices_of_nans2]
insta_vel2 = [(x - min(insta_vel2)) / (max(insta_vel2) - min(insta_vel2)) for x in insta_vel2]
param2 = [89,90,91,92,93]



test_data_holder = []   
for nk in param2:
    chan = loadmat(f'CSC{iiii}_5.mat')
    data = chan['data'][0]
    #Account for the velocity calculation
    data = data[instruction_switch2+1:len(data)-1]
    data = data.tolist()
    data = data[0:len(data):downsample_rate]
    data = [value for index, value in enumerate(data) if index not in indices_of_nans2]   
    test_data_holder.append(data)
    
flat_data = [item for sublist in test_data_holder for item in sublist]
min_value = min(flat_data)
max_value = max(flat_data)
test_data_holder = [[(x - min_value) / (max_value - min_value) for x in row] for row in test_data_holder]
test_data_holder = np.array(test_data_holder).T.tolist()

input_data = np.array(test_data_holder)

print(input_data)

graph_data2 = []
for lk in range(len(test_data_holder)):
    e_input = input_data[lk]
    # Do forwards propegation
    xis = forward(e_input, mlp_weights, encoder_sizes, latent_space_size, decoder_sizes, activations)


    graph_data2.append(xis[-len(decoder_sizes) - 1])
    

########################################################################################
# Points for 5
########################################################################################

########################
#Forwards Propegation
########################  

def forward2(input_vals, mlp_weights,layer_sizes):

    #Problem with one leyer network?
    #mlp_weights = mlp_weights[0]
    
    
    
    
    xi = []
    xi.append(input_vals)
    
    for k in range(len(layer_sizes)-1):
        
        #print('mlp')
        #print(mlp_weights)
        
        #print('elayer')
        
        elayer = mlp_weights[k]
        
        #print(elayer)
        
        values = []
        
        #print(k)
        
        for m in range(len(elayer)):
            #raw weights
            wis = np.array(elayer[m])
            
            #print('wis')
            #print(wis)
            
            xis = np.array(xi[-1])
            
            #print('xis')
            #print(xis)
            
            value2 = sum(wis*xis)
            #print('val')
            #print(value2)
            
            if k < len(layer_sizes)-2:
                value = 1/(1+math.e**(-value2))    #sigmoid function
            else:
                #print('hello')
                value = value2          
            values.append(value)
            
        xi.append(values)
        
    #print(xi)
    return xi
    
########################
#Backwards Propegation
########################    

def backprop2(xis,label,mlp_weights,layer_sizes,eta, epochs):
    
    #print('heloo')
    #print(xis)
    #print('\n')
    #print(mlp_weights)
    #print('\n')
    deltas = []
    #Got rid of sig' (linear output)
    #deltas.append([(xis[-1][0] - label)*xis[-1][0]*(1-xis[-1][0])])
    #should be target - guess?
    #deltas.append([(xis[-1][0] - label)])
    deltas.append([(label-xis[-1][0])])
    
    #Update weights and get new deltas
    for m in range(len(layer_sizes)-1):
        layer = mlp_weights[len(layer_sizes)-2-m]
        

        for k in range(len(layer)):
            weight_set = mlp_weights[len(layer_sizes)-2-m][k]
            for z in range(len(weight_set)):                
                #wji = wji-eta*delta*hi
                #warning ** Changed to + ???? WHY?
                mlp_weights[len(layer_sizes)-2-m][k][z] = mlp_weights[len(layer_sizes)-2-m][k][z] + eta*deltas[-1][k]*xis[-2-m][z]
                
        #print(mlp_weights)
        #print('\n')
        
        if m < (len(layer_sizes)-2): 
            hidden_deltas = []
            for q in range(len(xis[-2-m])):
                prev_dels = np.array(deltas[-1])
                wjk = []
                for b in range(len(layer)):
                    wjk.append(mlp_weights[len(layer_sizes)-2-m][b][q])
                wjk = np.array(wjk)
                
                sum_calc = sum(prev_dels*wjk)
                new_delta = sum_calc*xis[-2-m][q]*(1-xis[-2-m][q])
                hidden_deltas.append(new_delta)
                               
            deltas.append(hidden_deltas)
     
def make_net2(layer_sizes):  
    mlp_init = []
    
    #Build the weights for the encoder (before sigma and mu)
    for k in range(len(layer_sizes)-1):
        random_numbers = [[random.random() for _ in range(layer_sizes[k])]for _ in range(layer_sizes[k+1])]
        mlp_init.append(random_numbers)
    
    print('ehllo')
    print(mlp_init)
    
    return mlp_init   
     
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

layer_sizes = [2,5,1]
epochs = 1000
eta = 0.001

mlp_weights = make_net2(layer_sizes)

#print('ehllo')
#print(mlp_weights)


train_data_holder = graph_data.T

for ep in range(epochs):
    #Track loss
    loss = []
    for k in range(len(train_data_holder)):
        #Forwards Propegate
        xis = forward2(train_data_holder[k], mlp_weights,layer_sizes)
        
        #print('input')
        #print(mlp_weights)
        
        #print('values')
        #print(xis)
        
        #Print Loss
        loss.append((xis[-1][0]-insta_vel[k])**2)
        #Backwards Propegate
        backprop2(xis,insta_vel[k],mlp_weights,layer_sizes,eta, epochs)
        
        #print(mlp_weights)
        
    print(np.mean(loss))
        
#print(np.mean(loss))
 
 
test_data_holder = graph_data2

#Testing
output = []
for tes in range(len(test_data_holder)):

    #print(test_data_holder[tes])

    xis = forward2(test_data_holder[tes], mlp_weights,layer_sizes)
    
    #print(xis)
    output.append(xis[-1][0])

#print(mlp_weights)

print('MSE')
mse = np.mean((np.array(insta_vel2)-np.array(output))**2)
print(mse)

#print('\n')
#print(param1)
#print(np.std(output))
#print('\n')


#print(mlp_weights)

#print(output)

fig = plt.figure(2)
plt.plot(insta_vel2[250:350], label='Target velocity')
plt.plot(output[250:350], label='Predicted velocity')
plt.legend()

# Customize the plot with labels and title
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Regression: AE')



            
#print(tracker)     
#print(max(tracker))
#print(tracker.index(max(tracker)))

plt.show()


print('Graph_data')
print(graph_data)
print('Graph_data2')
print(graph_data2)




'''colors = insta_vel
# Create a scatter plot with colormap
fig = plt.figure(1)
plt.scatter(graph_data[0], graph_data[1],s = 50, c=colors, cmap='Oranges', edgecolors='none', alpha=0.7)

# Add a colorbar to show the mapping of colors
cbar = plt.colorbar()
cbar.set_label('Velocity')

# Customize the plot with labels and title
plt.xlabel('First Latent Dimension')
plt.ylabel('Second Latent Dimension')
plt.title('VAE Latent Space')

plt.show()

# Show the plot
plt.show()'''



