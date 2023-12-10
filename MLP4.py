#Goal here is just to make normal AE

def make_net(encoder_sizes,latent_space_size,decoder_sizes):  
    #Initilize the network to have random weights between 0 and 1.
    #Hidden state ahead determines the number of sets and the current state determines the number of weights per set.
    mlp_init = []   #The network
    nw_shape = np.concatenate((encoder_sizes, latent_space_size,decoder_sizes), axis=None)
    
    for k in range(len(nw_shape)-1):
        random_numbers = [[random.random() for _ in range(nw_shape[k])]for _ in range(nw_shape[k+1])]
        mlp_init.append(random_numbers)
    
    return mlp_init    

def forward(sec_dat, mlp_weights,encoder_sizes,latent_space_size,decoder_sizes,activations):

    #Going to store all of the values that the nodes in the network take on in an array called xi    
    xi = []
    
    input_vals = sec_dat.tolist()
    xi.append(input_vals)
    #All of the encoder layers that are not special.
    for k in range(len(encoder_sizes)+len(decoder_sizes)):
        elayer = mlp_weights[k]
        values = []
        for m in range(len(elayer)):
            #raw weights
            wis = np.array(elayer[m])
            xis = np.array(xi[-1])
            #For now just linear activation -- Not anymore... going to do sigma for delta rule
            value2 = sum(wis*xis)
            
            if activations[k] == 1:
                value = 1/(1+math.e**(-value2))    #sigmoid function
            else:
                value = value2
            
            values.append(value)
            
        xi.append(values)
    

    return xi
    
def backprop(xis, mlp_weights,encoder_sizes,latent_space_size,decoder_sizes,eta,insta_vel,Y_val,activations,lk):
    deltas = []
    
    for k in range(len(encoder_sizes)+len(decoder_sizes)):
        #Output delta has recon loss
        hidden_deltas = []
        for m in range(len(xis[-k-1])):
            if k == 0:
                val = (xis[0][m] - xis[-1][m])
            else:
                for q in range(len(xis[-k-1])):
                    wjk = []
                    prev_dels = np.array(deltas[-1])
                    for b in range(len(deltas[-1])):
                        wjk.append(mlp_weights[len(encoder_sizes)+len(decoder_sizes)-k][b][q])
                    wjk = np.array(wjk)
                    val = sum(prev_dels*wjk)
                    
            if activations[k] == 1:
                delta = val * xis[-k-1][m] * (1 - xis[-k-1][m])
            else:   #assuming sigmoid hidden layers (could change to like ReLu if stuff works)
                delta = val
                
            if k == len(decoder_sizes):
                delta = delta + 2*(xis[-k-1][m]-insta_vel[lk])*Y_val
                
            hidden_deltas.append(delta)
            
        deltas.append(hidden_deltas)
        
        #Update weights       
        layer = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-k]
        for v in range(len(layer)):
            weight_set = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-k][v]
            for z in range(len(weight_set)):  
                #wji = wji-eta*delta*hi            
                mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-k][v][z] = mlp_weights[len(decoder_sizes)+len(encoder_sizes)-1-k][v][z] - eta*deltas[-1][v]*xis[-2-k][z]
                        
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
    min_value = min(data)
    max_value = max(data)    
    data = [(x - min_value) / (max_value - min_value) for x in data]    
    
    #print(max(data))
    #print(min(data))   
    data_holder.append(data)

#Try normalizing based on everything
#Something seems off here... come back to it maybe
'''flat_data = [item for sublist in data_holder for item in sublist]


min_value = min(flat_data)
max_value = max(flat_data)

data_holder = [[(x - min_value) / (max_value - min_value) for x in row] for row in data_holder]'''

#print(max(max(data_holder)))

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
eta = 0.0001

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
        #Do forwards propegation
        xis = forward(e_input,mlp_weights,encoder_sizes,latent_space_size,decoder_sizes,activations)
        graph_data.append(xis[-len(decoder_sizes)-1])
        
        #print(xis)
            
        #Do backwards propegation
        backprop(xis, mlp_weights,encoder_sizes,latent_space_size,decoder_sizes,eta,insta_vel,Y_val,activations,lk)

        #Calculate Reconstruction loss
        recon = sum((np.array(xis[-1]) - np.array(xis[0]))**2)
        #Calculate KL loss
        custom_loss = sum((np.array(xis[-len(decoder_sizes)-1])-(np.ones(latent_space_size)*insta_vel[lk]))**2)
        #Calculate total loss
        
        recon_loss.append(recon)
        custom_loss_tot.append(custom_loss)
        
    print('recon_loss')
    print(np.mean(recon_loss))
    print('custom_loss')
    print(sum(custom_loss_tot))
    print('\n')

graph_data = np.array(graph_data).T

colors = insta_vel
# Create a scatter plot with colormap
fig = plt.figure(1)
plt.scatter(graph_data[0], graph_data[1],s = 50, c=colors, cmap='Oranges', edgecolors='none', alpha=0.7)

# Add a colorbar to show the mapping of colors
cbar = plt.colorbar()
cbar.set_label('Color Intensity')

# Customize the plot with labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Colormap')

plt.show()

#print(mlp_weights)

#Forwards propegate
#e_input = input_data[0]
#xis = forward(e_input,mlp_weights,encoder_sizes,latent_space_size,decoder_sizes,activations)

#print(xis)

#lk = 0

#backprop(xis, mlp_weights,encoder_sizes,latent_space_size,decoder_sizes,eta,insta_vel,Y_val,activations,lk)

#print(mlp_weights)





