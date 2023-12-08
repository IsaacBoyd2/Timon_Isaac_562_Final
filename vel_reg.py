#Goals

#1. Useing the raw data perform regression using MSE error loss function. (Can just take the encoder from the prvious model to do this)
#2a. Try using PCA as input
#2b. Try using VAE as input

#3. Record and report results. 

#The bulk of the program will utilize 2 functions:
    #Forwards propegation
    #Backwards propegation
    
#######################################
#Create the architecture of the model
#######################################

def make_net(layer_sizes):  
    mlp_init = []
    
    #Build the weights for the encoder (before sigma and mu)
    for k in range(len(layer_sizes)-1):
        random_numbers = [[random.random() for _ in range(layer_sizes[k])]for _ in range(layer_sizes[k+1])]
        mlp_init.append(random_numbers)

    return mlp_init   

########################
#Forwards Propegation
########################  

def forward(input_vals, mlp_weights,layer_sizes):
    #print('heloo')
    
    #This will store the in cell values
    xi = []
    xi.append(input_vals)
    
    for k in range(len(layer_sizes)-1):
        elayer = mlp_weights[k]
        values = []
        for m in range(len(elayer)):
            #raw weights
            wis = np.array(elayer[m])
            xis = np.array(xi[-1])
            value2 = sum(wis*xis)
            if k < len(layer_sizes)-2:
                value = 1/(1+math.e**(-value2))    #sigmoid function
            else:
                #print('hello')
                value = value2          
            values.append(value)
            
        xi.append(values)
    return xi
    
########################
#Backwards Propegation
########################    

def backprop(xis,label,mlp_weights,layer_sizes,eta, epochs):
    
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
            
            
    
    #print(deltas)
    #print('\n')
       

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

#Bring in imports
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat

########################
#Get the behavior data
########################

button = loadmat('behavioralDataPerEphysSample_samplingFrequency2000Hz_run2.mat')

#User X-pos
uxpog = button['unityData'][0][0][0][0]
#User Y-Pos
uypog = button['unityData'][0][0][1][0]

#Start of velocity task
instruction_switch = np.where(button['unityData'][0][0][3][0] == 1)[0][0]

#Get the user position
uxp = uxpog[instruction_switch:len(uxpog)]
uyp = uypog[instruction_switch:len(uypog)]

#Calculate instantaneous velocity
dxp = uxp[1:len(uxp)]-uxp[0:len(uxp)-1]
dyp = uyp[1:len(uyp)]-uyp[0:len(uyp)-1]

vel = np.sqrt(dxp**2 + dyp**2)*2000
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

print(min(insta_vel))
print(max(insta_vel))
print('here')

insta_vel = [(x - min(insta_vel)) / (max(insta_vel) - min(insta_vel)) for x in insta_vel]

print(min(insta_vel))
print(max(insta_vel))
def run_network(insta_vel,indices_of_nans,param1,param2,count):

    ########################
    #Get the LFP data
    ########################

    data_holder = []
    #num_channels = param1
    for iiii in param2:#range(89,89+num_channels):
        chan = loadmat(f'CSC{iiii}.mat')
        data = chan['data'][0]
        #Account for the velocity calculation
        data = data[instruction_switch+1:len(data)-1]
        data = data.tolist()
        data = data[0:len(data):downsample_rate]
        data = [value for index, value in enumerate(data) if index not in indices_of_nans]   
        data_holder.append(data)
        

    flat_data = [item for sublist in data_holder for item in sublist]
    min_value = min(flat_data)
    max_value = max(flat_data)
    data_holder = [[(x - min_value) / (max_value - min_value) for x in row] for row in data_holder]
    data_holder = np.array(data_holder).T.tolist()

    #print(len(insta_vel))
    #print(np.shape(data_holder))
    pre_shuffle_data = list(zip(data_holder, insta_vel))

    random.shuffle(pre_shuffle_data)

    shuffled_features, shuffled_labels = zip(*pre_shuffle_data)    

    #split 80 20
    split_index = int(len(shuffled_labels) * 0.8)
    train_data = shuffled_features[:split_index]
    test_data = shuffled_features[split_index:]

    #print(len(test_data))

    train_labels = shuffled_labels[:split_index]
    test_labels = shuffled_labels[split_index:]

    #print(len(test_labels))
        
    #######################
    #Set HyperParams
    #######################

    layer_sizes = [len(param2),1]
    epochs = 10000
    eta = 0.0001
    
    ##Main
    mlp_weights = make_net(layer_sizes)

    #Training
   
    for ep in range(epochs):
        #Track loss
        loss = []
        for k in range(len(train_labels)):
            #Forwards Propegate
            xis = forward(train_data[k], mlp_weights,layer_sizes)
            
            #print(xis)
            
            #Print Loss
            loss.append((xis[-1][0]-train_labels[k])**2)
            #Backwards Propegate
            backprop(xis,train_labels[k],mlp_weights,layer_sizes,eta, epochs)
            
            #print(mlp_weights)
            
    print(np.mean(loss))
     
    #Testing
    output = []
    for tes in range(len(test_labels)):
        xis = forward(test_data[tes], mlp_weights,layer_sizes)
        output.append(xis[-1][0])
    
    print('\n')
    #print(param1)
    print(np.std(output))
    print('\n')
    
    
    print(mlp_weights)

    plt.figure(count)
    plt.plot(test_labels)
    plt.plot(output)
    

    #Regression tasks.

    #1. Write a script to find the combination of layer values that 
    return np.std(output)
    
param2s = []  
for num in range(0,17):
    mid = []
    for thing in range(1,5):
        mid.append(thing+88+(5*num))
    param2s.append(mid)
print(param2s)

param1s = [0.0001]
param2s = [[89,90,91,92,93]]
#param2s = [2,3,4,5]
#param3s = [2,3,4,5]
#param4s = [2,3,4,5]
  
tracker = []
count = 1

for param1 in param1s:
    for param2 in param2s:
        tracker.append(run_network(insta_vel,indices_of_nans,param1,param2,count))
        count = count + 1
        #for param2 in param2s:
        #for param3 in param3s:
        #for param4 in param4s:
                
            
print(tracker)     
print(max(tracker))
print(tracker.index(max(tracker)))

plt.show()

# 4 channels
# 2 hidden layers  3,2 in len
#0.12

# 5 channels
# 0 hidden layer
#0.21
#89-93

