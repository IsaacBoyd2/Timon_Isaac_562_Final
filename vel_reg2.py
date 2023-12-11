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

    #Problem with one leyer network?
    mlp_weights = mlp_weights[0]
    
    
    
    
    xi = []
    xi.append(input_vals)
    
    for k in range(len(layer_sizes)-1):
        elayer = mlp_weights[k]
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
button2 = loadmat('behavioralDataPerEphysSample_samplingFrequency2000Hz_run5.mat')
#User X-pos
uxpog = button['unityData'][0][0][0][0]
#User Y-Pos
uypog = button['unityData'][0][0][1][0]

#User X-pos
uxpog2 = button2['unityData'][0][0][0][0]
#User Y-Pos
uypog2 = button2['unityData'][0][0][1][0]

#Start of velocity task
instruction_switch = np.where(button['unityData'][0][0][3][0] == 1)[0][0]
instruction_switch2 = np.where(button2['unityData'][0][0][3][0] == 1)[0][0]

#Get the user position
uxp = uxpog[instruction_switch:len(uxpog)]
uyp = uypog[instruction_switch:len(uypog)]

#Get the user position
uxp2 = uxpog2[instruction_switch2:len(uxpog2)]
uyp2 = uypog2[instruction_switch2:len(uypog2)]

#Calculate instantaneous velocity
dxp = uxp[1:len(uxp)]-uxp[0:len(uxp)-1]
dyp = uyp[1:len(uyp)]-uyp[0:len(uyp)-1]

#Calculate instantaneous velocity
dxp2 = uxp2[1:len(uxp2)]-uxp2[0:len(uxp2)-1]
dyp2 = uyp2[1:len(uyp2)]-uyp2[0:len(uyp2)-1]

vel = np.sqrt(dxp**2 + dyp**2)*2000
vel = np.array(vel)

vel2 = np.sqrt(dxp2**2 + dyp2**2)*2000
vel2 = np.array(vel2)

downsample_rate = 1000

#Estimate instantaneous velocity by taking midpoint between velocities
insta_vel = (vel[1:len(vel)] + vel[0:len(vel)-1])/2
insta_vel = insta_vel.tolist()
insta_vel = insta_vel[0:len(insta_vel):downsample_rate]

#Estimate instantaneous velocity by taking midpoint between velocities
insta_vel2 = (vel2[1:len(vel2)] + vel2[0:len(vel2)-1])/2
insta_vel2 = insta_vel2.tolist()
insta_vel2 = insta_vel2[0:len(insta_vel2):downsample_rate]

# Find indices of NaN values
nan_indices = np.isnan(insta_vel)

# Find indices of NaN values
nan_indices2 = np.isnan(insta_vel2)

# Get the indices where the value is True (indicating NaN)
indices_of_nans = np.where(nan_indices)[0]

# Get the indices where the value is True (indicating NaN)
indices_of_nans2 = np.where(nan_indices2)[0]

# Remove elements based on indices
insta_vel = [value for index, value in enumerate(insta_vel) if index not in indices_of_nans]

# Remove elements based on indices
insta_vel2 = [value for index, value in enumerate(insta_vel2) if index not in indices_of_nans2]


#print(min(insta_vel))
#print(max(insta_vel))
#print('here')

insta_vel = [(x - min(insta_vel)) / (max(insta_vel) - min(insta_vel)) for x in insta_vel]
insta_vel2 = [(x - min(insta_vel2)) / (max(insta_vel2) - min(insta_vel2)) for x in insta_vel2]

#print(min(insta_vel))
#print(max(insta_vel))


param1 = [0.0001]
param2 = [89,90,91,92,93]


########################
#Get the LFP data
########################

train_data_holder = []
#num_channels = param1
for iiii in param2:#range(89,89+num_channels):
    chan = loadmat(f'CSC{iiii}.mat')
    data = chan['data'][0]
    #Account for the velocity calculation
    data = data[instruction_switch+1:len(data)-1]
    data = data.tolist()
    data = data[0:len(data):downsample_rate]
    data = [value for index, value in enumerate(data) if index not in indices_of_nans]   
    train_data_holder.append(data)
   
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
    

flat_data = [item for sublist in train_data_holder for item in sublist]
min_value = min(flat_data)
max_value = max(flat_data)
train_data_holder = [[(x - min_value) / (max_value - min_value) for x in row] for row in train_data_holder]
train_data_holder = np.array(train_data_holder).T.tolist()

flat_data = [item for sublist in test_data_holder for item in sublist]
min_value = min(flat_data)
max_value = max(flat_data)
test_data_holder = [[(x - min_value) / (max_value - min_value) for x in row] for row in test_data_holder]
test_data_holder = np.array(test_data_holder).T.tolist()

#print(test_data_holder)

#print(len(insta_vel))
#print(np.shape(data_holder))
#pre_shuffle_data = list(zip(data_holder, insta_vel))

#random.shuffle(pre_shuffle_data)

#shuffled_features, shuffled_labels = zip(*pre_shuffle_data)    

#split 80 20
#split_index = int(len(shuffled_labels) * 0.8)
#train_data = shuffled_features[:split_index]
#test_data = shuffled_features[split_index:]

#print(len(test_data))

#train_labels = shuffled_labels[:split_index]
#test_labels = shuffled_labels[split_index:]

#print(len(test_labels))
    
#######################
#Set HyperParams
#######################

layer_sizes = [len(param2),1]
epochs = 1000
eta = 0.0001

##Main
mlp_weights = make_net(layer_sizes)

#print(mlp_weights)

#Training


#test_data_holder = test_data_holder.T
#train_data_holder = train_data_holder.T

#print('right here')
#print(len(train_data_holder))




for ep in range(epochs):
    #Track loss
    loss = []
    for k in range(len(train_data_holder)):
        #Forwards Propegate
        xis = forward(train_data_holder[k], mlp_weights,layer_sizes)
        
        #print(xis)
        
        #Print Loss
        loss.append((xis[-1][0]-train_data_holder[k])**2)
        #Backwards Propegate
        backprop(xis,train_data_holder[k],mlp_weights,layer_sizes,eta, epochs)
        
        #print(mlp_weights)
        
    print(np.mean(loss))
        
#print(np.mean(loss))
 
#Testing
output = []
for tes in range(len(test_data_holder)):

    #print(test_data_holder[tes])

    xis = forward(test_data_holder[tes], mlp_weights,layer_sizes)
    
    #print(xis)
    output.append(xis[-1][0])


print('MSE')
mse = np.mean((np.array(insta_vel2)-np.array(output))**2)
print(mse)

#print('\n')
#print(param1)
#print(np.std(output))
#print('\n')


#print(mlp_weights)

#print(output)


plt.plot(insta_vel2[250:350], label='Target velocity')
plt.plot(output[250:350], label='Predicted velocity')
plt.legend()

# Customize the plot with labels and title
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Regression: No Dimensionality Reduction')



            
#print(tracker)     
#print(max(tracker))
#print(tracker.index(max(tracker)))

plt.show()

# 4 channels
# 2 hidden layers  3,2 in len
#0.12

# 5 channels
# 0 hidden layer
#0.21
#89-93

