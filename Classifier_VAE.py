

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
        
        hidden_deltas.append(new_delta)#+custom_loss)
           
          
    
    deltas.append(hidden_deltas)

    ################################################################################################################ 

    #Delta "4": Pre sampling = 1*delta3 - 2mu  &   epsilon*delta3 + 1/sig - sig
    #Note: Epsilons may or may not need to be the same. ALSO! KL lambda can be put in here
    #Note2: Check the signs on mu and sigma (this could be a source of error)
    hidden_deltas = []
    #Sigmas
    for k in range(latent_space_size):
        #Recon + -DKL
        sig_del = epsilons[k]*deltas[-1][k] + ((1/xis[len(encoder_sizes)][0][k]) - xis[len(encoder_sizes)][0][k])*kl_lambda
        hidden_deltas.append(sig_del)
    
    #This is done seperately for ordering... first 2 in this layer are sigmas and then second 2 are mus due to the way we defined them in sampling.
    for k in range(latent_space_size):
        #Recon + -DKL
        mu_del = deltas[-1][k] - xis[len(encoder_sizes)][1][k]*kl_lambda
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
import seaborn as sns

button = loadmat('behavioralDataPerEphysSample_samplingFrequency2000Hz_run2.mat')
button2 = loadmat('behavioralDataPerEphysSample_samplingFrequency2000Hz_run5.mat')


#Start of velocity task
instruction_switch = np.where(button['unityData'][0][0][3][0] == 1)[0][0]
instruction_switch2 = np.where(button2['unityData'][0][0][3][0] == 1)[0][0]

button_press = button['unityData'][0][0][2][0]
button_press = button_press[instruction_switch:len(button_press)]

button_press2 = button2['unityData'][0][0][2][0]
button_press2 = button_press2[instruction_switch:len(button_press2)]


#print(button2['unityData'][0][0][2][0])
#print('jello')

#plt.plot(button2['unityData'][0][0][2][0])
#plt.show()

downsample_rate = 1000

button_press = button_press[0:len(button_press):downsample_rate]
button_press2 = button_press2[0:len(button_press2):downsample_rate]

#plt.plot(button2['unityData'][0][0][2][0])
#plt.show()

#print(len(button_press))

# No Nans in this data. Checked.
num_channels = 54

# Find indices of NaN values
nan_indices1 = np.isnan(button_press)
indices_of_nans1 = np.where(nan_indices1)[0]

nan_indices2 = np.isnan(button_press2)
indices_of_nans2 = np.where(nan_indices2)[0]

#print(indices_of_nans)

button_press = [value for index, value in enumerate(button_press) if index not in indices_of_nans1]
button_press2 = [value for index, value in enumerate(button_press2) if index not in indices_of_nans2]

#print(button_press2)

#Anticipatory signal
#(shift labels back 1 second)
button_press = button_press[2:len(button_press)]
button_press2 = button_press2[2:len(button_press2)]

#print(button_press2)

train_data_holder = []
for iiii in range(89,89+num_channels):
    chan = loadmat(f'CSC{iiii}.mat')
    data = chan['data'][0]
    data = data.tolist()
    data = data[instruction_switch:len(data)]
    data = data[0:len(data):downsample_rate]
    
    data = [value for index, value in enumerate(data) if index not in indices_of_nans1]
    
    data = data[0:len(data)-2]
    train_data_holder.append(data)
    
#print(np.shape(button_press))
    
flat_data = [item for sublist in train_data_holder for item in sublist]
min_value = min(flat_data)
max_value = max(flat_data)
train_data_holder = [[(x - min_value) / (max_value - min_value) for x in row] for row in train_data_holder]
train_data_holder = np.array(train_data_holder).T.tolist()
    
    
################################################

test_data_holder = []
for iiii in range(89,89+num_channels):
    chan = loadmat(f'CSC{iiii}_5.mat')
    data = chan['data'][0]
    data = data.tolist()
    data = data[instruction_switch:len(data)]
    data = data[0:len(data):downsample_rate]
    data = [value for index, value in enumerate(data) if index not in indices_of_nans2]
    
    data = data[0:len(data)-2]
    test_data_holder.append(data)
    
#print(np.shape(button_press))

#print('here')    
#print(button_press2)

    
    
flat_data = [item for sublist in test_data_holder for item in sublist]
min_value = min(flat_data)
max_value = max(flat_data)
test_data_holder = [[(x - min_value) / (max_value - min_value) for x in row] for row in test_data_holder]
test_data_holder = np.array(test_data_holder).T.tolist()
    
#print(test_data_holder[0:5])
#print(train_data_holder[0:5])


#Define network shape
encoder_sizes = [num_channels,20,20]
latent_space_size = 10
decoder_sizes = [20,20,num_channels]

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

input_data = np.array(train_data_holder)



#print(train_data_holder)

print('len of input data')
print(len(train_data_holder))
print(len(button_press))

for pocs in range(epochs):
    print(pocs)
    graph_data = []
    recon_loss = []
    kl_loss = []
    custom_loss_tot = []
    chan1og = []
    chan1re = []
    for lk in range(len(train_data_holder)):
        e_input = input_data[lk]
        #print('input')
        #print(e_input)
        
        #Do forwards propegation
        [xis,epsilons,grapher] = forward(e_input,mlp_weights,encoder_sizes,latent_space_size,decoder_sizes)
        
        #if pocs == epochs-1:
        #graph_data.append(xis[-len(decoder_sizes)-1])
        #print(grapher)
        graph_data.append(grapher)
        #print('graph data')
        #print(grapher)
            
        #Do backwards propegation
        #Going to throw labels at backprop for new loss function
        cus_loss_iter = backprop(xis, mlp_weights,encoder_sizes,latent_space_size,decoder_sizes,eta,kl_lambda, epochs,epsilons,button_press,graph_data,lk,Y_val)
        custom_loss_tot.append(cus_loss_iter)
        #Calculate Reconstruction loss
        recon = sum((np.array(xis[-1]) - np.array(xis[0])))**2
        #Calculate KL loss
        kl = (1/2)*(1 + math.log(xis[len(encoder_sizes)][0][0]**2) - xis[len(encoder_sizes)][1][0] - xis[len(encoder_sizes)][0][0]**2)
        #Calculate total loss
        
        recon_loss.append(recon)
        kl_loss.append(kl)
        
        chan1og.append(xis[0][0])
        chan1re.append(xis[-1][0])
        
    print('recon_loss')
    print(sum(recon_loss))
    #print('kl_loss')
    #print(np.mean(kl_loss))
    #print('\n')
    #print(xis)
    
for_graphing = np.array(graph_data).T
    
averaged_chan1og = []
averaged_chan1re = [] 
for j in range(len(chan1og)-50):
    averaged_chan1og.append(np.mean(chan1og[j:j+50]))
    averaged_chan1re.append(np.mean(chan1re[j:j+50]))
    
#Reconstructed signal

fig = plt.figure(8)
plt.plot(averaged_chan1og,'k')
plt.plot(averaged_chan1re,'c')
plt.ylim(0.25, 0.75)

plt.xlabel('Trial')
plt.ylabel('Magnitude')
plt.title('VAE - reconstruction visualization')



colors = button_press
# Create a scatter plot with colormap
fig = plt.figure(10)
plt.scatter(for_graphing[0], for_graphing[1],s = 50, c=colors, cmap='cividis', edgecolors='none', alpha=0.7)

# Add a colorbar to show the mapping of colors
cbar = plt.colorbar()
cbar.set_label('Button Press')

# Customize the plot with labels and title
plt.xlabel('First Latent Dimension')
plt.ylabel('Second Latent Dimension')
plt.title('VAE - First 2 Latent Dimensions')
plt.autoscale()

plt.show()

#graph_data = np.array(graph_data).T

###############################################################################################
#Trained on 2 and graph_data contains our points
#####################################################################################



input_data = np.array(test_data_holder)

#print(input_data)

graph_data2 = []
for lk in range(len(test_data_holder)):
    e_input = input_data[lk]
    #Do forwards propegation
    [xis,epsilons,grapher] = forward(e_input,mlp_weights,encoder_sizes,latent_space_size,decoder_sizes)

    graph_data2.append(grapher)
    

########################################################################################
# Points for 5
########################################################################################

########################
#Forwards Propegation
########################  

def forward2(input_vals, mlp_weights,layer_sizes,flag):

    #Problem with one leyer network?
    #mlp_weights = mlp_weights[0]
    
    
    
    
    xi = []
    xi.append(input_vals)
    
    
    #print('input vals')
    #print(input_vals)
    
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
            
            #print(len(wis))
            
            #print('wis')
            #print(wis)
            
            xis = np.array(xi[-1])
            
            #print(len(xis))
            
            #print('xis')
            #print(xis)
            
            value2 = sum(wis*xis)
            #print('val')
            #print(value2)
            
            value = 1/(1+math.e**(-value2))    #sigmoid function
            
            values.append(value)
                
        '''if k == len(layer_sizes)-2 and flag == 1:
            
            #print('hello')
            max_val = values.index(max(values))
            
            #print(max_val)
            
            #print(values)
            
            #print(max(values))
            
            
            if max_val == 1:
                
                #Since we pull from xi [0] this says that the first class max_val[0] is postitive and the second class is negetive
                
                xi.append([0,1])#Might be backwards will see in graph if so.
            else:
                xi.append([1,0])    
        else:'''        
        xi.append(values) 

    
    
    #print('All vals')
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
    
    #print(label[0])
    #print()
    
    deltas = []
    categorical_gross_entropy1 = ((label[0]/xis[-1][0])-((1-label[0])/(1-xis[-1][0])))
    categorical_gross_entropy2 = ((label[1]/xis[-1][1])-((1-label[1])/(1-xis[-1][1])))
    
    delta1 = categorical_gross_entropy1*xis[-1][0]*(1-xis[-1][0])
    delta2 = categorical_gross_entropy2*xis[-1][1]*(1-xis[-1][1])
    
    deltas.append([delta1,delta2])
    
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
     
def make_net2(layer_sizes):  
    mlp_init = []
    
    #Build the weights for the encoder (before sigma and mu)
    for k in range(len(layer_sizes)-1):
        random_numbers = [[random.random() for _ in range(layer_sizes[k])]for _ in range(layer_sizes[k+1])]
        mlp_init.append(random_numbers)
    
    #print('ehllo')
    #print(mlp_init)
    
    return mlp_init   
     
import seaborn as sns


########################
#Get the behavior data
########################


layer_sizes = [latent_space_size,5,5,2]
epochs = 100
eta = 0.0001

mlp_weights = make_net2(layer_sizes)
#print(mlp_weights)


#Get the data from the latent space



train_data_holder = graph_data


labels = []
#One hot encode for label
for asd in button_press:
    if asd == 1:
        labels.append([1,0])
    elif asd == 0:
        labels.append([0,1])

print(labels)     
        
#print('training_len')
#print(train_data_holder)

for ep in range(epochs):
    #Track loss
    
    print(ep)
    
    training_output = []
    loss = []
    for k in range(len(train_data_holder)):
        #Forwards Propegate
        xis = forward2(train_data_holder[k], mlp_weights,layer_sizes,0)
        
        
        
        #print(len(train_data_holder))
        
        #print(button_press[k])
        
        
        
        #Print Loss
        
        #print(labels[k][0]*math.log(xis[-1][0]))
        
        loss.append(-(labels[k][0]*math.log(xis[-1][0])+(1-labels[k][0])*math.log(1-xis[-1][0])))
        #Backwards Propegate
        backprop2(xis,labels[k],mlp_weights,layer_sizes,eta, epochs)
        
        #print(mlp_weights)
        training_output.append(xis[-1])
        
        #print(training_output)
        
    print(np.mean(loss))


test_data_holder = graph_data2

#Testing
output = []
for tes in range(len(test_data_holder)):

    #print(test_data_holder[tes])

    xis = forward2(test_data_holder[tes], mlp_weights,layer_sizes,1)
    
    #print(xis)
    output.append(xis[-1])

##################################################################
##################################################################
##################################################################
count0 = 0
count1 = 0
count2 = 0
count3 = 0
#Calculate confusion matrix



print('stuff')
print(len(button_press))
print(len(training_output))

maxer = []
for jk in training_output:
    decision = round(jk[0])
    maxer.append(decision)

training_output = maxer

maxer = []
for jk in output:
    decision = round(jk[0])
    maxer.append(decision)

output = maxer


for hg in range(len(button_press)-2):
    #TP
    if training_output[hg] == 1 and button_press[hg] == 1:
        count0 = count0 + 1
    
    #FP
    if training_output[hg] == 1 and button_press[hg] == 0:
        count1 = count1 + 1
    
    #FN
    if training_output[hg] == 0 and button_press[hg] == 1:
        count2 = count2 + 1
    
    #TN    
    if training_output[hg] == 0 and button_press[hg] == 0:
        count3 = count3 + 1
  
plt.figure(1)
plt.plot(button_press, label='Target Button Presses')
plt.plot(training_output, label='Predicted Anticipation Signal')
plt.legend(loc='upper right')

# Customize the plot with labels and title
plt.xlabel('Time')
plt.ylabel('Button Press: 0 off ; 1 on')
plt.title('Classification: VAE')


cf_matrix = np.asarray([[count0,count1],[count2,count3]])


plt.figure(2)

  
sns.heatmap(cf_matrix, annot=True,cmap='Blues')






##################################################################
##################################################################
##################################################################
count0 = 0
count1 = 0
count2 = 0
count3 = 0



button_press2 = [int(x) for x in button_press2]

print(button_press2)
print(output)

#Calculate confusion matrix
for hg in range(len(button_press2)-2):
    #TP
    if output[hg] == 1 and button_press2[hg] == 1:
        count0 = count0 + 1
    
    #FP
    if output[hg] == 1 and button_press2[hg] == 0:
        count1 = count1 + 1
    
    #FN
    if output[hg] == 0 and button_press2[hg] == 1:
        count2 = count2 + 1
    
    #TN    
    if output[hg] == 0 and button_press2[hg] == 0:
        count3 = count3 + 1
  
plt.figure(3)
plt.plot(button_press2, label='Target Button Presses')
plt.plot(output, label='Predicted Anticipation Signal')
plt.legend(loc='upper right')

# Customize the plot with labels and title
plt.xlabel('Time')
plt.ylabel('Button Press: 0 off ; 1 on')
plt.title('Classification: VAE')


cf_matrix = np.asarray([[count0,count1],[count2,count3]])


plt.figure(4)

  
sns.heatmap(cf_matrix, annot=True,cmap='Oranges')


plt.show()


##################################################################
##################################################################
##################################################################




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



