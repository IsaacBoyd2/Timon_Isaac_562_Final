#Goals

#1. Useing the raw data perform regression using MSE error loss function. (Can just take the encoder from the prvious model to do this)
#2a. Try using PCA as input
#2b. Try using VAE as input

#3. Record and report results.

#The bulk of the program will utilize 2 functions:
    #Forwards propegation
    #Backwards propegation

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

#Estimate instantaneous velocity by taking midpoint between velocities
inta_vel = (vel[1:len(vel)] - vel[0:len(vel)-1])/2

#
####Will need to randomize the index that we draw from.
#

########################
#Get the LFP data
########################

data_holder = []
num_channels = 5
for iiii in range(89,89+num_channels):
    chan = loadmat(f'CSC{iiii}.mat')
    data = chan['data'][0]
    #Account for the velocity calculation
    data = data[instruction_switch+1:len(data)-1]
    data_holder.append(data)
