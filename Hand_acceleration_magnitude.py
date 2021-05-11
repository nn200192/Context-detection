import numpy as np
import pandas as pd
import pickle
import sklearn
from numpy import unique
from numpy import where
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from matplotlib import pyplot
import matplotlib.pyplot as plt
import random
import math
import csv

random.seed(1)


#Get real time data for acceleration 
#Find three axes of acceleration
acceX = np.loadtxt("linacce_x.txt", skiprows=1, delimiter='\t')
acceY = np.loadtxt("linacce_y.txt", skiprows=1, delimiter='\t')
acceZ = np.loadtxt("linacce_z.txt", skiprows=1, delimiter='\t')

#Combine into 3D array
acceleration = np.column_stack((acceX,acceY,acceZ))
magnitude = []


#Range calculate magnitude
for i in range(100):
    x_squared = abs(acceleration[i][0]) ** 2
    y_squared = abs(acceleration[i][1]) ** 2
    z_squared = abs(acceleration[i][2]) ** 2
    magnitude_val = math.sqrt(x_squared + y_squared + z_squared)
    magnitude.append(magnitude_val)

#Convert float to int for the magnitude
magnitude = [int(item) for item in magnitude]
print(magnitude)








    
    
    




