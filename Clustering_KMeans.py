import numpy as np
import pandas as pd
import pickle
import sklearn
from numpy import unique
from numpy import where
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib import pyplot
from numpy.random import default_rng
import matplotlib.pyplot as plt
import random
import math
import csv

np.set_printoptions(suppress=True)
np.random.seed(0)



'''
#Walking
mu, sigma = 1.5, 0.3 # mean and standard deviation
s = np.random.default_rng(seed=1).normal(mu, sigma, 1000)
np.random.shuffle(s)
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        linewidth=2, color='r')

'''


#SPEED
#Combine Walking, Running, Cycling, Car, Train respectively, 1000 data in each
mean_std_speed = [1.5, 0.3, 5, 0.3, 7, 0.3, 20, 2, 56, 2]
speed = []
count=0
for i in range(5):
    s = np.random.default_rng(seed=1).normal(mean_std_speed[count], mean_std_speed[count+1], 1000)
    np.random.shuffle(s)
    count = count+2
    for j in range(1000):
        speed.append(s[j])
        
        

#HEART RATE
#Combine Walking, Running, Cycling, Car, Train respectively, 1000 data in each
mean_std_heart = [120, 5, 135, 5, 135, 5, 85, 4, 85, 4]
heart_rate=[]
count=0
for i in range(5):
    s = np.random.default_rng(seed=1).normal(mean_std_heart[count], mean_std_heart[count+1], 1000)
    np.random.shuffle(s)
    count = count+2
    for j in range(1000):
        heart_rate.append(s[j])


#HAND ACCELERATION MAGNITUDE
#Combine Walking, Running, Cycling, Car, Train respectively, 1000 data in each
mean_std_magnitude = [8, 2, 8, 2, 0, 1, 0, 1, 0, 1]
magnitude=[]
count=0
for i in range(5):
    s = np.random.default_rng(seed=1).normal(mean_std_magnitude[count], mean_std_magnitude[count+1], 1000)
    np.random.shuffle(s)
    count = count+2
    for j in range(1000):
        magnitude.append(abs(s[j]))



#AMPLITUDE
#Combine Walking, Running, Cycling, Car, Train respectively, 1000 data in each
mean_std_amplitude = [0.8, 0.2, 0.8, 0.2, 0, 0.05, 0, 0.05, 0, 0.05]
amplitude=[]
count=0
for i in range(5):
    s = np.random.default_rng(seed=1).normal(mean_std_amplitude[count], mean_std_amplitude[count+1], 1000)
    np.random.shuffle(s)
    count = count+2
    for j in range(1000):
        amplitude.append(abs(s[j]))


#TIME OF DAY
#Combine Walking, Running, Cycling, Car, Train respectively, 1000 data in each
mean_std_time=[6.5, 0.5, 19, 0.5, 6.5, 0.5, 19, 0.5, 6.5, 0.5, 19, 0.5,
               8, 0.2, 17.5, 0.2, 8, 0.2, 17.5, 0.2]                         
time=[]
count=0
for i in range(5):
    s1 = np.random.default_rng(seed=i).normal(mean_std_time[count], mean_std_time[count+1], 500)
    np.random.shuffle(s1)
    count = count+2
                         
    s2 = np.random.default_rng(seed=i).normal(mean_std_time[count], mean_std_time[count+1], 500)
    np.random.shuffle(s2)
    count = count+2

    s = np.concatenate((s1,s2),axis=0)
    np.random.shuffle(s)
    for j in range(1000):
        time.append(s[j])


#TRUE CLASSFICATION
context=["Walking","Running","Cycling","Car","Train"]
true_class=[]
for i in range(5):
    for j in range(1000):
        true_class.append(context[i])
    

#Join arrays together
Y = np.column_stack((speed, heart_rate, magnitude, amplitude, time, true_class))
np.random.shuffle(Y)
X = Y[:,:5]
X = X.astype(float)
X = np.around(X, decimals=1)



# define the model
model = KMeans(n_clusters=10,random_state=0)
#model = AffinityPropagation(damping=0.8, random_state=0)

# fit the model
model.fit(X)


# assign a cluster to each example
yhat = model.predict(X)
#print(yhat)


# retrieve unique clusters
clusters = unique(yhat)


#Set array axis x, y, z
x_vals = X[:,0]
y_vals = X[:,1]
z_vals = X[:,2]


# Generate the graph Speed vs. Heart rate
speed_heart = plt.scatter(X[:,0], X[:,1],c=model.labels_.astype(float))
speed_heart = plt.xlabel('Speed (m/s)')
speed_heart = plt.ylabel('Heart rate (BPM)')
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5);


#Generate the graph Speed vs. Acceleration
#speed_acceleration = plt.scatter(X:,0], X[:,2],c=model.labels_.astype(float))
#speed_acceleration = plt.xlabel('Speed (m/s)')
#speed_acceleration = plt.ylabel('Acceleration (m/s2)')


#Generate the graph Heart rate vs. Acceleration
#heart_acceleration = plt.scatter(X:,1], X[:,2],c=model.labels_.astype(float))
#heart_acceleration = plt.xlabel('Heart rate (BPM)')
#heart_acceleration = plt.ylabel('Acceleration (m/s2)')

# Generate the graph Heart rate vs. Time of day
#speed_heart = plt.scatter(X[:,1], X[:,4],c=model.labels_.astype(float))
#speed_heart = plt.xlabel('Heart rate (m/s)')
#speed_heart = plt.ylabel('Time of day (BPM)')
#centers = model.cluster_centers_
#plt.scatter(centers[:, 1], centers[:, 4], c='black', s=100, alpha=0.5);




#Set up for printing to CSV
array_print=np.column_stack((X, yhat, Y[:,5]))
array_print = sorted(array_print, key=lambda x: x[5])


#Tie a number of classification to the name by
#counting the occurences of each context in a classification
context=["Walking","Running","Cycling","Car","Train"]
predicted=[]
context_count=[0,0,0,0,0]
count=0
for element in array_print:
    if int(element[5]) == int(count):
        context_index=context.index(element[6])
        context_count[context_index] = context_count[context_index] + 1 #increment count
    else:
        max_index = context_count.index(max(context_count)) #index with max count
        predicted.append(context[max_index])
        context_count=[0,0,0,0,0];
        count=count+1
        context_index=context.index(element[6])
        context_count[context_index] = context_count[context_index] + 1

max_index = context_count.index(max(context_count)) #index with max count
predicted.append(context[max_index])
#print(predicted)

predicted_class=[]
count=0
for element in array_print:
    predicted_class.append(predicted[int(element[5])])


array_print=np.column_stack((array_print, predicted_class))

#Print to a CSV file
with open("test_KMeans.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter=",")
    writer.writerow(["Speed (m/s)", "Heart rate (BPM)", "Hand Acceleration (m/s^2)", "Hand Amplitude (m)", "Time of Day", "Classification", "True class", "Predicted class"])  # write header
    writer.writerows(array_print)


#Create confusion matrix and report
true_values=array_print[:,6]
predicted_values=predicted_class
labels=["Walking","Running","Cycling","Car","Train"]
confusion_matrix = pd.DataFrame(
    metrics.confusion_matrix(true_values, predicted_values, labels=labels), 
    index=labels, 
    columns=labels
)
print(confusion_matrix)
print(metrics.classification_report(true_values, predicted_values, digits=3))


#3D plot
fig = plt.figure()
plot3d = fig.add_subplot(111, projection='3d')

# Generate the values
plot3d.scatter(X[:,0], X[:,1], X[:,2], c=model.labels_.astype(float))
plot3d.set_xlabel('Speed (m/s)')
plot3d.set_ylabel('Heart rate (BPM)')
plot3d.set_zlabel('Hand Acceleration (m/s)')


'''
plot3d.scatter(X[:,0], X[:,1], X[:,3], c=model.labels_.astype(float))
plot3d.set_xlabel('Speed (m/s)')
plot3d.set_ylabel('Heart rate (BPM)')
plot3d.set_zlabel('Amplitude (m)')
'''

'''
plot3d.scatter(X[:,0], X[:,1], X[:,4], c=model.labels_.astype(float))
plot3d.set_xlabel('Speed (m/s)')
plot3d.set_ylabel('Heart rate (BPM)')
plot3d.set_zlabel('Time')
'''

'''
plot3d.scatter(X[:,1], X[:,2], X[:,3], c=model.labels_.astype(float))
plot3d.set_xlabel('Heart rate (BPM)')
plot3d.set_ylabel('Magnitude (m/s^2)')
plot3d.set_zlabel('Amplitude (m)')
'''

'''
plot3d.scatter(X[:,2], X[:,3], X[:,4], c=model.labels_.astype(float))
plot3d.set_xlabel('Magnitude (m/s^2)')
plot3d.set_ylabel('Amplitude (m)')
plot3d.set_zlabel('Time (m)')
'''

plt.show()


    
    
    




