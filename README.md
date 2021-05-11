# Context-detection

The goal of this code is to determine the context of users based on a list of available axes from the embedded device’s sensors. An activity of the user, by other means, what he/she is doing, such as sleeping, eating, driving, or walking is referred to as a context. A category or metric values of collected data from sensors, such as a user’s heart rate, location, speed, or acceleration is refered to as an axis. An axis is the type of data collected from various sensors. For instance, heart rate axis and location axis are collected from PPG sensor and GPS respectively. Gaussian normal distribution is used to simulate relevant datasets for applying Machine Learning algorithm.

AxisTable.xlsx contains information of various axes that can be collected from sensors, which can be referenced for simulated data generation.

Five axes and five contexts have been selected for our context detection research.

Chosen axes: Speed, heart rate, hand acceleration, hand amplitude, Time of Day

Chosen contexts: Walking, Running, Cycling, Car, Train
  
 
To calculate the magnitude of hand acceleration from three x, y, and z directions in the linnace* files, please run this code: python Hand_acceleration_magnitude.py
  
To better visualize graphical results of Gaussian normal distribution of datasets, please run this code: python gaussian.py

To generate simulated Gaussian datasets for the chosen axes and contexts, appply KMeans Clustering Algorithm on the simulated datasets, output Excel and graphical results of classification, and output assessment of performance report, run this code: python Clustering_KMeans.py
  
The test_KMeans.csv is a result of the classification after running Clustering_KMeans.py
  
