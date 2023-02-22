"""
Author: Yung Kipreos
Date: 22 February 2023

This machine learning project was written for Astronomy 9506S, taught by Denis Vida (Western University).
The purpose of this program is to search for the secondary stream found in Borovička, 
Spurný, and Shrbenŷ (2022) (https://www.aanda.org/articles/aa/pdf/2022/11/aa44184-22.pdf). This paper used 
optical data of Geminid fireballs and found that there are two populations in their data with different
characteristics. This project searches for these populations using radar data from CMOR. This tests whether
or not we see these populations in radar data, which is a smaller size regime than optical data. The 
clustering machine learning algorithms used in this project are DBSCAN, HDBSCAN, and agglomerative heirarchy 
clustering. 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
import hdbscan
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import kneed
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import os
import fnmatch


"""
This function imports the data as a Pandas DataFrame from a local txt file. 

Input: 
    filename = (string) the name of the local file to get the data from
Output:
    new_data = (Pandas DataFrame) the new dataframe
"""

def import_data(filename):

    new_data = pd.read_csv(filename, usecols=[29,31,33,39,43,45], dtype=np.float, names=["a", "e", "i", "vel", "ll0", "beta"], header = None, sep = ' ')

    return new_data



"""
This function creates a big correlation plot of each of the parameters in data with each other (semi-major 
axis, eccentricity, inclination, sun-centered ecliptic longitude, sun-centered ecliptic latitude).

Input:
    data = (Pandas DataFrame) This DataFrame contains the following information about 
        the meteor observations: semi-major axis, eccentricity, inclination, geocentric velocity, sun-centered 
        ecliptic longitude, and sun-centered ecliptic longitude.
"""
def correlation_plot(data):

    df = data[['a','i','e', 'vel']]
    sns.pairplot(df, kind="scatter")
    plt.show()



"""
Note: To choose the epsilon for DBSCAN, I followed the tutorials from https://towardsdatascience.com/machine-
learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc and
https://towardsdatascience.com/detecting-knee-elbow-points-in-a-graph-d13fc517a63c

This function chooses the best eps parameter for DBSCAN. Eps is the furthest distance two data points have
to be to each other to be considered to be a part of the same cluster.  When the distance (eps) is plotted 
against the data points (sorted by distance), the point of maximum curvature in the plot tells you the ideal 
values for eps.

Input: 
    X_train: (Pandas DataFrame)the meteor data to apply DBSCAN on (semi-major axis, inclination, eccentricity)
Output: 
    eps: (float) The furthest distance that two data points have to be from each other that they will be 
        considered part of the same cluster.
"""
def getEps(X_train):

    # Making plot
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X_train)
    distances, indices = nbrs.kneighbors(X_train)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    print ('dsitances = ', distances)
    i = np.arange(len(distances))

    # Finding elbow point of the plot
    kneedle = kneed.KneeLocator(i,distances, curve = "convex", direction = "increasing", online= "False", 
                                S = 1.0, interp_method='polynomial')
    knee_point = kneedle.knee #elbow_point = kneedle.elbow
    print('Knee: ', knee_point) #print('Elbow: ', elbow_point)
    kneedle.plot_knee() 
    eps = distances[kneedle.knee]
    
    # Plot results
    plt.plot(distances)
    plt.title("Sklearn Nearest Neighbors Implementation")
    plt.ylabel("eps")
    plt.xlabel("Data points (sorted by distance)")
    plt.show()
    plt.clf()
    
    return eps



"""
Note: I followed KD Nugget's tutorial on DBSCAN clustering for this function. 
(https://www.kdnuggets.com/2022/08/implementing-dbscan-python.html#:~:text=What%20is%20DBSCAN%3F,cluste
ring%20generates%20spherical%2Dshaped%20clusters.)

This function applies DBSCAN to the meteor data (inclination angle, eccentricity, and semi-major axis)
to identify clusters within the data. After applying DBSCAN this function plots the results. The min_samples
parameter of DBSCAN should be chosen to fit the data.

Input:
    data = (Pandas DataFrame) This DataFrame contains the following information about the meteor 
        observations: semi-major axis, eccentricity, inclination, geocentric velocity, sun-centered 
        ecliptic longitude, and sun-centered ecliptic longitude.
"""
def dbscanClustering(data):

    # Scale the data
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Apply DBSCAN
    X_train = data[['i', 'e', 'a']]
    epsilon = getEps(X_train)
    clustering = DBSCAN(eps=epsilon, min_samples=100).fit(X_train)
    DBSCAN_dataset = X_train.copy()
    DBSCAN_dataset.loc[:,'Cluster'] = clustering.labels_ 
    DBSCAN_dataset.Cluster.value_counts().to_frame()

    # Plot DBSCAN results
    outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster']==-1]
    fig2, (axes) = plt.subplots(1,2,figsize=(12,5))
    sns.scatterplot('e', 'a',

                    data=DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1],

                    hue='Cluster', ax=axes[0], palette='Set2', legend='full', s=100)
    sns.scatterplot('i', 'a',

                    data=DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1],

                    hue='Cluster', palette='Set2', ax=axes[1], legend='full', s=100)
    axes[0].scatter(outliers['e'], outliers['a'], s=10, label='outliers', c="k")
    axes[1].scatter(outliers['i'], outliers['a'], s=10, label='outliers', c="k")
    axes[0].set_ylabel("Semi-major axis (AU)")
    axes[1].set_ylabel("Semi-major axis (AU)")
    axes[0].set_xlabel("Eccentricity")
    axes[1].set_xlabel("Inclination (deg)")
    axes[0].legend()
    axes[1].legend()
    plt.setp(axes[0].get_legend().get_texts(), fontsize='12')
    plt.setp(axes[1].get_legend().get_texts(), fontsize='12')
    plt.show()



"""
This function checks whether or not the clusters found by the HDBSCAN application is actually the secondary 
stream we're looking for. This is determined by comparing the velocities for each cluster population. 
This function doesn't make assumptions about the number of clusters found.

Input:
    clusterer: (hdbscan.hdbscan_.HDBSCAN object) holds the HDBSCAN information
    data = (Pandas DataFrame) the meteor observation data
"""
def isStream(clusterer, data):

    cluster_types = np.asarray(np.unique(clusterer.labels_))
    j = 0
    while (j < len(cluster_types)):
        dff = pd.DataFrame()

        i = 0
        while (i < len(clusterer.labels_)):
            if (clusterer.labels_[i] == cluster_types[j]):
                dff = dff.append(data.iloc[i])
            i += 1
    
        print ("Velocity of cluster " + str(cluster_types[j]), " = ", dff['vel'].mean())
        j += 1
   


"""
Note: I followed https://michael-fuchs-python.netlify.app/2020/06/20/hdbscan/ tutorial on HDBSCAN clustering.

This function uses HDBSCAN to search for clustering in the meteor data, then it plots the results. 
Additionally, it tests if the results of the HDBSCAN clustering is the secondary stream that we're 
searching for. 

Input:
    hdbscan_data = (Pandas DataFrame) Data for HDBSCAN to be applied to (eccentricity and semi-major 
        axis columns for example)
    names = (list of strings) the x and y axis labels for the plot
    full_data = (Pandas DataFrame) the meteor observation data with all columns
"""
def hdbscanClustering(hdbscan_data, names, full_data):

    # Scale the data
    scaler = MinMaxScaler()
    hdbscan_data = pd.DataFrame(scaler.fit_transform(hdbscan_data), columns=hdbscan_data.columns)

    # Compute the clustering using HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
    print ("CLUSTERER = ", type(clusterer))
    clusterer.fit(hdbscan_data)
    isStream(clusterer, full_data)
    
    # Plot the results
    plt.scatter(hdbscan_data.iloc[:,0], hdbscan_data.iloc[:,1], c=clusterer.labels_, cmap='rainbow', s=20, alpha = 0.2)
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.title("HDBSCAN Clustering")
    plt.show()



"""
This function produces the dendrogram plot for the heirarchy clustering method. 

Input:
    data = (Pandas DataFrame) the meteor observation data
"""
def plotDendrogram(data):

    plt.figure(figsize=(10, 7))  
    plt.title("Dendrogram")  
    dend = shc.dendrogram(shc.linkage(data, method='ward'))
    plt.axhline(y=11, color='r', linestyle='--')
    plt.xlabel("Samples")
    plt.ylabel("Distance between samples")
    plt.show()
    plt.clf()



"""
Note: I followed https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/ 's
 tutorial on how to perform hierarchy clustering on a dataset.

This function applies heirarchy clustering on the meteor dataset. First the data is scaled, then a
dendrogramis plotted, then the results are plotted and the velocity of each clustering is calculated.

Input:
    full_data = (Pandas DataFrame) the full meteor data with all columns
    heirarchy_data = (Pandas DataFrame) the columns of meteor data that the heirarchy clustering 
        should be applied to
    names = (list of strings) A list of the names of the data the clustering is being applied to 
        (Semi-major axis etc)
"""
def heirarchy(full_data, heirarchy_data, names):
    
    # Scale the data
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(heirarchy_data), columns=heirarchy_data.columns)
    #data = pd.DataFrame(scaler.fit_transform(data[['a', 'i']]), columns=data[['a', 'i']].columns)

    # Plot the Dendrogram
    plotDendrogram(data)

    # Apply hierarchy clustering to the meteor data
    model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    model.fit(data)

    # Get labels for plotting colors
    labels = model.labels_
    labels = np.asarray(labels)
    i = 0
    labels_list = []
    while(i < len(labels)):
        labels_list.append(labels[i])
        i += 1

    # Plot the results of the hierarchy clustering
    plt.figure(figsize=(10, 7))  
    plt.title("Hierarchical Clustering")
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.scatter(heirarchy_data.iloc[:,0], heirarchy_data.iloc[:,1], c=np.asarray(labels_list) )
    plt.show()

    isStream(model, full_data)



if __name__ == "__main__":

    # ------------- Defining Directories and Shower Info -------------- #
    data_directory = "./../data/cleaned_orb_data/GEM_meteors/"
    sl_start = 239 # Beginning of the shower
    #sl_start = 260 #max
    sl_end = 271 # End of the shower
    all_slon = True # False if you want to access the data per solar longitude
    os.chdir(data_directory)

    slon = sl_start
    data = pd.DataFrame()
    while (slon <= sl_end): #Custom

        for filename in os.listdir('.'): 
            if fnmatch.fnmatch(filename, '*' + str(slon) + '*'):
                print("SL = ", slon)
                if (all_slon == True):
                    new_data = import_data(filename)
                    data = pd.concat([data, new_data], ignore_index=True)
                else:
                    data = import_data(filename)
                    #dbscanClustering(data)
                    #hdbscanClustering(data[['a', 'i']].copy(), ["Semi-major axis", "Inclination"], data)
                    #hdbscanClustering(data[['a', 'e']].copy(), ["Semi-major axis", "Eccentricity"], data)
                    #correlation_plot(data)
                    #streamletSearch(data)
                    #principalComponentAnalysis(data)
                    #heirarchy(data, data[['a', 'i']].copy(), ["Semi-major axis", "Inlcination"])
                    #heirarchy(data, data[['a', 'i']].copy(), ["Semi-major axis", "Eccentricity"])
            
        slon += 1

    #dbscanClustering(data)
    hdbscanClustering(data[['a', 'i']].copy(), ["Semi-major axis", "Inclination"], data)
    hdbscanClustering(data[['a', 'e']].copy(), ["Semi-major axis", "Eccentricity"], data)
    #correlation_plot(data)
    #streamletSearch(data)
    #principalComponentAnalysis(data)
    #heirarchy(data, data[['a', 'i']].copy(), ["Semi-major axis", "Inlcination"])
    #heirarchy(data, data[['a', 'i']].copy(), ["Semi-major axis", "Eccentricity"])
