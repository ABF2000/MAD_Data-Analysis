import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random


def data_reading(file_name):
    data = pd.read_csv(file_name)
    housingData_all = data[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']].values
    loopList = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']
    meanStdList = np.array([])
    for i in range(len(loopList)):
        tempData = data[loopList[i]].values
        featureMean = np.mean(tempData)
        featureDeviation = np.std(tempData)
        meanStdList = np.append(meanStdList, [featureMean, featureDeviation])
    meanStdList = meanStdList.reshape(9, 2)
    return housingData_all, meanStdList


#TODO: Might have to change filepath
housingData, housingDataMeanStd = data_reading("C:/Users/adamf/Desktop/Datalogi/year2/Semester1/Modelling and Analysis of Data Exam/exam_2022/data/housing.csv")

#a)

for j in range(housingData.shape[1]):
    for i in range(housingData.shape[0]):
        housingData[i][j] = (housingData[i][j] - housingDataMeanStd[j][0])/housingDataMeanStd[j][1]

#b)

# Generates two unique indices
def GenerateInitialIndices(dataSize):
    K0CentroidIndex = random.randint(0, dataSize - 1)
    while True:
        K1CentroidIndex = random.randint(0, dataSize - 1)
        if (K0CentroidIndex != K1CentroidIndex):
            return K0CentroidIndex, K1CentroidIndex

# Calculates the euclidean distance
def EuclideanDistance(p, q):
    distance = np.linalg.norm(p - q)
    return distance

# Given data and two centroids, this function will for all data-samples
# determine which of the two centroids, they are closest to
def DetermineClusters(data, firstCentroid, secondCentroid):
    firstCentroidCluster = np.array([])
    secondCentroidCluster = np.array([])
    for dataSample in data:
        distDataSampleFirst = EuclideanDistance(firstCentroid, dataSample)
        distDataSampleSecond = EuclideanDistance(secondCentroid, dataSample)
        if (distDataSampleFirst < distDataSampleSecond):
            firstCentroidCluster = np.append(firstCentroidCluster, dataSample)
        else:
            secondCentroidCluster = np.append(secondCentroidCluster, dataSample)
    firstCentroidCluster = firstCentroidCluster.reshape(int(firstCentroidCluster.shape[0]/9), 9)
    secondCentroidCluster = secondCentroidCluster.reshape(int(secondCentroidCluster.shape[0]/9), 9)
    return firstCentroidCluster, secondCentroidCluster

# Find mean-centroid of a cluster
def FindMeanCentroid(cluster):
    tempArray = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(cluster.shape[0]):
        tempArray = np.add(tempArray, cluster[i])
    newCentroid = tempArray/cluster.shape[0]
    return newCentroid

# Main function, which runs recursively
def Hierarchical_kMeans(data, maxSubDivisions):
    K0InitialIndex, K1InitialIndex = GenerateInitialIndices(data.shape[0])
    K0Centroid = data[K0InitialIndex]
    K1Centroid = data[K1InitialIndex]
    K0PreviousCentroid = [0,0,0,0,0,0,0,0,0]
    K1PreviousCentroid = [0,0,0,0,0,0,0,0,0]
    while (not np.array_equal(K0Centroid, K0PreviousCentroid) and not np.array_equal(K1Centroid, K1PreviousCentroid)):
        K0CentroidCluster, K1CentroidCluster = DetermineClusters(data, K0Centroid, K1Centroid)
        K0PreviousCentroid = K0Centroid
        K1PreviousCentroid = K1Centroid
        K0Centroid = FindMeanCentroid(K0CentroidCluster)
        K1Centroid = FindMeanCentroid(K1CentroidCluster)
    if (maxSubDivisions == 0):
        return K0CentroidCluster, K1CentroidCluster
    else: 
        return np.array(Hierarchical_kMeans(K0CentroidCluster, maxSubDivisions - 1), dtype=object), np.array(Hierarchical_kMeans(K1CentroidCluster, maxSubDivisions - 1),dtype=object)

# Computes the intra-distance for a given cluster
def ComputeIntraDistance(cluster):
    intraDistance = 0
    for i in range(cluster.shape[0]):
        for j in range(cluster.shape[1]):
            centroid = FindMeanCentroid(cluster[i][j])
            for k in range(len(cluster[i][j])):
                intraDistance += EuclideanDistance(centroid, cluster[i][j][k])
    return intraDistance

# Test that creates 5 kMeans-tests and selects the best of those five
kMeansBestCluster = np.array(Hierarchical_kMeans(housingData, 1), dtype=object)
kMeansBestClusterIntraDistance = ComputeIntraDistance(kMeansBestCluster)

for i in range(4):
    tempCluster = np.array(Hierarchical_kMeans(housingData, 1), dtype=object)
    tempClusterIntraDistance = ComputeIntraDistance(tempCluster)
    if (kMeansBestClusterIntraDistance > tempClusterIntraDistance):
        kMeansBestCluster = tempCluster
        kMeansBestClusterIntraDistance = tempClusterIntraDistance

# Printing the sub-clusters for the best cluster
print("The number of samples in the best cluster are: "
         + str(kMeansBestCluster[0][0].shape[0])
         + ", "
         + str(kMeansBestCluster[0][1].shape[0]) 
         + ", " 
         + str(kMeansBestCluster[1][0].shape[0]) 
         + ", " 
         + str(kMeansBestCluster[1][1].shape[0]))

#c)

# Using PCA-class from sklearn to compute the principal components
def ComputePCA(data):
    pcaComponent = PCA(n_components=2)
    return pcaComponent.fit(data)

flatBestCluster = np.concatenate((kMeansBestCluster[0][0], kMeansBestCluster[0][1], kMeansBestCluster[1][0], kMeansBestCluster[1][1]))
PCA_flatBestClusterData = ComputePCA(flatBestCluster)


#d)


# Transforming the data with the previously found principal components
cluster002D = PCA_flatBestClusterData.transform(kMeansBestCluster[0][0])
cluster012D = PCA_flatBestClusterData.transform(kMeansBestCluster[0][1])
cluster102D = PCA_flatBestClusterData.transform(kMeansBestCluster[1][0])
cluster112D = PCA_flatBestClusterData.transform(kMeansBestCluster[1][1])

# This function is primarily just to split up the transformed clusters into an x-coord list and y-coord list
def GetClusterXAndY(leafCluster):
    xCoords = np.array([])
    yCoords = np.array([])
    for i in range(len(leafCluster)):
        xCoords = np.append(xCoords, leafCluster[i][0])
        yCoords = np.append(yCoords, leafCluster[i][1])
    return xCoords, yCoords

cluster002DxCoords, cluster002DyCoords = GetClusterXAndY(cluster002D)
cluster012DxCoords, cluster012DyCoords = GetClusterXAndY(cluster012D)
cluster102DxCoords, cluster102DyCoords = GetClusterXAndY(cluster102D)
cluster112DxCoords, cluster112DyCoords = GetClusterXAndY(cluster112D)

# This function just finds the mean point for each cluster
def Get2DClusterCentroidCoords(coordListX, coordListY):
     return np.mean(coordListX), np.mean(coordListY)

cluster002DCentroidxCoord, cluster002DCentroidyCoord = Get2DClusterCentroidCoords(cluster002DxCoords, cluster002DyCoords)
cluster012DCentroidxCoord, cluster012DCentroidyCoord = Get2DClusterCentroidCoords(cluster012DxCoords, cluster012DyCoords)
cluster102DCentroidxCoord, cluster102DCentroidyCoord = Get2DClusterCentroidCoords(cluster102DxCoords, cluster102DyCoords)
cluster112DCentroidxCoord, cluster112DCentroidyCoord = Get2DClusterCentroidCoords(cluster112DxCoords, cluster112DyCoords)


plt.scatter(cluster002DxCoords, cluster002DyCoords, c="red",s = 15)
plt.scatter(cluster002DCentroidxCoord, cluster002DCentroidyCoord, c="black")
plt.scatter(cluster012DxCoords, cluster012DyCoords, c="aqua",s = 15)
plt.scatter(cluster012DCentroidxCoord, cluster012DCentroidyCoord, c="black")
plt.scatter(cluster102DxCoords, cluster102DyCoords, c="green",s = 15)
plt.scatter(cluster102DCentroidxCoord, cluster102DCentroidyCoord, c="black")
plt.scatter(cluster112DxCoords, cluster112DyCoords, c="orange",s = 15)
plt.scatter(cluster112DCentroidxCoord, cluster112DCentroidyCoord, c="black")
plt.show()










    







    

        












