import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import random
import os

def data_reading(file_name):
    data = pd.read_csv(file_name)
    data_numerical = data[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']].values
    data_combined = data[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Sex', 'ChestPainType']].values
    data_labels = data[['HeartDisease']].values

    #print("Number of samples: %i" % data_numerical.shape[0])
    #print("Number of numerical features: %i" % data_numerical.shape[1])
    #print("Number of combined features: %i" % data_combined.shape[1])
    return data_numerical, data_combined, data_labels

#a)

class NearestNeighborRegressor:
    
    def __init__(self, n_neighbors = 3):
        """
        Initializes the model.
        
        Parameters
        ----------
        n_neighbors : The number of nearest neigbhors (default 1)
        weights : weighting factors for numerical and categorical features
        """
        
        self.n_neighbors = n_neighbors
        self.type = 'unknown'
        self.predictions = []
        self.closestNeighbors = []
    


    def fit(self, X, t, type = 'numerical', weights = [1, 1]):
        """
        Fits the nearest neighbor regression model.
        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of labels [n_samples]
        type: Could be 'numerical' or 'combined'
        weights: coefficients that are used to be 
        """ 
        #.....
    
    def predict(self, trainFeatures, trainLabels, testFeatures, typeDistance, weights):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of length n_samples
        """         

        for testFeat in testFeatures:
            classList = []
            self.LocateNeighbors(trainFeatures, trainLabels, testFeat, self.n_neighbors, typeDistance, weights)
            for neighbor in self.closestNeighbors:
                classList.append(neighbor[1][0])
            self.predictions.append(max(set(classList), key=classList.count))


            
    def LocateNeighbors(self, trainFeatures, trainLabels, testFeatures, nNeighbors, typeDistance, weights):
        distanceList = []
        for i in range(trainFeatures.shape[0]):
            if (typeDistance == "numerical"):
                distance = self.__numericalDistance(testFeatures, trainFeatures[i])
                distanceList.append((distance, trainLabels[i]))
                continue
            if (typeDistance == "mixed"):
                distance = self.__mixedDistance(testFeatures, trainFeatures[i], weights)
                distanceList.append((distance, trainLabels[i]))
        distanceList.sort()

        self.closestNeighbors = distanceList[:nNeighbors]


    def __numericalDistance(self, p, q):
        """
        Computes the Euclidean distance between 
        two points.
        """
        #.....

        distance = np.linalg.norm(p - q)
        return distance

    def IsCategoricalSame(self, p, q):
        if (not np.array_equal(p, q)):
            return 1
        else:
            return 0

    def __mixedDistance(self, p, q, weights):
        """
        Computes the distance between 
        two points via the pre-defined matrix.
        """
        
        #.....

        distance = (weights[0] * np.linalg.norm(p[0:4] - q[0:4])) + (weights[1] * self.IsCategoricalSame(p[5:6], q[5:6]))
        
        return distance

    def rmse(self, testLabels):
        """ Computes the RMSE for two
        input arrays 't' and 'tp'.
        """
        return np.sqrt(np.mean((self.predictions-testLabels)**2))


    def accuracy(self, testLabels):
        """ Computes the RMSE for two
        input arrays 't' and 'tp'.
        """
        accuracy = 0
        for i in range(len(testLabels)):
            if (self.predictions[i] == testLabels[i]):
                accuracy += 1
        return accuracy/len(testLabels)


class DataComponent:

    #TODO: Might have to change filepath
    def __init__(self):
        self.dataPath = "C:/Users/adamf/Desktop/Datalogi/year2/Semester1/Modelling and Analysis of Data Exam/exam_2022/data/"

    def _PCA(data):
        #---put your code here
        dataMean = np.mean(data)
        newData = (data - dataMean).T
        covMatrix = np.cov(newData)
        PCevals, PCevecs = np.linalg.eigh(covMatrix)
        PCevals = np.flip(PCevals, 0)
        PCevecs = np.flip(PCevecs, 1)
        
        return PCevals, PCevecs

    def _transformData(features, PCevecs):
        return np.dot(features,  PCevecs[:, 0:2])

    def getData(self, filePathTrain, filePathTest):        
        self.trainData_numerical, self.trainData_combined, self.trainData_labels = data_reading(self.dataPath + filePathTrain)
        self.testData_numerical, self.testData_combined, self.testData_labels = data_reading(self.dataPath + filePathTest)


NNRegressor = NearestNeighborRegressor()
dataComp = DataComponent()
dataComp.getData("heart_simplified_train.csv", "heart_simplified_test.csv")

kList = []
accuracyList = []

for k in range(1, 11):
    NNRegressor.n_neighbors = k
    kList.append(k)
    print("Current Neighbors: ", k, end="; ")
    NNRegressor.predict(dataComp.trainData_numerical, dataComp.trainData_labels, dataComp.testData_numerical, "numerical", [1, 1])
    print("accuracy_score: ", NNRegressor.accuracy(dataComp.testData_labels), end="; ")    
    accuracyList.append(NNRegressor.accuracy(dataComp.testData_labels))
    print("rmse_score: ", NNRegressor.rmse(dataComp.testData_labels), end=" ")
    NNRegressor.predictions = []
    NNRegressor.closestNeighbors = []
    print("")


print("--------------------------------------------------------------------------------")
#b)
NNRegressor.predictions = []
NNRegressor.closestNeighbors = []

catWeightList = [0.01, 0.025, 0.05, 0.1]
fixedNeighbors = 5
NNRegressor.n_neighbors = fixedNeighbors

for catWeight in catWeightList:
    NNRegressor.predict(dataComp.trainData_combined, dataComp.trainData_labels, dataComp.testData_combined, "mixed", [1, catWeight])
    print("Current weights: " + str([1, catWeight]), end="; ")
    print("accuracy_score: ", NNRegressor.accuracy(dataComp.testData_labels), end="; ")  
    print("rmse_score: ", NNRegressor.rmse(dataComp.testData_labels), end=" ")
    print("")

    NNRegressor.predictions = []
    NNRegressor.closestNeighbors = []
    
    

plt.scatter(kList, accuracyList)
plt.xlim(min(kList) - 0.1, max(kList) + 0.1)
plt.ylim(min(accuracyList) - 0.1, max(accuracyList) + 0.1)
plt.show()






