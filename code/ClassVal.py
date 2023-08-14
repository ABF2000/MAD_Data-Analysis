import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random


def data_reading(file_name):
    data = pd.read_csv(file_name)
    data_numerical = data[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']].values
    data_combined = data[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Sex', 'ChestPainType']].values
    data_labels = data[['HeartDisease']].values

    #print("Number of samples: %i" % data_numerical.shape[0])
    #print("Number of numerical features: %i" % data_numerical.shape[1])
    #print("Number of combined features: %i" % data_combined.shape[1])
    return data_numerical, data_combined, data_labels


#TODO:Might have to change filepath
trainData_numerical, trainData_combined, trainData_labels = data_reading("C:/Users/adamf/Desktop/Datalogi/year2/Semester1/Modelling and Analysis of Data Exam/exam_2022/data/heart_simplified_train.csv")
validationData_numerical, validationData_combined, validationData_labels = data_reading("C:/Users/adamf/Desktop/Datalogi/year2/Semester1/Modelling and Analysis of Data Exam/exam_2022/data/heart_simplified_validation.csv")


#a)
trainData_combined_converted = trainData_combined

def catGenderConverter(genderString):
    if (genderString == "M"):
        return 1
    else:
        return 0

def catPainConverter(painTypeString, array):
    tempArray = array
    if (painTypeString == "ATA"):
        tempArray[5] = 1
    if (painTypeString == "NAP"):
        tempArray[6] = 1
    if (painTypeString == "ASY"):
        tempArray[7] = 1
    if (painTypeString == "TA"):
        tempArray[8] = 1
    return tempArray

trainData_combined_converted_new = np.array([])
for i in range(trainData_combined_converted.shape[0]):
    trainData_combined_converted[i][4] = catGenderConverter(trainData_combined_converted[i][4])
    chestPainType = trainData_combined_converted[i][5]
    tempSubArray = np.array([trainData_combined_converted[i][:5]])
    for i in range(4):
        tempSubArray = np.append(tempSubArray, 0)
    tempSubArray = catPainConverter(chestPainType, tempSubArray)
    trainData_combined_converted_new = np.append(trainData_combined_converted_new, tempSubArray)
trainData_combined_converted_new = trainData_combined_converted_new.reshape(trainData_combined_converted.shape[0], 9)

#b)

trainingData_combined_data = trainData_combined_converted_new[:250]
trainingData_combined_labels = trainData_labels[:250]
testingData_combined_data = trainData_combined_converted_new[250:]
testingData_combined_labels = trainData_labels[250:]

RandomForest = RandomForestClassifier(n_estimators=100)
RandomForest.fit(trainingData_combined_data, np.ravel(trainingData_combined_labels))
predictions = RandomForest.predict(testingData_combined_data)
modelAccuracy = accuracy_score(testingData_combined_labels, predictions)
print(modelAccuracy)

#c)
validData_combined_converted = validationData_combined

validData_combined_converted_new = np.array([])
for i in range(validData_combined_converted.shape[0]):
    validData_combined_converted[i][4] = catGenderConverter(validData_combined_converted[i][4])
    chestPainType = validData_combined_converted[i][5]
    tempSubArray = np.array([validData_combined_converted[i][:5]])
    for i in range(4):
        tempSubArray = np.append(tempSubArray, 0)
    tempSubArray = catPainConverter(chestPainType, tempSubArray)
    #print(tempSubArray)
    validData_combined_converted_new = np.append(validData_combined_converted_new, tempSubArray)
validData_combined_converted_new = validData_combined_converted_new.reshape(validData_combined_converted.shape[0], 9)
#print(validData_combined_converted.shape[0])

validationDataConv_combined_data = validData_combined_converted_new[:50]
validationDataConv_combined_labels = validationData_labels[:50]
validationDataConvTesting_combined_data = validData_combined_converted_new[50:]
#print(validationDataConvTesting_combined_data)
validationDataConvTesting_combined_labels = validationData_labels[50:]

NewRandomForest = RandomForestClassifier(n_estimators=100)

criterionList = ["entropy", "gini"]
treeDepthList = [2, 5, 7, 10, 15]
#nFeatureList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
nFeatureList = ["sqrt", "log2"]


def ComputeNCorrect(predictions, testLabels):
    nCorrect = 0
    for i in range(len(testLabels)):
        if (predictions[i] == testLabels[i]):
            nCorrect += 1
    return nCorrect

def FormatPrint(criterion, maxDepth, maxFeatures, accuracy, nCorrectAmount):
    print("")
    print("criterion = " + criterion, end=" ;")
    print("max_depth = " + str(maxDepth), end=" ;")
    print("max_features = " + str(maxFeatures), end=" ;")
    print("accuracy on validation data = " + str(accuracy), end=" ;")
    print("number of correctly classified validation samples = " + str(nCorrectAmount), end=" ")

currentBestAccuracy = 0

for criterion in criterionList:
    NewRandomForest.criterion = criterion
    for Depth in treeDepthList:
        NewRandomForest.max_depth = Depth
        for featureAmount in nFeatureList:
            NewRandomForest.max_features = featureAmount
            NewRandomForest.fit(validationDataConv_combined_data, np.ravel(validationDataConv_combined_labels))
            NewPredictions = NewRandomForest.predict(validationDataConvTesting_combined_data)
            prediction_nCorrectAmount = ComputeNCorrect(NewPredictions, validationDataConvTesting_combined_labels)
            prediction_accuracy = accuracy_score(validationDataConvTesting_combined_labels, NewPredictions)
            if (prediction_accuracy > currentBestAccuracy or currentBestAccuracy == 0):
                currentBestAccuracy = prediction_accuracy
                FormatPrint(criterion, Depth, featureAmount, prediction_accuracy, prediction_nCorrectAmount)










