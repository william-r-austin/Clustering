'''
Created on Mar 14, 2020

@author: William
'''
import os
import numpy as np
import cs584.project3.constants as constants
from datetime import datetime
from os.path import dirname, realpath
from scipy.spatial.distance import euclidean

PRINT_LEVEL = 10

def print2(printObj, level):
    if level <= PRINT_LEVEL:
        print(printObj)
    

def getProjectRootDirectory():
    dirPath = dirname(realpath(__file__))
    projectDirPath = dirname(dirname(dirname(dirPath)))
    return projectDirPath

def readIrisFile():
    relativePath = constants.IRIS_DATA_FILE
    irisData = np.zeros(shape=(150,4), dtype=np.float64)
    rootDirectory = getProjectRootDirectory()
    filePath = os.path.join(rootDirectory, relativePath)
    
    dataFile = open(filePath, "r")
    index = 0
    for dataFileLine in dataFile:
        parts = dataFileLine.split(" ")
        for j in range(4):
            irisData[index, j] = float(parts[j].strip())
        index += 1
        
    dataFile.close()
    
    return irisData

def readIrisFileAsList():
    relativePath = constants.IRIS_DATA_FILE
    irisData = []
    rootDirectory = getProjectRootDirectory()
    filePath = os.path.join(rootDirectory, relativePath)
    
    dataFile = open(filePath, "r")
    index = 0
    for dataFileLine in dataFile:
        parts = dataFileLine.split(" ")
        numParts = len(parts)
        dataRowArray = np.array([float(j.strip()) for j in parts], dtype=np.float64)
        irisData.append(dataRowArray)
        for j in range(4):
            irisData[index, j] = float(parts[j].strip())
        index += 1
        
    dataFile.close()
    
    return irisData

def readDigitsFile():
    relativePath = constants.HANDWRITTEN_DIGITS_FILE
    digitsData = np.ndarray(shape=(10740, 784), dtype=np.uint8)
    
    rootDirectory = getProjectRootDirectory()
    filePath = os.path.join(rootDirectory, relativePath)
    
    dataFile = open(filePath, "r")
    index = 0
    for dataFileLine in dataFile:
        parts = dataFileLine.split(",")
        dataRow = np.array([int(k) for k in parts], dtype=np.uint8)
        digitsData[index, :] = dataRow
        index += 1
        
    dataFile.close()
    return digitsData

def writeResultsFile(resultsArray):
    rootDirectory = getProjectRootDirectory()
    x = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    relativePath = "resources\\output\\prediction_waustin_" + x + ".txt"
    fullPath = os.path.join(rootDirectory, relativePath)
    #print("Full Path = " + fullPath)

    arrayLength = resultsArray.shape[0]
    
    outputFile = open(fullPath, 'w', newline='')
    
    for index in range(arrayLength):
        intValue = int(resultsArray[index])
        outputLine = str(intValue) + "\n"
        outputFile.write(outputLine)
    
    outputFile.close()
    
    print("Finished creating output file. Path is: " + relativePath)

def euclideanDistanceFunction(a, b):
    return euclidean(a, b)

def digitsDataCenterFunction(X):
    totalElements = X.shape[0]
    if totalElements > 0:
        return np.sum(a=X, axis=0, dtype=np.single) / totalElements

def irisDataDistanceFunction(a, b):
    return euclidean(a, b)

def irisDataCenterFunction(X):
    totalElements = X.shape[0]
    if totalElements > 0:
        return np.sum(a=X, axis=0, dtype=np.float64) / totalElements
    
    raise Exception("Cannot compute center of empty set of points!")
