'''
Created on Mar 14, 2020

@author: William
'''
import cs584.project3.utilities as utilities
import os
import numpy as np
import cs584.project3.constants as constants
from datetime import datetime

def readIrisFile():
    relativePath = constants.IRIS_DATA_FILE
    irisData = np.zeros(shape=(150,4), dtype=np.float64)
    rootDirectory = utilities.getProjectRootDirectory()
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

def writeResultsFile(resultsArray):
    rootDirectory = utilities.getProjectRootDirectory()
    x = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    relativePath = "resources/output/prediction_waustin_" + x + ".txt"
    fullPath = os.path.join(rootDirectory, relativePath)
    #print("Full Path = " + fullPath)

    arrayLength = resultsArray.shape[0]
    
    outputFile = open(fullPath, 'w', newline='')
    
    for index in range(arrayLength):
        intValue = int(resultsArray[index])
        outputLine = str(intValue) + "\n"
        outputFile.write(outputLine)
    
    outputFile.close()
