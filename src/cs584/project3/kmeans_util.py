'''
Created on Apr 13, 2020

@author: William
'''
import numpy as np
import operator
import random

def fixEmptyClusters(X, clusterAssignments, clusterCenters, distanceFunction, missingClusters):
    totalPoints = X.shape[0]
    dataSetRangeList = list(range(totalPoints))
    
    for missingCluster in missingClusters:
        errorMap = computeClusterErrorMap(X, clusterAssignments, clusterCenters, distanceFunction)
        worstClusterLabel = max(errorMap.items(), key=operator.itemgetter(1))[0]
        worstClusterIndices = [k for k in dataSetRangeList if clusterAssignments[k] == worstClusterLabel]
        #func = np.vectorize(lambda t: t == worstClusterLabel)
        #indices = func(clusterAssignments)
        #print("Found indices for clusterClass = " + str(clusterClass))
        #worstClusterPoints = X[worstClusterIndices, :]
        newPointSubIndex = random.randrange(len(worstClusterIndices))
        newPointIndex = worstClusterIndices[newPointSubIndex]
        
        clusterCenters[missingCluster] = np.copy(X[newPointIndex, :])
        clusterAssignments[newPointIndex] = missingCluster

# This function takes the labels that we expect! Might not be entire data set!
def calculateCentersFromAssignments(X, clusterAssignments, clusterLabels, centerFunction):
    clusterCenters = {}
    emptyClusters = []
    for clusterLabel in clusterLabels:
        func = np.vectorize(lambda t: t == clusterLabel)
        indices = func(clusterAssignments)
        #print("Found indices for clusterClass = " + str(clusterClass))
        clusterPoints = X[indices, :]
        #clusterCount = clusterPoints.shape[0]
        #print(clusterPoints)
        
        clusterPointsCount = clusterPoints.shape[0]
        if clusterPointsCount > 0:
            currentClusterCenter = centerFunction(clusterPoints)
            clusterCenters[clusterLabel] = currentClusterCenter
        else:
            emptyClusters.append(clusterLabel)
        #clusterCenter = np.sum(a=clusterPoints, axis=0, dtype=np.float64) / clusterCount
        #print("Center = " + str(clusterCenter))
        #if clusterCenter is not None:
            
        #else:
        #print("Got empty cluster!!")
    
    return clusterCenters, emptyClusters

def calculateAssignmentsFromCenters(X, clusterAssignments, clusterCenters, distanceFunction, initialAssignment=False):
    totalSamples = X.shape[0]
    #clusterAssignments = np.ndarray(shape=(totalSamples,), dtype=np.int8)
    totalChanges = 0
    for index in range(totalSamples):
        currentSample = X[index, :]
        closestClusterNum = None
        closestClusterDistance = None
        for clusterNum, clusterCenter in clusterCenters.items():
            currentDistance = distanceFunction(currentSample, clusterCenter)
            if closestClusterNum is None or currentDistance < closestClusterDistance:
                closestClusterNum = clusterNum
                closestClusterDistance = currentDistance
        
        if initialAssignment or clusterAssignments[index] != closestClusterNum:
            totalChanges += 1
            clusterAssignments[index] = closestClusterNum
    
    #return clusterAssignments
    return totalChanges
    
def computeClusterErrorMap(X, clusterAssignments, clusterCenters, distanceFunction):
    clusterErrorMap = {}
    for clusterLabel, clusterCenter in clusterCenters.items():
        func = np.vectorize(lambda t: t == clusterLabel)
        indices = func(clusterAssignments)
        clusterPoints = X[indices, :]
        squaredErrorFunc = lambda t: distanceFunction(t, clusterCenter) ** 2
        sseArray = np.apply_along_axis(squaredErrorFunc, axis=1, arr=clusterPoints)
        totalClusterSSE = np.sum(a=sseArray, axis=0, dtype=np.float64)
        clusterErrorMap[clusterLabel] = totalClusterSSE
    
    return clusterErrorMap
