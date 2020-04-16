'''
Created on Apr 13, 2020

@author: William
'''
import numpy as np
import cs584.project3.kmeans_util as kmutil
import random

def initKMeansRandomly(X, clusterLabels, centerFunction, distanceFunction):
    clusterAssignments = np.random.choice(clusterLabels, X.shape[0])
    clusterCenters, emptyClusters = kmutil.calculateCentersFromAssignments(X, clusterAssignments, clusterLabels, centerFunction)
    kmutil.fixEmptyClusters(X, clusterAssignments, clusterCenters, distanceFunction, emptyClusters)
    return clusterCenters

def initKMeansSampling(X, clusterLabels):
    clusterCenters = {}
    labelCount = clusterLabels.shape[0]
    sampleCount = X.shape[0]
    sampleIndexArray = list(range(sampleCount))
    initialCenterIndices = np.random.choice(a=sampleIndexArray, size=labelCount, replace=False)
    for clusterLabel, initialCenterIndex in zip(clusterLabels, initialCenterIndices):
        clusterCenters[clusterLabel] = np.copy(X[initialCenterIndex, :])
    
    '''
    clusterAssignments = np.ndarray(shape=(sampleCount,), dtype=np.int8)
    kmutil.calculateAssignmentsFromCenters(X, clusterAssignments, clusterCenters, distanceFunction, initialAssignment=True)
    '''
    
    #return (clusterAssignments, clusterCenters)
    return clusterCenters

def initKMeansKMeansPlusPlus(X, clusterLabels, distanceFunction):
    availableClusterIndexList = list(range(X.shape[0])) 
    chosenClusterIndexList = []
    currentClusterNum = 0
    totalClusterCount = clusterLabels.shape[0]
    
    # Choose the first center
    firstClusterIndex = np.random.choice(a=availableClusterIndexList, size=None)
    availableClusterIndexList.remove(firstClusterIndex)
    chosenClusterIndexList.append(firstClusterIndex)
    currentClusterNum += 1
        
    while currentClusterNum < totalClusterCount:
        availableClusterCount = len(availableClusterIndexList)
        distanceInitialized = False
        availableClusterDistance = np.ndarray(shape=(availableClusterCount,), dtype=np.single)
        
        for existingCenterIndex in chosenClusterIndexList:
            for availableIndex, globalIndex in enumerate(availableClusterIndexList):
                centerSample = X[existingCenterIndex, :]
                candidateSample = X[globalIndex, :]
                distance = distanceFunction(centerSample, candidateSample)
                if not distanceInitialized:
                    availableClusterDistance[availableIndex] = distance
                else:
                    if availableClusterDistance[availableIndex] < distance:
                        availableClusterDistance[availableIndex] = distance
                    distanceInitialized = True
                
        weights = np.square(availableClusterDistance)
        sumWeights = np.sum(a=weights, axis=0)
        weights = weights / sumWeights
        
        chosenCluster = np.random.choice(a=availableClusterIndexList, size=None, p=weights)
        availableClusterIndexList.remove(chosenCluster)
        chosenClusterIndexList.append(chosenCluster)
        currentClusterNum += 1
        
    clusterCenters = {}
    for clusterLabel, clusterCenterIndex in zip(clusterLabels, chosenClusterIndexList):
        clusterCenters[clusterLabel] = X[clusterCenterIndex, :]
    
    return clusterCenters
    