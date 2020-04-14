'''
Created on Apr 13, 2020

@author: William
'''
import numpy as np
import cs584.project3.kmeans_util as kmutil

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

def initKMeansKMeansPlusPlus(X, labels):
    pass
    