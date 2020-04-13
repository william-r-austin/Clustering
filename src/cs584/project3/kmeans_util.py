'''
Created on Apr 13, 2020

@author: William
'''
import numpy as np

def getCenterForIrisData(X):
    totalElements = X.shape
    return np.sum(a=X, axis=0, dtype=np.float64) / totalElements
    
    '''
    dataSum = np.zeros(shape=(4,), dtype=np.float64)
    for x in X:
        dataSum += x
    return dataSum / totalElements
    '''

def getCenterForDigitsData(X):
    pass
    

def calculateCentersFromAssignments(X, clusterAssignments, clusterLabels, centerFunction):
    clusterCenters = {}
    for clusterLabel in clusterLabels:
        func = np.vectorize(lambda t: t == clusterLabel)
        indices = func(clusterAssignments)
        #print("Found indices for clusterClass = " + str(clusterClass))
        clusterPoints = X[indices, :]
        #clusterCount = clusterPoints.shape[0]
        #print(clusterPoints)
        
        clusterCenter = centerFunction(clusterPoints)
        
        #clusterCenter = np.sum(a=clusterPoints, axis=0, dtype=np.float64) / clusterCount
        #print("Center = " + str(clusterCenter))
        clusterCenters[clusterLabel] = clusterCenter
    
    return clusterCenters

def calculateAssignmentsFromCenters(X, clusterCenters):