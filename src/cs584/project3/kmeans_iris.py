'''
Created on Apr 14, 2020

@author: William
'''
import cs584.project3.common as common
from sklearn.preprocessing import normalize
from scipy.spatial.distance import euclidean
import numpy as np
import cs584.project3.kmeans_clustering as kmclustering
import cs584.project3.kmeans_init as kminit

def submission02(createOutput):
    numTrials = 25
    X = common.readIrisFile()
    X_normalized = normalize(X, norm='l2', axis=0)
    distanceFunc = common.irisDataDistanceFunction
    centerFunc = common.irisDataCenterFunction
    clusterLabels = np.array(list(range(1, 4)), dtype=np.int8)
    bestAssignments = None
    bestErrorTotal = None
    for z in range(numTrials):
        model = kmclustering.BasicKMeansClusteringModel(distanceFunc, centerFunc)
        #initFunc = lambda q, z: kminit.initKMeansSampling(q, z)
        #assignments, centers = kminit.initKMeansSampling(X_normalized, clusterLabels, distanceFunc)
        model.runBasicKMeansClustering(X_normalized, clusterLabels, kminit.initKMeansSampling)
        
        print("====== Done with Trial # " + str(z + 1) + " / " + str(numTrials))
        #print("Assignments = " + str(model.finalClusterAssignments))
        #print("Centers = " + str(model.finalClusterCenters))
        #print("Error Map = " + str(model.finalClusterErrorMap))
        print("Error Total = " + str(model.finalClusterErrorTotal))
        
        if bestErrorTotal is None or model.finalClusterErrorTotal < bestErrorTotal:
            bestErrorTotal = model.finalClusterErrorTotal
            bestAssignments = model.finalClusterAssignments
            print("Improved total SSE! New Value = " + str(bestErrorTotal))
        
    print("Finished K-Means. Best Error Total = " + str(bestErrorTotal))
    print("Best Assignments = " + str(bestAssignments))
    if createOutput:
        common.writeResultsFile(bestAssignments)
    
def submission02a(createOutput):
    numTrials = 25
    X = common.readIrisFile()
    X_normalized = normalize(X, norm='l2', axis=0)
    distanceFunc = common.irisDataDistanceFunction
    centerFunc = common.irisDataCenterFunction
    clusterLabels = np.array(list(range(1, 4)), dtype=np.int8)
    bestAssignments = None
    bestErrorTotal = None
    for z in range(numTrials):
        model = kmclustering.BisectingKMeansClusteringModel(distanceFunc, centerFunc, 10)
        model.runBisectingKMeansClustering(X_normalized, clusterLabels, kminit.initKMeansSampling)
        
        #model = kmclustering.BasicKMeansClusteringModel(distanceFunc, centerFunc)
        #initFunc = lambda q, z: kminit.initKMeansSampling(q, z)
        #assignments, centers = kminit.initKMeansSampling(X_normalized, clusterLabels, distanceFunc)
        #model.runBasicKMeansClustering(X_normalized, clusterLabels, kminit.initKMeansSampling)
        
        print("====== Done with Trial # " + str(z + 1) + " / " + str(numTrials))
        #print("Assignments = " + str(model.finalClusterAssignments))
        #print("Centers = " + str(model.finalClusterCenters))
        #print("Error Map = " + str(model.finalClusterErrorMap))
        print("Error Total = " + str(model.finalClusterErrorTotal))
        
        if bestErrorTotal is None or model.finalClusterErrorTotal < bestErrorTotal:
            bestErrorTotal = model.finalClusterErrorTotal
            bestAssignments = model.finalClusterAssignments
            print("Improved total SSE! New Value = " + str(bestErrorTotal))
        
    print("Finished K-Means. Best Error Total = " + str(bestErrorTotal))
    print("Best Assignments = " + str(bestAssignments))
    if createOutput:
        common.writeResultsFile(bestAssignments)
