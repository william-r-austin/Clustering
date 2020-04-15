'''
Created on Apr 14, 2020

@author: William
'''
import cs584.project3.common as common
import cs584.project3.digit_util as digitutil
from sklearn.preprocessing import normalize
import numpy as np
import cs584.project3.kmeans_clustering as kmclustering
import cs584.project3.kmeans_init as kminit


def submission01(createOutput):
    X = common.readDigitsFile()
    print("Shape of Digits file  = " + str(X.shape))
    X_new = digitutil.preprocessDownsampling(X)
    print("Got X_new. Shape = " + str(X_new.shape))
    #print("First 10 rows are: ")
    #print(X_new[0:10, :])
    
    numTrials = 10
    X_normalized = normalize(X, norm='l2', axis=0)
    distanceFunc = common.euclideanDistanceFunction
    centerFunc = common.digitsDataCenterFunction
    clusterLabels = np.array(list(range(1, 10)), dtype=np.int8)
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
    #print("Best Assignments = " + str(bestAssignments))
    if createOutput:
        common.writeResultsFile(bestAssignments)
