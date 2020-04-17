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
from sklearn.preprocessing import StandardScaler
from sklearn.manifold.t_sne import TSNE

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

def compareInitialization():
    X = common.readIrisFile()
    scaler = StandardScaler()
    X_new = scaler.fit_transform(X)
    
    distanceFunc = common.irisDataDistanceFunction
    centerFunc = common.irisDataCenterFunction
    clusterLabels = np.array(list(range(1, 4)), dtype=np.int8)
    numTrials = 5
    
    kmSampling = kminit.initKMeansSampling
    kmRandom = lambda x, c: kminit.initKMeansRandomly(x, c, centerFunc, distanceFunc)
    kmPlusPlus = lambda x, c: kminit.initKMeansKMeansPlusPlus(x, c, distanceFunc)
    
    initFunctions = [(kmSampling, "Sampling"), (kmRandom, "Random"), (kmPlusPlus, "K-Means++")]
    
    ##############################################################
    # Basic K-means
    ##############################################################

    for initFunction, initFunctionDesc in initFunctions:
        bestErrorTotal = None
        errorSum = 0
        
        for z in range(numTrials):
            model = kmclustering.BasicKMeansClusteringModel(distanceFunc, centerFunc, 100)
            model.runBasicKMeansClustering(X_new, clusterLabels, initFunction)
            errorSum += model.finalClusterErrorTotal
            print("Done with Basic K-Means with initialization type = " + initFunctionDesc + ", Trial # " + str(z + 1) + " / " + str(numTrials))
            print("Error Total = " + str(model.finalClusterErrorTotal))
            
            if bestErrorTotal is None or model.finalClusterErrorTotal < bestErrorTotal:
                bestErrorTotal = model.finalClusterErrorTotal
        
        avgError = errorSum / numTrials
        print("Finished Basic K-Means with initialization type = " + initFunctionDesc + ". Avg Error = " + str(avgError) + " and Best Error = " + str(bestErrorTotal))

def compareBasicAndBisecting():
    X = common.readIrisFile()
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    
    tsne = TSNE(n_components=2, perplexity=20, learning_rate=200, n_iter=1000, init='random')
    X_new = tsne.fit_transform(X_normalized)
    
    distanceFunc = common.irisDataDistanceFunction
    centerFunc = common.irisDataCenterFunction
    clusterLabels = np.array(list(range(1, 4)), dtype=np.int8)
    numTrials = 5
    
    ##############################################################
    # Basic K-means
    ##############################################################
    bestAssignments = None
    bestErrorTotal = None
    errorSum = 0
    for z in range(numTrials):
        model = kmclustering.BasicKMeansClusteringModel(distanceFunc, centerFunc, 100)
        model.runBasicKMeansClustering(X_new, clusterLabels, kminit.initKMeansSampling)
        errorSum += model.finalClusterErrorTotal
        print("Done with Basic K-Means Trial # " + str(z + 1) + " / " + str(numTrials))
        print("Error Total = " + str(model.finalClusterErrorTotal))
        
        if bestErrorTotal is None or model.finalClusterErrorTotal < bestErrorTotal:
            bestErrorTotal = model.finalClusterErrorTotal
            bestAssignments = model.finalClusterAssignments
    
    avgError = errorSum / numTrials
    print("Finished Basic K-Means. Avg Error = " + str(avgError) + " and Best Error = " + str(bestErrorTotal))
    print("Creating output now for Basic K-Means.")
    common.writeResultsFile(bestAssignments)    
    
    ##############################################################
    # Bisecting K-means
    ##############################################################
    bestAssignments = None
    bestErrorTotal = None
    errorSum = 0
    for z in range(numTrials):
        model = kmclustering.BisectingKMeansClusteringModel(distanceFunc, centerFunc, 5)
        model.runBisectingKMeansClustering(X_new, clusterLabels, kminit.initKMeansSampling)
        errorSum += model.finalClusterErrorTotal
        print("Done with Bisecting K-Means Trial # " + str(z + 1) + " / " + str(numTrials))
        print("Error Total = " + str(model.finalClusterErrorTotal))
        
        if bestErrorTotal is None or model.finalClusterErrorTotal < bestErrorTotal:
            bestErrorTotal = model.finalClusterErrorTotal
            bestAssignments = model.finalClusterAssignments
    
    avgError = errorSum / numTrials
    print("Finished Bisecting K-Means. Avg Error = " + str(avgError) + " and Best Error = " + str(bestErrorTotal))
    print("Creating output now for Bisecting K-Means.")
    common.writeResultsFile(bestAssignments)
    