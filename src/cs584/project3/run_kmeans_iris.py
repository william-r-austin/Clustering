'''
Created on Apr 13, 2020

@author: William
'''
import cs584.project3.common as common
from sklearn.preprocessing import normalize
from scipy.spatial.distance import euclidean
import numpy as np
import cs584.project3.kmeans_clustering as kmclustering
import cs584.project3.kmeans_init as kminit

def calculateClassCenters(X, clusterLabels, clusterClasses):
    centers = {}
    for clusterClass in clusterClasses:
        func = np.vectorize(lambda t: t == clusterClass)
        indices = func(clusterLabels)
        #print("Found indices for clusterClass = " + str(clusterClass))
        clusterPoints = X[indices, :]
        clusterCount = clusterPoints.shape[0]
        #print(clusterPoints)
        
        clusterCenter = np.sum(a=clusterPoints, axis=0, dtype=np.float64) / clusterCount
        #print("Center = " + str(clusterCenter))
        centers[clusterClass] = clusterCenter
    
    return centers

def kmeans(X, clusterCount):
    dataSize = X.shape[0]
    
    clusterClasses = np.arange(1, clusterCount + 1, dtype=np.int8)
    #print("Printing cluster classes")
    #print(clusterClasses.shape)
    #print(clusterClasses)
    
    clusterLabels = np.random.choice(a=clusterClasses, size=dataSize)
    #print("Cluster Labels")
    #print(clusterLabels.shape)
    #print(clusterLabels)
    
    converged = False
    iteration = 1
    while not converged and iteration < 1000:
        centers = calculateClassCenters(X, clusterLabels, clusterClasses)
        totalChanges = 0
        
        for r in range(dataSize):
            closestClusterNum = None
            smallestDistance = None
            for clusterNum, clusterCenter in centers.items():
                centerDistance = euclidean(X[r, :], clusterCenter)
                if closestClusterNum is None or centerDistance < smallestDistance:
                    closestClusterNum = clusterNum
                    smallestDistance = centerDistance
            
            if clusterLabels[r] != closestClusterNum:
                clusterLabels[r] = closestClusterNum
                totalChanges += 1
        
        
        print("FINISHED ITERATION " + str(iteration))
        if totalChanges == 0:
            converged = True
            print("Complete! Final Centers are below: ")
            
            for clusterNum, clusterCenter in centers.items():
                print("Center of Cluster #" + str(clusterNum) + " = " + str(clusterCenter))
        else:
            print("Total changes = " + str(totalChanges))

        iteration += 1
     
    return clusterLabels

def runSubmission01():
    #print("This is a sample file.")
    X = common.readIrisFile()
    
    featureCount = X.shape[1]
    
    n = normalize(X, norm='l2', axis=0)
    
    #print(n)
    result = kmeans(n, 3)
    common.writeResultsFile(result)

def runSubmission02():
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
    common.writeResultsFile(bestAssignments)
    
def runSubmission02a():
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
    common.writeResultsFile(bestAssignments)

if __name__ == '__main__':
    runSubmission02a()    
