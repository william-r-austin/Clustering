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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import math

import matplotlib.pyplot as plt

def tuneTsneBasic(createOutput):
    numTrials = 5
    X = common.readDigitsFile()
    print("Shape of Digits file  = " + str(X.shape))
    print("Starting t-SNE.")
    #pca = PCA(n_components=50)
    #X_temp = pca.fit_transform(X)
    tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=35, learning_rate=400, n_iter=2000)
    X_new = tsne.fit_transform(X)
    #X_new = digitutil.preprocessDownsampling(X)
    print("Finished t-SNE, and got X_new. Shape = " + str(X_new.shape))
    #print("First 10 rows are: ")
    #print(X_new[0:10, :])
    
    #numTrials = 10
    #X_normalized = normalize(X, norm='l2', axis=0)
    distanceFunc = common.euclideanDistanceFunction
    centerFunc = common.digitsDataCenterFunction
    clusterLabels = np.array(list(range(1, 11)), dtype=np.int8)
    bestAssignments = None
    bestErrorTotal = None
    for z in range(numTrials):
        model = kmclustering.BasicKMeansClusteringModel(distanceFunc, centerFunc, 100)
        #initFunc = lambda q, z: kminit.initKMeansSampling(q, z)
        #assignments, centers = kminit.initKMeansSampling(X_normalized, clusterLabels, distanceFunc)
        model.runBasicKMeansClustering(X_new, clusterLabels, kminit.initKMeansSampling)
        
        print("====== Done with Trial # " + str(z + 1) + " / " + str(numTrials) + ", Error Total = " + str(model.finalClusterErrorTotal) + " ======")
        #print("Assignments = " + str(model.finalClusterAssignments))
        #print("Centers = " + str(model.finalClusterCenters))
        #print("Error Map = " + str(model.finalClusterErrorMap))
        #print()
        
        if bestErrorTotal is None or model.finalClusterErrorTotal < bestErrorTotal:
            bestErrorTotal = model.finalClusterErrorTotal
            bestAssignments = model.finalClusterAssignments
            #print("Improved total SSE! New Value = " + str(bestErrorTotal))
        
    print("Finished K-Means with t-SNE, P=. Best Error Total = " + str(bestErrorTotal))
    #print("Best Assignments = " + str(bestAssignments))
    if createOutput:
        common.writeResultsFile(bestAssignments)

def tuneTsneBisecting(createOutput):
    numTrials = 5
    X = common.readDigitsFile()
    print("Shape of Digits file  = " + str(X.shape))
    print("Starting t-SNE.")
    #pca = PCA(n_components=50)
    #X_temp = pca.fit_transform(X)
    tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=35, learning_rate=400, n_iter=2000)
    X_new = tsne.fit_transform(X)
    #X_new = digitutil.preprocessDownsampling(X)
    print("Finished t-SNE, and got X_new. Shape = " + str(X_new.shape))
    #print("First 10 rows are: ")
    #print(X_new[0:10, :])
    
    #numTrials = 10
    #X_normalized = normalize(X, norm='l2', axis=0)
    distanceFunc = common.euclideanDistanceFunction
    centerFunc = common.digitsDataCenterFunction
    clusterLabels = np.array(list(range(1, 11)), dtype=np.int8)
    bestAssignments = None
    bestErrorTotal = None
    for z in range(numTrials):
        
        
        model = kmclustering.BisectingKMeansClusteringModel(distanceFunc, centerFunc, 3)
        model.runBisectingKMeansClustering(X_new, clusterLabels, kminit.initKMeansSampling)
        #model = kmclustering.BasicKMeansClusteringModel(distanceFunc, centerFunc, 100)
        #initFunc = lambda q, z: kminit.initKMeansSampling(q, z)
        #assignments, centers = kminit.initKMeansSampling(X_normalized, clusterLabels, distanceFunc)
        #model.runBasicKMeansClustering(X_new, clusterLabels, kminit.initKMeansSampling)
        
        print("====== Done with Trial # " + str(z + 1) + " / " + str(numTrials) + ", Error Total = " + str(model.finalClusterErrorTotal) + " ======")
        #print("Assignments = " + str(model.finalClusterAssignments))
        #print("Centers = " + str(model.finalClusterCenters))
        #print("Error Map = " + str(model.finalClusterErrorMap))
        #print()
        
        if bestErrorTotal is None or model.finalClusterErrorTotal < bestErrorTotal:
            bestErrorTotal = model.finalClusterErrorTotal
            bestAssignments = model.finalClusterAssignments
            #print("Improved total SSE! New Value = " + str(bestErrorTotal))
        
    print("Finished K-Means with t-SNE, P=. Best Error Total = " + str(bestErrorTotal))
    #print("Best Assignments = " + str(bestAssignments))
    if createOutput:
        common.writeResultsFile(bestAssignments)


def tuneParametersMNIST():
    X, y = common.readMNIST(1000)
    
    print("Got X, y. Shapes are: " + str(X.shape) + ", and " + str(y.shape))
    
    #X2 = digitutil.preprocessDownsampling(X)
    #print("Finished downsampling. New shape is: " + str(X2.shape))
    
    # No need to bother with normalizing
    #X3 = normalize(X2, norm='l2', axis=0)
    
    tsnePerplexities = [5, 10, 25, 45, 70, 100] #[35]
    tsneLearningRates = [400] #[100, 200, 400, 800]
    
    #tsneLearningRates = [12, 50, 100, 200, 600, 1000]
    
    for p in tsnePerplexities:
        for lr in tsneLearningRates:
            chartTitle = "t-SNE Output with P = " + str(p) + " and LR = " + str(lr) + ", no downsampling" 
            tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=p, learning_rate=lr, n_iter=2000)
            X_transformed = tsne.fit_transform(X)
            print("Finished TSNE for chart: " + chartTitle)
            
            tsneFigure = plt.figure(figsize=(8,8))
            subplot = tsneFigure.add_subplot(1, 1, 1, title=chartTitle)
        
            subplot.scatter(x=X_transformed[:,0], y=X_transformed[:,1], c=y, cmap=plt.cm.get_cmap('Paired'), alpha=0.35)
    
    plt.show()
    
    #if createOutput:
    #    common.writeResultsFile(resultsArray)
    
    
def tuneParametersMNIST2():
    X, y = common.readMNIST(1000)
    
    print("Got X, y. Shapes are: " + str(X.shape) + ", and " + str(y.shape))
    
    #X2 = digitutil.preprocessDownsampling(X)
    #print("Finished downsampling. New shape is: " + str(X2.shape))
    
    # No need to bother with normalizing
    #X3 = normalize(X2, norm='l2', axis=0)
    
    #tsnePerplexities = [5, 10, 25, 45, 70, 100] #[35]
    tsnePerplexity = int(math.sqrt(X.shape[0]))
    tsneLearningRates = [150, 200, 260, 330, 410, 500, 600] #[100, 200, 400, 800]
    tsneDims = [2, 3]
    distanceFunc = common.euclideanDistanceFunction
    centerFunc = common.digitsDataCenterFunction
    clusterLabels = np.array(list(range(1, 11)), dtype=np.int8)
    #tsneLearningRates = [12, 50, 100, 200, 600, 1000]
    
    for tsneDim in tsneDims:
        for lr in tsneLearningRates:
            chartTitle = "t-SNE Output with P = " + str(tsnePerplexity) + ", LR = " + str(lr) + ", no downsampling, and dims = " + str(tsneDim) 
            tsne = TSNE(n_components=tsneDim, init='random', random_state=0, perplexity=tsnePerplexity, learning_rate=lr, n_iter=2000)
            X_transformed = tsne.fit_transform(X)
            print("Finished TSNE for chart: " + chartTitle)
            

            
            #model = kmclustering.BasicKMeansClusteringModel(distanceFunc, centerFunc, 100)
            #initFunc = lambda q, z: kminit.initKMeansSampling(q, z)
            #assignments, centers = kminit.initKMeansSampling(X_normalized, clusterLabels, distanceFunc)
            #model.runBasicKMeansClustering(X_new, clusterLabels, kminit.initKMeansSampling)
        
        #print("====== Done with Trial # " + str(z + 1) + " / " + str(numTrials) + ", Error Total = " + str(model.finalClusterErrorTotal) + " ======")
            
    '''            
            tsneFigure = plt.figure(figsize=(8,8))
            subplot = tsneFigure.add_subplot(1, 1, 1, title=chartTitle)
        
            subplot.scatter(x=X_transformed[:,0], y=X_transformed[:,1], c=y, cmap=plt.cm.get_cmap('Paired'), alpha=0.35)
    
    plt.show()
    '''
    
    #if createOutput:
    #    common.writeResultsFile(resultsArray)
    

def submission02(createOutput, numTrials):
    X = common.readDigitsFile()
    print("Shape of Digits file  = " + str(X.shape))
    pca = PCA(n_components=50)
    X_temp = pca.fit_transform(X)
    tsne = TSNE(n_components=2, init='pca')
    X_new = tsne.fit_transform(X_temp)
    #X_new = digitutil.preprocessDownsampling(X)
    print("Got X_new. Shape = " + str(X_new.shape))
    #print("First 10 rows are: ")
    #print(X_new[0:10, :])
    
    #numTrials = 10
    X_normalized = normalize(X, norm='l2', axis=0)
    distanceFunc = common.euclideanDistanceFunction
    centerFunc = common.digitsDataCenterFunction
    clusterLabels = np.array(list(range(1, 11)), dtype=np.int8)
    bestAssignments = None
    bestErrorTotal = None
    for z in range(numTrials):
        model = kmclustering.BasicKMeansClusteringModel(distanceFunc, centerFunc, 55)
        #initFunc = lambda q, z: kminit.initKMeansSampling(q, z)
        #assignments, centers = kminit.initKMeansSampling(X_normalized, clusterLabels, distanceFunc)
        model.runBasicKMeansClustering(X_normalized, clusterLabels, kminit.initKMeansSampling)
        
        print("====== Done with Trial # " + str(z + 1) + " / " + str(numTrials) + ", Error Total = " + str(model.finalClusterErrorTotal) + " ======")
        #print("Assignments = " + str(model.finalClusterAssignments))
        #print("Centers = " + str(model.finalClusterCenters))
        #print("Error Map = " + str(model.finalClusterErrorMap))
        #print()
        
        if bestErrorTotal is None or model.finalClusterErrorTotal < bestErrorTotal:
            bestErrorTotal = model.finalClusterErrorTotal
            bestAssignments = model.finalClusterAssignments
            #print("Improved total SSE! New Value = " + str(bestErrorTotal))
        
    print("Finished K-Means. Best Error Total = " + str(bestErrorTotal))
    #print("Best Assignments = " + str(bestAssignments))
    if createOutput:
        common.writeResultsFile(bestAssignments)

def submission01(createOutput, numTrials):
    X = common.readDigitsFile()
    print("Shape of Digits file  = " + str(X.shape))
    X_new = digitutil.preprocessDownsampling(X)
    print("Got X_new. Shape = " + str(X_new.shape))
    #print("First 10 rows are: ")
    #print(X_new[0:10, :])
    
    #numTrials = 10
    X_normalized = normalize(X, norm='l2', axis=0)
    distanceFunc = common.euclideanDistanceFunction
    centerFunc = common.digitsDataCenterFunction
    clusterLabels = np.array(list(range(1, 11)), dtype=np.int8)
    bestAssignments = None
    bestErrorTotal = None
    for z in range(numTrials):
        model = kmclustering.BasicKMeansClusteringModel(distanceFunc, centerFunc, 55)
        #initFunc = lambda q, z: kminit.initKMeansSampling(q, z)
        #assignments, centers = kminit.initKMeansSampling(X_normalized, clusterLabels, distanceFunc)
        model.runBasicKMeansClustering(X_normalized, clusterLabels, kminit.initKMeansSampling)
        
        print("====== Done with Trial # " + str(z + 1) + " / " + str(numTrials) + ", Error Total = " + str(model.finalClusterErrorTotal) + " ======")
        #print("Assignments = " + str(model.finalClusterAssignments))
        #print("Centers = " + str(model.finalClusterCenters))
        #print("Error Map = " + str(model.finalClusterErrorMap))
        #print()
        
        if bestErrorTotal is None or model.finalClusterErrorTotal < bestErrorTotal:
            bestErrorTotal = model.finalClusterErrorTotal
            bestAssignments = model.finalClusterAssignments
            #print("Improved total SSE! New Value = " + str(bestErrorTotal))
        
    print("Finished K-Means. Best Error Total = " + str(bestErrorTotal))
    #print("Best Assignments = " + str(bestAssignments))
    if createOutput:
        common.writeResultsFile(bestAssignments)

def chartClusterErrorVsClusterCount():
    X = common.readDigitsFile()
    X = X / 255.0 
    print("Shape of Digits file  = " + str(X.shape))
    print("Starting t-SNE.")
    tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=100, learning_rate=400, n_iter=2000)
    X_new = tsne.fit_transform(X)
    print("Finished t-SNE, and got X_new. Shape = " + str(X_new.shape))

    distanceFunc = common.euclideanDistanceFunction
    centerFunc = common.digitsDataCenterFunction
    numTrials = 5
    errorMapForClusterSize = {}
    clusterSizeList = list(range(2, 21, 2))
    
    for clusterSize in clusterSizeList:
        errorSum = 0
        clusterLabels = np.array(list(range(1, clusterSize + 1)), dtype=np.int8)
        
        for z in range(numTrials):
            model = kmclustering.BasicKMeansClusteringModel(distanceFunc, centerFunc, 100)
            model.runBasicKMeansClustering(X_new, clusterLabels, kminit.initKMeansSampling)
            
            print("====== Done with Trial # " + str(z + 1) + " / " + str(numTrials) + " for K = " + str(clusterSize) + \
                  ", Error Total = " + str(model.finalClusterErrorTotal) + " ======")
            
            errorSum += model.finalClusterErrorTotal
        
        avgError = errorSum / numTrials
        errorMapForClusterSize[clusterSize] = avgError
    
    print("Done with K-Means.")
    
    for clusterSize in clusterSizeList:
        print("For K = " + str(clusterSize) + "     Error is:      " + str(errorMapForClusterSize[clusterSize]))
