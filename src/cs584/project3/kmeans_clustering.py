'''
Created on Apr 12, 2020

@author: William
'''
import numpy as np
import cs584.project3.kmeans_util as kmutil
#import cs584.project3.kmeans_init as kminit

class BasicKMeansClusteringModel(object):
    '''
    classdocs
    '''
    def __init__(self, distanceFunction, centerFunction):
        '''
        Constructor
        '''
        self.distanceFunction = distanceFunction
        self.centerFunction = centerFunction

        self.finalClusterCenters = None
        self.finalClusterAssignments = None
        self.finalClusterErrorMap = None
        self.finalClusterErrorTotal = None
    
    # random, sampling, kmeans++
    # Centers must be set!
    def runBasicKMeansClustering(self, X, clusterLabels, initFunction):
        dataSize = X.shape[0]
        centers = initFunction(X, clusterLabels)
        clusterAssignments = np.ndarray(shape=(dataSize,), dtype=np.int8)
        #np.array([None] * dataSize, dtype=np.int8)
    
        #clusterClasses = np.arange(1, clusterCount + 1, dtype=np.int8)
        #print("Printing cluster classes")
        #print(clusterClasses.shape)
        #print(clusterClasses)
        
        #clusterLabels = np.random.choice(a=clusterClasses, size=dataSize)
        #print("Cluster Labels")
        #print(clusterLabels.shape)
        #print(clusterLabels)
        
        converged = False
        iteration = 0
        while not converged and iteration < 25:
            #centers = calculateClassCenters(X, clusterLabels, clusterClasses)
            isInitialAssignment = (iteration == 0)
            totalChanges = \
                kmutil.calculateAssignmentsFromCenters(X, clusterAssignments, centers, self.distanceFunction, initialAssignment=isInitialAssignment)
            '''
            totalChanges = 0
            
            for r in range(dataSize):
                closestClusterNum = None
                smallestDistance = None
                currentSample = X[r, :]
                for clusterNum, clusterCenter in centers.items():
                    centerDistance = self.distanceFunction(clusterCenter, currentSample)
                    if closestClusterNum is None or centerDistance < smallestDistance:
                        closestClusterNum = clusterNum
                        smallestDistance = centerDistance
                
                if clusterAssignments[r] is None or clusterAssignments[r] != closestClusterNum:
                    clusterAssignments[r] = closestClusterNum
                    totalChanges += 1
            '''
            #print("FINISHED ITERATION " + str(iteration))
            if totalChanges == 0:
                converged = True
                #print("Complete! Final Centers are below: ")
                
                #for clusterNum, clusterCenter in centers.items():
                #    print("Center of Cluster #" + str(clusterNum) + " = " + str(clusterCenter))
            else:
                #print("Total changes = " + str(totalChanges) + ". Recomputing cluster centers now")
                tempCenters, emptyClusters = kmutil.calculateCentersFromAssignments(X, clusterAssignments, clusterLabels, self.centerFunction)
                
                if emptyClusters:
                    kmutil.fixEmptyClusters(X, clusterAssignments, tempCenters, self.distanceFunction, emptyClusters)
                else:
                    iteration += 1
                
                centers = tempCenters
                
                '''
                if tempCenters is not None:
                    centers = tempCenters
                    iteration += 1
                else:
                    #print("Got empty cluster. Restarting.")
                    iteration = 0
                    centers = initFunction(X, clusterLabels)
                '''
                
    
            #iteration += 1
        
        self.finalClusterAssignments = clusterAssignments
        self.finalClusterCenters = centers
        self.finalClusterErrorMap = kmutil.computeClusterErrorMap(X, clusterAssignments, centers, self.distanceFunction)
        self.finalClusterErrorTotal = sum(self.finalClusterErrorMap.values())      
        
        # Initialization
        '''
        if labels is None and centers is None:
            if initMode == 'random':
                pass
            elif initMode == 'sampling'
                pass
            elif initMode = 'kmeans++'
                pass
            else:
                raise Exception("Invalid Initialization method: " + str(initMode))
        elif labels is None and centers is not None:
            pass
        elif labels is not None and centers is None:
            pass
        '''

class BisectingKMeansClusteringModel(object):
    '''
    classdocs
    '''
    def __init__(self, distanceFunction, centerFunction, maxIterations):
        '''
        Constructor
        '''
        self.distanceFunction = distanceFunction
        self.centerFunction = centerFunction
        self.maxIterations = maxIterations
        
        self.finalClusterCenters = None
        self.finalClusterAssignments = None
        self.finalClusterErrorMap = None
        self.finalClusterErrorTotal = None

    def runBisectingKMeansClustering(self, X, clusterLabels, initFunction):
        sampleCount = X.shape[0]
        sampleRangeList = list(range(sampleCount))
        currentClusterLabel = clusterLabels[0]
        clusterAssignments = np.array([currentClusterLabel] * sampleCount, dtype=np.int8)
        expectedLabels = clusterLabels[0:1]
        # This is ok because there is only 1 cluster
        clusterCenters, emptyClusters = kmutil.calculateCentersFromAssignments(X, clusterAssignments, expectedLabels, self.centerFunction)
        if emptyClusters:
            kmutil.fixEmptyClusters(X, clusterAssignments, clusterCenters, self.distanceFunction, emptyClusters)
        
        clusterErrorMap = kmutil.computeClusterErrorMap(X, clusterAssignments, clusterCenters, self.distanceFunction)
        
        currentClusterIndex = 1
        while currentClusterIndex < clusterLabels.shape[0]:
            expectedLabels = clusterLabels[0:currentClusterIndex+1]
            currentClusterLabel = clusterLabels[currentClusterIndex]
            
            sourceClusterLabel = None
            sourceClusterScore = None
            
            for label, score in clusterErrorMap.items():
                if sourceClusterScore is None or score > sourceClusterScore:
                    sourceClusterLabel = label
                    sourceClusterScore = score
            
            sourceClusterIndices = np.array([q for q in sampleRangeList if clusterAssignments[q] == sourceClusterLabel])
            
            X_partial = X[sourceClusterIndices, :]
            partialClusterLabels = np.array([sourceClusterLabel, currentClusterLabel], dtype=np.int8)
            
            currentIteration = 0
            bestClusterAssignments = None
            bestErrorTotal = None
            while currentIteration < self.maxIterations:
                #initialAssignments, initialClusters = kminit.initKMeansSampling(X_partial, clusterLabels, self.distanceFunction)
                kmModel = BasicKMeansClusteringModel(distanceFunction=self.distanceFunction, centerFunction=self.centerFunction)
                kmModel.runBasicKMeansClustering(X_partial, partialClusterLabels, initFunction)
                currentErrorTotal = kmModel.finalClusterErrorTotal
                #print("###########################=========> Error Map: " + str(kmModel.finalClusterErrorMap))
                #print("###########################=========> Error Total: " + str(currentErrorTotal))
                
                if bestErrorTotal is None or currentErrorTotal < bestErrorTotal:
                    bestClusterAssignments = np.copy(kmModel.finalClusterAssignments)
                    bestErrorTotal = currentErrorTotal
                
                currentIteration += 1
            
            for sourceClusterIndex, clusterAssignment in zip(sourceClusterIndices, bestClusterAssignments):
                clusterAssignments[sourceClusterIndex] = clusterAssignment

            # This should be ok because basic k-means does not return empty clusters
            clusterCenters, emptyClusters = kmutil.calculateCentersFromAssignments(X, clusterAssignments, expectedLabels, self.centerFunction)
            
            if emptyClusters:
                kmutil.fixEmptyClusters(X, clusterAssignments, clusterCenters, self.distanceFunction, emptyClusters)
            
            clusterErrorMap = kmutil.computeClusterErrorMap(X, clusterAssignments, clusterCenters, self.distanceFunction)
            currentClusterIndex += 1
        
        self.finalClusterCenters = clusterCenters
        self.finalClusterAssignments = clusterAssignments
        self.finalClusterErrorMap = clusterErrorMap
        self.finalClusterErrorTotal = sum(self.finalClusterErrorMap.values())   
        