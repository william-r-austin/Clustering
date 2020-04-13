'''
Created on Apr 12, 2020

@author: William
'''

class BasicKMeansClusteringModel(object):
    '''
    classdocs
    '''
    def __init__(self, clusterCount=3, distanceFunction):
        '''
        Constructor
        '''
        self.distanceFunction = distanceFunction
        self.clusterCount = clusterCount
        self.finalClusterCenters = None
        self.finalClusterSSE = None
    
    def calculateCenters(self, labels):
        pass
    
    def calculateLabels(self):
        pass
    
    # random, sampling, kmeans++
    def runBasicKMeansClustering(self, X, clusterAssignments, clusterCenters):
        
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
        
        converged = False
        while not converged:
            pass 

class BisectingKMeansClusteringModel(object):
    '''
    classdocs
    '''
    def __init__(self, distanceFunction, clusters=10):
        '''
        Constructor
        '''
        self.distanceFunction = distanceFunction
        self.clusters = clusters
    
    def executeBisectingKMeans
    
    def runBisectingKMeansInitial(X, labels, ):


    def runBisectingKMeans(X, totalClusterCount):
        labels = [0] * len(X)
        calculateClusterCenters
        
        for j in range(2)
    
    
    
        