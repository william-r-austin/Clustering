'''
Created on Mar 7, 2020

@author: William
'''
import numpy as np
from os.path import dirname, realpath

def getProjectRootDirectory():
    dirPath = dirname(realpath(__file__))
    projectDirPath = dirname(dirname(dirname(dirPath)))
    return projectDirPath

def getCenterForIrisData(X):
    totalElements = len(X)
    dataSum = np.zeros(shape=(4,), dtype=np.float64)
    for x in X:
        dataSum += x
    return dataSum / totalElements

def getCenterForDigitsData(X):
    
    
