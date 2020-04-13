'''
Created on Mar 7, 2020

@author: William
'''

from os.path import dirname, realpath

def getProjectRootDirectory():
    dirPath = dirname(realpath(__file__))
    projectDirPath = dirname(dirname(dirname(dirPath)))
    return projectDirPath
