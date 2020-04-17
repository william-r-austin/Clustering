'''
Created on Apr 13, 2020

@author: William
'''
import cs584.project3.kmeans_digits as kmdigits
import cs584.project3.common as common

def tuneSubmission01():
    kmdigits.submission01(False, 5)
    
def tuneSubmission02():
    kmdigits.submission02(False, 5)
    
def tuneWithMNIST():
    kmdigits.tuneParametersMNIST()
    
def tuneWithMNIST2():
    kmdigits.tuneParametersMNIST2()    

if __name__ == '__main__':
    #common.testConversion()
    kmdigits.chartClusterErrorVsClusterCount()
