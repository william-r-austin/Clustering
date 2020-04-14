'''
Created on Apr 13, 2020

@author: William
'''
import cs584.project3.common as common

if __name__ == '__main__':
    X = common.readDigitsFile()
    print("Shape of Digits file  = " + str(X.shape))