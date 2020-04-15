'''
Created on Apr 14, 2020

@author: William
'''
import numpy as np

def preprocessDownsampling(X):
    numSamples = X.shape[0]
    X_new = np.ndarray(shape=(numSamples,49), dtype=np.single)
    for index in range(numSamples):
        currentSquareArray = np.reshape(X[index, :], newshape=(28, 28))
        newSquareArray = np.ndarray(shape=(7, 7), dtype=np.single)
        for i in range(7):
            for j in range(7):
                subArray = currentSquareArray[i*4:(i+1)*4, j*4:(j+1)*4]
                newSquareArray[i, j] = np.sum(subArray) / 16
        
        '''
        if index < 10:
            print("Downsampled Digit for i = " + str(index) + " is below.")
            print(newSquareArray)
        '''
                
        X_new[index, :] = np.reshape(a=newSquareArray, newshape=(1, 49))
    
    return X_new
