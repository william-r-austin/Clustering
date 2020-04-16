'''
Created on Apr 13, 2020

@author: William
'''

import cs584.project3.kmeans_digits as kmdigits

def runSubmission01():
    kmdigits.submission01(True, 5)

def runSubmission02():
    kmdigits.submission02(True, 5)

def runTnseBasic():
    kmdigits.tuneTsneBasic(True)
    
def runTnseBisecting():
    kmdigits.tuneTsneBisecting(True)    

if __name__ == '__main__':
    runTnseBisecting()
