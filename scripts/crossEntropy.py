import numpy as np
import pandas as pd

def correlationMatrix(df, showExtra=False):
    corrMat = df.corr()
    if showExtra == True:
        x = list(dfcorr1.columns)
        for label in x:
            z = list(dfcorr1.columns)
            y = dfcorr1[label]
            xx = sorted(zip(y,z), reverse=True)
            print(label, xx[1])
    return corrMat



def crossEntropy(confScore, correctEst):
    if correctEst == 1:
        return -np.log(confScore)
    else:
        return -np.log(1 - confScore)
