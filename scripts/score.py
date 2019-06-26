import numpy as np
import pandas as pd


def accuracy(x, y):
    tot = len(x)
    cor = 0
    for n in range(len(x)):
        if x[n] == y[n]:
            cor += 1
    return cor/tot

def recall(x, y):
    tp = 0
    fn = 0
    for n in range(len(x)):
        if x[n] == 1:
            if y[n] == 1:
                tp += 1
            else:
                fn += 1
    return tp/(tp + fn)

def precision(x, y):
    tp = 0
    fp = 0
    for n in range(len(x)):
        if y[n] == 1:
            if x[n] == 1:
                tp += 1
            else:
                fp += 1
    return tp/(tp + fp)

def fscore(x, y):
    prec = precision(x, y)
    recl = recall(x, y)
    print(prec, recl)
    return 2 * (prec * recl) / (prec + recl)

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
