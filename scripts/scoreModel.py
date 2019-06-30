import extractSents as es
import pandas as pd
import numpy as np
import cleandatas as cd
import learnclassify as lc
import score as sc
import corpusFuncs as corpf


"""
scoreModel() is used to get a print of:
'Precision, Recall
F1-score
Accuracy
Cross Entropy'
The function will ask for input on what
"""
def scoreModel():
    try:
        testX = np.array(pd.read_csv('../data/testX.csv'))
        testY =  np.array(pd.read_csv('../data/testY.csv'),dtype=int)
    except:
        print("No valid datasets were found")
    try:
        (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
        unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
        trigramTgtPos) = corpf.loadNLP()
    except:
        print('No ngram models were found, making new ones...')
        (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
        unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
        trigramTgtPos) = corpf.getNgramModels()
    method = input('What model would you like to use \nSee README for available options: ' )
    if method == 'SVM':
        svm = lc.loadmodel('../data/svm.joblib')
        scores = lc.classCLF(svm, testX)
    elif method == 'LR':
        lr = lc.loadmodel('../data/lr.joblib')
        scores = lc.classCLF(lr, testX)
    elif method == 'MLP':
        mlp = lc.loadmodel('../data/mlp.joblib')
        scores = lc.classCLF(mlp, testX)
    elif method == 'NBC':
        NBC = lc.loadmodel('../data/NBC.joblib')
        scores = lc.NBC.classnb(NBC,testX)
    else:
        print('No valid method was given, please see the README for instrucions')
    fscore = sc.fscore(scores[0], testY)
    print('F1score:',fscore)
    accuracy = sc.accuracy(scores[0], testY)
    print('Accuracy:', accuracy)
    crossent = []
    for i in range(len(testY)):
        correctEst = 0
        if scores[0][i] == testY[i]:
            correctEst = 1
        crossent.append(sc.crossEntropy(scores[1][i], correctEst))
    print('CrossEnt:', np.mean(crossent))
    return fscore, method, scores, crossent




if __name__ == "__main__":
    scoreModel()
