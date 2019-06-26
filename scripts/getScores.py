import extractSents as es
import pandas as pd
import numpy as np
import cleandatas as cd
import learnclassify as lc
import score as sc

def getScores():
    fullDf = pd.read_csv("../data/en-nl.tsv", sep="\t")
    cleandf = cd.cleandata(fullDf)
    (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
    unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
    trigramTgtPos) = loadNLP()
    method = "LR"
    testX = np.array(pd.read_csv('../data/testX.csv'))
    testY =  np.array(pd.read_csv('../data/testY.csv'),dtype=int)
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
    print(fscore)
    crossent = []
    for i in range(len(testY)):
        correctEst = 0
        if scores[0][i] == testY[i]:
            correctEst = 1
        crossent.append(sc.crossEntropy(scores[1][i], correctEst))
    print(np.mean(crossent))
    return fscore

def loadNLP():
    unigramSrc = lc.loadmodel('../data/unigramSrc.joblib')
    bigramSrc = lc.loadmodel('../data/bigramSrc.joblib')
    trigramSrc = lc.loadmodel('../data/trigramSrc.joblib')
    unigramTgt = lc.loadmodel('../data/unigramTgt.joblib')
    bigramTgt = lc.loadmodel('../data/bigramTgt.joblib')
    trigramTgt = lc.loadmodel('../data/trigramTgt.joblib')
    unigramSrcPos = lc.loadmodel('../data/unigramSrcPos.joblib')
    bigramSrcPos = lc.loadmodel('../data/bigramSrcPos.joblib')
    trigramSrcPos = lc.loadmodel('../data/trigramSrcPos.joblib')
    unigramTgtPos = lc.loadmodel('../data/unigramTgtPos.joblib')
    bigramTgtPos = lc.loadmodel('../data/bigramTgtPos.joblib')
    trigramTgtPos = lc.loadmodel('../data/trigramTgtPos.joblib')
    return unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,trigramTgtPos



getScores()
