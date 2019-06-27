import extractSents as es
import pandas as pd
import numpy as np
import cleandatas as cd
import learnclassify as lc

def confidenceScore():
    fullDf = pd.read_csv("../data/en-nl.tsv", sep="\t")
    cleandf = cd.cleandata(fullDf)
    (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
    unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
    trigramTgtPos) = loadNLP()
    method = input('What model would you like to use \nSee README for available options: ' )
    while True:
        source = input('Source: \n')
        target = input('Target: \n')
        cl, score = scoreder(cleandf, source, target, method,unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,trigramTgtPos)
        if cl[0] == 0:
            print((score[0]*100))
        else:
            print(1-score[0])
        print('\n Press ctrl+c to quit \n')
    return


def scoreder(cleandf,source, target, method,unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,trigramTgtPos):
    sentFeats = np.array(es.extractor2(cleandf, source, target, unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
            unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
            trigramTgtPos))
    if method == 'SVM':
        svm = lc.loadmodel('../data/svm.joblib')
        score = lc.classCLF(svm, sentFeats)
    elif method == 'LR':
        lr = lc.loadmodel('../data/lr.joblib')
        score = lc.classCLF(lr, sentFeats)
    elif method == 'MLP':
        mlp = lc.loadmodel('../data/mlp.joblib')
        score = lc.classCLF(mlp, sentFeats)
    elif method == 'NBC':
        NBC = lc.loadmodel('../data/NBC.joblib')
        score = lc.NBC.classnb(NBC,sentFeats)
        print(score)
    else:
        print('No valid method was given, please see the README for instrucions')
    return score

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

if __name__ == "__main__":
    confidenceScore()
