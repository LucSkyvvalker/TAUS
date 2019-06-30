import extractSents as es
import corpusFuncs as corpf
import pandas as pd
import numpy as np
import cleandatas as cd
import learnclassify as lc

def scoreInput():
    try:
        cleandf = pd.read_csv('../data/cleandf.csv',dtype=object)
    except:
        fullDf = pd.read_csv("../data/en-nl.tsv", sep="\t")
        cleandf = cd.cleandata(fullDf)
        cleandf.to_csv('../data/cleandf.csv')
    try:
        (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
        unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
        trigramTgtPos) = corpf.loadNLP()
    except:
        (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
                unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
                trigramTgtPos) = corpf.getNgramModels()
    method = input('What model would you like to use \nSee README for available options: ' )
    while True:
        source = input('Source: \n')
        target = input('Target: \n')
        cl, score = scoreder(cleandf, source, target, method,unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,trigramTgtPos)
        if cl[0] == 0:
            print('Confidence Score:',(score[0]*100))
        else:
            print('Confidence Score:', 1-score[0])
        print('\n Press ctrl+c to quit \n')
    return

# classifies the features based on the given method
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

if __name__ == "__main__":
    scoreInput()
