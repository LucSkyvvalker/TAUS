import extractSents as es
import pandas as pd
import numpy as np
import cleandatas as cd
import learnclassify as lc
import score as sc

def confidenceScore2():
    fullDf = pd.read_csv("../data/en-nl.tsv", sep="\t")
    cleandf = cd.cleandata(fullDf)
    (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
    unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
    trigramTgtPos) = loadNLP()
    testX = np.array(pd.read_csv('../data/testX.csv'))
    testY =  np.array(pd.read_csv('../data/testY.csv'),dtype=int)
    scores = scoreder(cleandf,testX)
    fscore = sc.fscore(scores[0], testY)
    print(fscore)
    crossent = []
    for i in range(len(testY)):
        correctEst = 0
        if scores[0][i] == testY[i]:
            correctEst = 1
        crossent.append(sc.crossEntropy(scores[1][i], correctEst))
    print('foo', np.mean(crossent))
    return


def scoreder(cleandf,sentFeats):
    newscores = [[],[]]
    NBC = lc.loadmodel('../data/NBC.joblib')
    lr = lc.loadmodel('../data/lr.joblib')
    for i in range(len(sentFeats)):
        print(i,'/',len(sentFeats))
        score = lc.NBC.classnb(NBC,[sentFeats[i]])
        if score[0][0] == 0:
            score = lc.classCLF(lr, sentFeats)
        newscores[0].append(score[0][0])
        newscores[1].append(score[1][0])
    return newscores

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
    confidenceScore2()
