import extractSents as es
import pandas as pd
import numpy as np
import cleandatas as cd
import learnclassify as lc
import score as sc
import corpusFuncs as corpf


def scoreSentences():
    print("Warning this takes quite a while")
    try:
        cleandf = pd.read_csv('../data/cleandf.csv', dtype=object)
    except:
        fullDf = pd.read_csv("../data/en-nl.tsv", sep="\t")
        cleandf = cd.cleandata(fullDf)
        cleandf.to_csv('../data/cleandf.csv')
    try:
        (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
        unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
        trigramTgtPos) = corpf.loadNLP()
    except:
        print('No ngram models were found, making new ones...')
        (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
        unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
        trigramTgtPos) = corpf.getNgramModels()
    try:
        testX = np.array(pd.read_csv('../data/testX.csv'))
        testY =  np.array(pd.read_csv('../data/testY.csv'),dtype=int)
    except:
        print("No valid datasets were found")

    scores = scoreder(cleandf,testX)
    fscore = sc.fscore(scores[0], testY)
    print('F1-score:', fscore)
    accuracy = sc.accuracy(scores[0], testY)
    print('Accuracy', accuracy)
    crossent = []
    for i in range(len(testY)):
        correctEst = 0
        if scores[0][i] == testY[i]:
            correctEst = 1
        crossent.append(sc.crossEntropy(scores[1][i], correctEst))
    print('Cross Entropy:', np.mean(crossent))
    return

# uses 2 models to check on the low edit distances
def scoreder(cleandf,sentFeats):
    newscores = [[],[]]
    NBC = lc.loadmodel('../data/NBC.joblib')
    lr = lc.loadmodel('../data/lr.joblib')
    for i in range(len(sentFeats)):
        score = lc.NBC.classnb(NBC,[sentFeats[i]])
        if score[0][0] == 0:
            score = lc.classCLF(lr, sentFeats)
        newscores[0].append(score[0][0])
        newscores[1].append(score[1][0])
    return newscores


if __name__ == "__main__":
    scoreSentences()
