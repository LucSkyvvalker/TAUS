import extractor as ex
import pandas as pd
import cleandatas as cd
import numpy as np
import learnclassify as lc
# from joblib import dump, load


#"../data/en-nl.tsv"

def main():
    try:
        temp = input('Would you like to re-train?[y/n]: ')
        if temp == 'y':
            1/0
        trainX = np.array(pd.read_csv('../data/trainX.csv'))
        trainY =  np.array(pd.read_csv('../data/trainY.csv'),dtype=int)
    except:
        print('Creating new train and test sets.')
        trainX, trainY = createTrainTestSets()
    # trainX = np.array(pd.read_csv('../data/trainX.csv'))
    # trainY =  np.array(pd.read_csv('../data/trainY.csv'), dtype=int)
    trainY = trainY.ravel()
    method = input('What model would you like to train \nSee README for available options: ' )

    if method == 'SVM':
        svm = lc.trainSVM(trainX,trainY)
        lc.savemodel(svm, '../data/svm.joblib')
    elif method == 'LR':
        lr = lc.trainLR(trainX,trainY)
        lc.savemodel(lr, '../data/lr.joblib')
    elif method == 'MLP':
        mlp = lc.trainMLP(trainX,trainY)
        lc.savemodel(mlp, '../data/mlp.joblib')
    elif method == 'NBC':
        NBC = lc.NBC()
        lc.NBC.trainnb(NBC,trainX,trainY)
        lc.savemodel(NBC, '../data/NBC.joblib')
    else:
        print('No valid method was given, please see the README for instrucions on calling main()')
    return



def createTrainTestSets():
    fullDf = pd.read_csv("../data/en-nl.tsv", sep="\t")
    cleandf = cd.cleandata(fullDf)
    try :
        features = pd.read_csv("../data/features.csv")
    except:
        features = extraction(cleandf)
    edited = []
    validation = cleandf['edit_distance'].copy()
    for i in range(len(validation)):
        if validation[i] > 0:
            edited.append(1)
        else:
            edited.append(0)
        if validation[i] > 100:
            validation[i] = 100

    features['edited'] = edited
    features['edit_distance'] = validation

    features = equalizer(np.array(features))
    a,c = split(features,0.8)
    print(len(c), sum(c), set(c))
    return a, c



def extraction(cleandf):
    (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
        unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
        trigramTgtPos) = ex.getNgramModels()
# uncomment when remaking the NgramModels
    # lc.savemodel(unigramSrc, '../data/unigramSrc.joblib')
    # lc.savemodel(bigramSrc, '../data/bigramSrc.joblib')
    # lc.savemodel(trigramSrc, '../data/trigramSrc.joblib')
    # lc.savemodel(unigramTgt, '../data/unigramTgt.joblib')
    # lc.savemodel(bigramTgt, '../data/bigramTgt.joblib')
    # lc.savemodel(trigramTgt, '../data/trigramTgt.joblib')
    # lc.savemodel(unigramSrcPos, '../data/unigramSrcPos.joblib')
    # lc.savemodel(bigramSrcPos, '../data/bigramSrcPos.joblib')
    # lc.savemodel(trigramSrcPos, '../data/trigramSrcPos.joblib')
    # lc.savemodel(unigramTgtPos, '../data/unigramTgtPos.joblib')
    # lc.savemodel(bigramTgtPos, '../data/bigramTgtPos.joblib')
    # lc.savemodel(trigramTgtPos, '../data/trigramTgtPos.joblib')
    dfnew = ex.extractor(cleandf,unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
        unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
        trigramTgtPos)
    dfnew.to_csv('../data/features.csv', encoding='utf-8', index=False)
    return dfnew


def split(x, ratio):
    c = [n for n in range(len(x))]
    r = round(ratio*len(x))
    np.random.shuffle(c)
    d = []
    for elem in c:
        d.append(x[elem])
    train = np.array(d[:r])
    test = np.array(d[r:])
    trainX = train[:, :-2]
    testX = test[:, :-2]
    trainY = train[:, -2]
    testY = test[:, -2]
    trainEditDist = train[:,-1]
    testEditDist = test[:,-1]
    np.savetxt('../data/trainX.csv', trainX, delimiter=',')
    np.savetxt('../data/testX.csv', testX, delimiter=',')
    np.savetxt('../data/trainY.csv', trainY, delimiter=',')
    np.savetxt('../data/testY.csv', testY, delimiter=',')
    np.savetxt('../data/trainEditDist.csv', trainEditDist, delimiter=',')
    np.savetxt('../data/testEditDist.csv', testEditDist, delimiter=',')
    return trainX, trainY

def onlyNoEdits(x):
    y = x[:, -2]
    for n in range(len(y)):
        if y[n] == 0:
            remlest.append(n)
    b = np.delete(x, remlist, axis=0)
    return b

def onlyYesEdits(x):
    y = x[:, -2]
    for n in range(len(y)):
        if y[n] == 1:
            remlest.append(n)
    b = np.delete(x, remlist, axis=0)
    return b

def equalizer(x):
    y = x[:, -2]
    oc = int(sum(y))
    zc = int(len(y) - oc)
    remlist = []
    print(oc, zc)
    if oc > zc:
        for n in range(len(y)):
            if y[n] == 1:
                remlist.append(n)
        np.random.shuffle(remlist)
        remlist = remlist[:(oc - zc)]
    elif oc < zc:
        for n in range(len(y)):
            if y[n] == 0:
                remlist.append(n)
        np.random.shuffle(remlist)
        remlist = remlist[:(zc - oc)]
    remlist = sorted(remlist, reverse=True)
    b = np.delete(x, remlist, axis=0)
    print(len(b))
    return b

if __name__ == "__main__":
    main()
