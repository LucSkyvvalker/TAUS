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
        testX =  np.array(pd.read_csv('../data/testX.csv'))
        trainY =  np.array(pd.read_csv('../data/trainY.csv'))
        testY =  np.array(pd.read_csv('../data/testY.csv'))
    except:
        print('Creating new train and test sets.')
        createTrainTestSets()
    trainX = np.array(pd.read_csv('../data/trainX.csv'))
    testX =  np.array(pd.read_csv('../data/testX.csv'))
    trainY =  np.array(pd.read_csv('../data/trainY.csv'), dtype=int)
    trainY = trainY.ravel()
    print(trainY)
    print(len(trainY), sum(trainY), set(trainY))
    testY =  np.array(pd.read_csv('../data/testY.csv'))
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
    validation = cleandf['edit_distance'].copy()
    for i in range(len(validation)):
        if validation[i] > 0:
            validation[i] = 1
    features['y'] = validation
    features = equalizer(np.array(features))
    print(len(features), sum(features[:,-1]), set(features[:,-1]))
    a,b,c,d = split(features,0.8)
    print(len(c), sum(c), set(c))
    return



def extraction(cleandf):
    (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
        unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
        trigramTgtPos) = ex.getNgramModels()
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
    trainX = train[:, :-1]
    testX = test[:, :-1]
    trainY = train[:, -1]
    testY = test[:, -1]
    np.savetxt('../data/trainX.csv', trainX, delimiter=',')
    np.savetxt('../data/testX.csv', testX, delimiter=',')
    np.savetxt('../data/trainY.csv', trainY, delimiter=',')
    np.savetxt('../data/testY.csv', testY, delimiter=',')
    return trainX, testX, trainY, testY

def equalizer(x):
    y = x[:, -1]
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
