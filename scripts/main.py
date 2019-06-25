import extractor as ex
import pandas as pd
import cleandatas as cd
import numpy as np


#"../data/en-nl.tsv"
def main():
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

    idx = np.arrange(len(features))
    shuffleIdx = np.random.shuffle(idx)
    featureNew = features.iloc[shuffleIdx]
    validationNew = features.iloc[shuffleIdx]

    train, test, trainVal, testVal  = split(features, validation)
    train.to_csv('../data/train.csv', encoding='utf-8', index=False)
    test.to_csv('../data/test.csv', encoding='utf-8', index=False)
    trainVal.to_csv('../data/trainVal.csv', encoding='utf-8', index=False)
    testVal.to_csv('../data/testVal.csv', encoding='utf-8', index=False)



def extraction(cleandf):
    (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
        unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
        trigramTgtPos) = ex.getNgramModels()
    dfnew = ex.extractor(cleandf,unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
        unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
        trigramTgtPos)
    dfnew.to_csv('../data/features.csv', encoding='utf-8', index=False)
    return dfnew


def equalize(validation):
    t0 = validation.value_counts()
    noEdit = t0[0]
    yesEdit = t0[1]
    if noEdit > yesEdit:

    else:


if __name__ == "__main__":
    main()
