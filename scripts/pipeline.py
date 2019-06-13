import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sentences

# load a pandas dataframe containing the sentence pairs
def loader(dataset, sep):
    df = pd.read_csv(dataset, sep=sep)
    estimators = obj.estimators()
    return df, estimators

# create the sentence class and analyse it's features
def createClass(source, target):
    sentence = sentences(source, target)
    sentence.wordCount()
    sentence.getCapitols()

# WIP
    # sentence.tokenize()
    # sentence.getPoSDif('NOUN')
    # sentence.getPoSDif('VERB')
# /WIP
    sentence.characterDifferences('.')
    sentence.characterDifferences(',')
    sentence.characterDifferences(':')
    sentence.characterDifferences(';')
    sentence.characterDifferences('!')
    sentence.characterDifferences('?')
    sentence.characterDifferences('-')
    sentence.characterDifferences('/')
    sentence.characterDifferences('_')
    return sentence


"""
The function main will form a sentence object and run the
analysing functions to derive their describing features.
It will than the run these features through the estimators in
the estimators object and give the estimated confidence scores.
TO DO:
    Weighting functions
"""
def main(sentence, estimators):

    wordcountEst = estimators.wc(sentence.srcWc)
    capitolEst = estimators.capitol(sentence.capitolDif)
    # print(sentence.source)
    # print(sentence.target)
    # print(sentence.srcToks)
    # print(sentence.tgtToks)
    # print(sentence.posDif)
    charEst = {}
    for key in estimators.estimatorDict:
        charEst[key] = estimators.estimatorDict[key](sentence.chars[key])
    estimation = weightEstimation(wordcountEst, capitolEst, charEst)
    return estimation


main(sentence, estimators)


# WIP: will apply weights to the estimations
def weightEstimation(wordCountEst, capitolEst, charEst):
    newList = []
    newList.append(wordCountEst)
    newList.append(capitolEst)
    newList.append(charEst)
    return newList

df,estimators = loader()
iterator = 2376

# sentence = createClass(source_sentence, target_sentence)
sentence = createClass(df['source'][iterator],df['target'][iterator])
