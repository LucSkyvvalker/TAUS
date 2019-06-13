import math
import numpy as np
import spacy
nlp_en_sm = spacy.load("en_core_web_sm")
nlp_nl_sm = spacy.load("nl_core_news_sm")


class sentences:
"""
The class sentences creates an object where all relevent data for
estimating a confidence score is extracted from a source-target
pair.
Input: source = the original sentence in string format
       target = the proposed MT or TM translation in string format
"""
    # initialize the object
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.chars = {}
        self.posDif = {}

    # calculate word counts and differences
    def wordCount(self):
        source = self.source
        target = self.target
        t1 = len(source.split())
        t2 = len(target.split())
        self.srcWc = t1
        self.tgtWc = t2
        self.wordDif =  abs(t1-t2)

    # count capitols and get their differences
    def getCapitols(self):
        source = self.source
        target = self.target
        t1 = 0
        t2 = 0
        for char in str(source):
            if char.isupper():
                t1 += 1
        for char in str(target):
            if char.isupper():
                t2 +=1
        self.capitolDif = abs(t1-t2)

    # count the differences in punction characters
    def characterDifferences(self, character):
        source = self.source
        target = self.target
        t1 = 0
        t2 = 0
        for char in str(source):
            if char == character:
                t1 += 1
        for char in str(target):
            if char == character:
                t2 += 1
        self.chars[character] = abs(t1-t2)

    # get PoS tokens of both sentences using spacy
    def tokenize(self):
        source = self.source
        target = self.target
        srcTok = []
        tgtTok = []
        doc1 = nlp_en_sm(source)
        doc2 = nlp_nl_sm(target)
        for token in doc1:
            srcTok.append(token.pos_)
        for token in doc2:
            tgtTok.append(token.pos_)
        self.srcToks = srcTok
        self.tgtToks = tgtTok

    # compare source-target based on given token
    def getPoSDif(self, tag):
        source = self.source
        target = self.target
        srcTagCnt = self.srcToks.count(tag)
        tgtTagCnt = self.tgtToks.count(tag)
        self.posDif[tag] = abs(srcTagCnt - tgtTagCnt)


"""
*** These estimators are a work in progress and do by
no means give a accurate estimation.***
The class estimators creates an object that contains all pre-trained
estimator functions to get a confidence score.
"""
class estimators:
    def __init__(self):
        self.wc = np.poly1d([-6.11421177e-05,  3.33835112e-03, -5.80682693e-02,  9.07941961e-01])
        self.capitol = np.poly1d([-0.02718206,  0.03651573,  0.75759027])
        self.estimatorDict = {'.' : np.poly1d([ -5.67977556e-04,   1.62411719e-02,  -1.23279540e-01, 7.52051518e-01]),
                              ',' : np.poly1d([ 0.00943841, -0.11294692,  0.73982521]),
                              ':' : np.poly1d([ 0.08213324, -0.17587561, -0.34565428,  0.73774508]),
                              '!' : np.poly1d([ 0.16111126, -0.67491547,  0.73240526]),
                              ';' : np.poly1d([-0.00667583,  0.13717901, -0.65401396,  0.74167615]),
                              '?' : np.poly1d([-0.1381966 , -0.1381966 ,  0.73148303]),
                              '-' : np.poly1d([ 0.00551715, -0.10479321,  0.75116463]),
                              '/' : np.poly1d([ -4.38319547e-04,   3.66597739e-02,  -3.49626968e-01, 7.32818816e-01]),
                              '_' : np.poly1d([ 0.00180061, -0.13545186,  0.7314145 ])
                             }
