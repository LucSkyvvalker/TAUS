import pandas as pd
import math
import spacy
import numpy as np
import nltk
import time
import matplotlib.pyplot as plt

# cell to import all functions
import helpFuncs as hf
import cleandatas as cd
import ngram as ng
import learnclassify as lc
import corpusFuncs as corpf

fullDf = pd.read_csv("../data/en-nl.tsv", sep="\t")
cleandf = cd.cleandata(fullDf)

corpus_en = '../data/brown.txt'
corpus_nl = '../data/kat.txt'
corpus_en_pos = '../data/sample_corpus_pos.en'
corpus_nl_pos = '../data/posCorp.txt'

(unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
        unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
        trigramTgtPos) = corpf.trainCorpera(corpus_en, corpus_nl, corpus_en_pos, corpus_nl_pos)



def extractor2(source, target):
    timeStart = time.time()
    # allocating arrays
    capitalCountDif = []

    commaDif = []
    exclamationDif = []
    questionmarkDif = []
    dotDif = []
    hyphenDif = []
    underscoreDif = []
    slashDif = []
    colonDif = []
    semicolonDif = []

    commaDifNorm = []
    exclamationDifNorm = []
    questionmarkDifNorm = []
    dotDifNorm = []
    hyphenDifNorm = []
    underscoreDifNorm = []
    slashDifNorm = []
    colonDifNorm = []
    semicolonDifNorm = []

    misMatch = []

    verbDif = []
    nounDif = []

    logPerpTgt0 = []
    logPerpSrcTok0 = []
    logPerpTgtTok0 = []

    logPerpTgt1 = []
    logPerpSrcTok1 = []
    logPerpTgtTok1 = []

    logPerpTgt2 = []
    logPerpSrcTok2 = []
    logPerpTgtTok2 = []

    logProbTgt0 = []
    logProbSrcTok0 = []
    logProbTgtTok0 = []

    logProbTgt1 = []
    logProbSrcTok1 = []
    logProbTgtTok1 = []

    logProbTgt2 = []
    logProbSrcTok2 = []
    logProbTgtTok2 = []

    unk_ngramsTgt0 = []
    unk_ngramsSrcTok0 = []
    unk_ngramsTgtTok0 = []

    unk_ngramsTgt1 = []
    unk_ngramsSrcTok1 = []
    unk_ngramsTgtTok1 = []

    unk_ngramsTgt2 = []
    unk_ngramsSrcTok2 = []
    unk_ngramsTgtTok2 = []

    # filling all lists to make new dataframe
    capitalCountDif.append(hf.capitalDif(source,target))
    commaDif.append(hf.characterDifferences(source, target, ','))
    exclamationDif.append(hf.characterDifferences(source, target, '!'))
    questionmarkDif.append(hf.characterDifferences(source, target, '?'))
    dotDif.append(hf.characterDifferences(source, target, '.'))
    hyphenDif.append(hf.characterDifferences(source, target, '-'))
    underscoreDif.append(hf.characterDifferences(source, target, '_'))
    slashDif.append(hf.characterDifferences(source, target, '/'))
    colonDif.append(hf.characterDifferences(source, target, ':'))
    semicolonDif.append(hf.characterDifferences(source, target, ';'))
    commaDifNorm.append(hf.characterDifferencesNormalized(source, target, ','))
    exclamationDifNorm.append(hf.characterDifferencesNormalized(source, target, '!'))
    questionmarkDifNorm.append(hf.characterDifferencesNormalized(source, target, '?'))
    dotDifNorm.append(hf.characterDifferencesNormalized(source, target, '.'))
    hyphenDifNorm.append(hf.characterDifferencesNormalized(source, target, '-'))
    underscoreDifNorm.append(hf.characterDifferencesNormalized(source, target, '_'))
    slashDifNorm.append(hf.characterDifferencesNormalized(source, target, '/'))
    colonDifNorm.append(hf.characterDifferencesNormalized(source, target, ':'))
    semicolonDifNorm.append(hf.characterDifferencesNormalized(source, target, ';'))
    misMatch.append(hf.getMismatch(source, target))

    tokensSrc,tokensTgt = hf.tokenize(source, target)
    tokensSrc, tokensTgt = hf.equalizeTokens(tokensSrc, tokensTgt)
    verbDif.append(hf.compareTokens(tokensSrc, tokensTgt, 'VERB'))
    nounDif.append(hf.compareTokens(tokensSrc, tokensTgt, 'NOUN'))

    #logPerpSrc0.append(ng.log_ppl(unigramSrc, source ))
    logPerpTgt0.append(ng.log_ppl(unigramTgt, target ))
    logPerpSrcTok0.append(ng.log_ppl(unigramSrcPos, " ".join(tokensSrc)))
    logPerpTgtTok0.append(ng.log_ppl(unigramTgtPos, " ".join(tokensTgt)))

    #logPerpSrc1.append(ng.log_ppl(bigramSrc, source ))
    logPerpTgt1.append(ng.log_ppl(bigramTgt, target))
    logPerpSrcTok1.append(ng.log_ppl(bigramSrcPos, " ".join(tokensSrc)))
    logPerpTgtTok1.append(ng.log_ppl(bigramTgtPos, " ".join(tokensTgt)))

    #logPerpSrc2.append(ng.log_ppl(trigramSrc, source ))
    logPerpTgt2.append(ng.log_ppl(trigramTgt, target))
    logPerpSrcTok2.append(ng.log_ppl(trigramSrcPos, " ".join(tokensSrc)))
    logPerpTgtTok2.append(ng.log_ppl(trigramTgtPos, " ".join(tokensTgt)))

    #logProbSrc0.append(ng.log_prob(unigramSrc, source ))
    logProbTgt0.append(ng.log_prob(unigramTgt, target))
    logProbSrcTok0.append(ng.log_prob(unigramSrcPos, " ".join(tokensSrc)))
    logProbTgtTok0.append(ng.log_prob(unigramTgtPos, " ".join(tokensTgt)))

    #logProbSrc1.append(ng.log_prob(bigramSrc, source ))
    logProbTgt1.append(ng.log_prob(bigramTgt, target))
    logProbSrcTok1.append(ng.log_prob(bigramSrcPos, " ".join(tokensSrc)))
    logProbTgtTok1.append(ng.log_prob(bigramTgtPos, " ".join(tokensTgt)))

    #logProbSrc2.append(ng.log_prob(trigramSrc, source ))
    logProbTgt2.append(ng.log_prob(trigramTgt, target))
    logProbSrcTok2.append(ng.log_prob(trigramSrcPos, " ".join(tokensSrc)))
    logProbTgtTok2.append(ng.log_prob(trigramTgtPos, " ".join(tokensTgt)))

    #unk_ngramsSrc0.append(ng.unk_ngrams(unigramSrc, target))
    unk_ngramsTgt0.append(ng.unk_ngrams(unigramTgt, target))
    unk_ngramsSrcTok0.append(ng.unk_ngrams(unigramSrcPos, " ".join(tokensSrc)))
    unk_ngramsTgtTok0.append(ng.unk_ngrams(unigramTgtPos, " ".join(tokensTgt)))

    #unk_ngramsSrc1.append(ng.unk_ngrams(bigramSrc, target))
    unk_ngramsTgt1.append(ng.unk_ngrams(bigramTgt, target))
    unk_ngramsSrcTok1.append(ng.unk_ngrams(bigramSrcPos, " ".join(tokensSrc)))
    unk_ngramsTgtTok1.append(ng.unk_ngrams(bigramTgtPos, " ".join(tokensTgt)))

    #unk_ngramsSrc2.append(ng.unk_ngrams(trigramSrc, target))
    unk_ngramsTgt2.append(ng.unk_ngrams(trigramTgt, target))
    unk_ngramsSrcTok2.append(ng.unk_ngrams(trigramSrcPos, " ".join(tokensSrc)))
    unk_ngramsTgtTok2.append(ng.unk_ngrams(trigramTgtPos, " ".join(tokensTgt)))

    temp = {'capitalCountDif' : capitalCountDif,
        'commaDif' : commaDif,
        'exclamationDif' : exclamationDif,
        'questionmarkDif' : questionmarkDif,
        'dotDif' : dotDif,
        'hyphenDif' : hyphenDif,
        'underscoreDif' : underscoreDif,
        'slashDif' : slashDif,
        'colonDif' : colonDif,
        'semicolonDif' : semicolonDif,
        'commaDifNorm' : commaDifNorm,
        'exclamationDifNorm' : exclamationDifNorm,
        'questionmarkDifNorm' : questionmarkDifNorm,
        'dotDifNorm' : dotDifNorm,
        'hyphenDifNorm' : hyphenDifNorm,
        'underscoreDifNorm' : underscoreDifNorm,
        'slashDifNorm' : slashDifNorm,
        'colonDifNorm' : colonDifNorm,
        'semicolonDifNorm' : semicolonDifNorm,
        'misMatch' : misMatch,
        'verbDif' : verbDif,
        'nounDif' : nounDif,
        #'logPerpSrc0' : logPerpSrc0,
        'logPerpTgt0' : logPerpTgt0,
        'logPerpSrcTok0' : logPerpSrcTok0,
        'logPerpTgtTok0' : logPerpTgtTok0,
        #'logPerpSrc1' : logPerpSrc1,
        'logPerpTgt1':logPerpTgt1,
        'logPerpSrcTok1' :logPerpSrcTok1,
        'logPerpTgtTok1' : logPerpTgtTok1,
        #'logPerpSrc2' : logPerpSrc2,
        'logPerpTgt2' : logPerpTgt2,
        'logPerpSrcTok2' : logPerpSrcTok2,
        'logPerpTgtTok2' : logPerpTgtTok2,
        #'logProbSrc0' : logProbSrc0,
        'logProbTgt0' :logPerpTgtTok2,
        'logProbSrcTok0' : logProbSrcTok0,
        'logProbTgtTok0' : logProbTgtTok0,
        #'logProbSrc1' : logProbSrc1,
        'logProbTgt1' :logProbTgt1,
        'logProbSrcTok1' :logProbSrcTok1,
        'logProbTgtTok1' :logProbTgtTok1,
        #'logProbSrc2' : logProbSrc2,
        'logProbTgt2' : logProbTgtTok1,
        'logProbSrcTok2': logProbSrcTok2,
        'logProbTgtTok2' : logProbTgtTok2,
        #'unk_ngramsSrc0' : unk_ngramsSrc0,
        'unk_ngramsTgt0' : unk_ngramsTgt0,
        'unk_ngramsSrcTok0' : unk_ngramsSrcTok0,
        'unk_ngramsTgtTok0' :unk_ngramsTgtTok0,
        #'unk_ngramsSrc1' : unk_ngramsSrc1,
        'unk_ngramsTgt1':unk_ngramsTgt1,
        'unk_ngramsSrcTok1' :unk_ngramsSrcTok1,
        'unk_ngramsTgtTok1' :unk_ngramsTgtTok1,
        #'unk_ngramsSrc2' : unk_ngramsSrc2,
        'unk_ngramsTgt2' :unk_ngramsTgt2,
        'unk_ngramsSrcTok2' :unk_ngramsSrcTok2,
        'unk_ngramsTgtTok2' :unk_ngramsTgtTok2
    }
    dfnew = pd.DataFrame(temp)
    dfnew['wordCountSrc'] = cleandf['source_wc']
    dfnew['wordCountTgt'] = cleandf['target_wc']
    dfnew['wordCountDif'] = abs(cleandf['source_wc'] - cleandf['target_wc'])
    timeCurrent = time.time()
    string = 'elapsed time: ' + str(timeCurrent - timeStart)
    print(string, end="\r")
    return dfnew
extracted_zin = extractor2("A sentence that is very very very absurdly long for no purpose except analysing the computional time of our alghorithm",
"Een zin die absurd lang is, met geen betekenis behalve het vergaren van behoeftige kennis over de computationele complexiteit van onze extractie stappen")
