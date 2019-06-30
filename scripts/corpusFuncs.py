import nltk
import ngram as ng
import learnclassify as lc

def trainCorpera(corpus_en, corpus_nl, corpus_en_pos, corpus_nl_pos):
    unigramSrc = ng.trainLM(0, corpus_en)
    bigramSrc = ng.trainLM(1, corpus_en)
    trigramSrc = ng.trainLM(2, corpus_en)

    unigramTgt = ng.trainLM(0, corpus_nl)
    bigramTgt = ng.trainLM(1, corpus_nl)
    trigramTgt = ng.trainLM(2, corpus_nl)

    unigramSrcPos = ng.trainLM(0, corpus_en_pos)
    bigramSrcPos = ng.trainLM(1, corpus_en_pos)
    trigramSrcPos = ng.trainLM(2, corpus_en_pos)

    unigramTgtPos = ng.trainLM(0, corpus_nl_pos)
    bigramTgtPos = ng.trainLM(1, corpus_nl_pos)
    trigramTgtPos = ng.trainLM(2, corpus_nl_pos)
    return (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
            unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
            trigramTgtPos)

def getNgramModels():
    corpus_en = '../data/brown.txt'
    corpus_nl = '../data/kat.txt'
    corpus_en_pos = '../data/sample_corpus_pos.en'
    corpus_nl_pos = '../data/posCorp.txt'

    (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
        unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
        trigramTgtPos) = trainCorpera(corpus_en, corpus_nl, corpus_en_pos, corpus_nl_pos)

    return (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
            unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
            trigramTgtPos)

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
    return (unigramSrc,bigramSrc, trigramSrc, unigramTgt,bigramTgt,trigramTgt,
            unigramSrcPos,bigramSrcPos,trigramSrcPos,unigramTgtPos,bigramTgtPos,
            trigramTgtPos)
