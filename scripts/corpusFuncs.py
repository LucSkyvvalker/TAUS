import nltk
import ngram as ng

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
