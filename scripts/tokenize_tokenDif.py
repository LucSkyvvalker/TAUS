import spacy
import pandas as pd
nlpSrc = spacy.load("en_core_web_sm")
nlpTgt = spacy.load("nl_core_news_sm")

"""
tokenize() takes a source target sentence pair and uses spacy
to create two lists containing the tokens in each
"""
def tokenize(source, target):
    source = source.lower()
    target = source.lower()
    tokensSrc = []
    tokensTgt = []
    # check for empty source
    if pd.isnull(source):
        pass
    else: 
        doc0 = nlpSrc(source)
        doc1 = nlpTgt(target)
        for token in doc0:
            tokensSrc.append(token.pos_)
        for token in doc1:
            tokensTgt.append(token.pos_)
    
    return tokensSrc,tokensTgt

# bundle tokens, add more for loops if needed 
def equalizeTokens(tokensSrc, tokensTgt):
    for i in range(len(tokensSrc)):
        if tokensSrc[i] == "PROPN":
            tokensSrc[i] = "NOUN"
    for i in range(len(tokensTgt)):
        if tokensTgt[i] == "PROPN":
            tokensTgt[i] = "NOUN"
    #repeat for any new token
    return tokensSrc, tokensTgt

# calculate the difference in given token
def compareTokens(tokensSrc, tokensTgt, tag):
    return abs(tokensSrc.count(tag) - tokensTgt.count(tag))

# calculate the percentage of the given tag
def percentageOfTag(tokensSrc, tokensTgt, tag):
    tokenCountSrc = 0
    tokenCountTgt = 0
    for i in range(len(tokensSrc)):
        if tokensSrc[i] == tag:
            tokenCountSrc += 1
    for i in range(len(tokensTgt)):
        if tokensTgt[i] == tag:
            tokenCountTgt += 1
    # returns two percentages, one for the source and the second for the target     
    return tokenCountSrc/len(tokensSrc), tokenCountTgt/len(tokensTgt)

# run all in order
def main(source, target, tag):
    tokensSrc, tokensTgt = tokenize(source, target)
    tokensSrc, tokensTgt = equalizeTokens(tokensSrc, tokensTgt)
    dif = compareTokens(tokensSrc, tokensTgt, tag)
    srcNouns, tgtNouns = percentageOfToken(tokensSrc, tokensTgt, "NOUN")
    srcVerbs, tgtVerbs = percentageOfToken(tokensSrc, tokensTgt, "VERB")
    return dif
