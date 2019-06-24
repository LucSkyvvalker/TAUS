import spacy
import pandas as pd
nlpSrc = spacy.load("en_core_web_sm")
nlpTgt = spacy.load("nl_core_news_sm")

def capitalDif(source, target):
    t1 = 0
    t2 = 0
    for char in str(source):
        if char.isupper():
            t1 += 1
    for char in str(target):
        if char.isupper():
            t2 +=1
    return abs(t1-t2)

# count the differences in punction characters
def characterDifferences(source, target, character):
    t1 = 0
    t2 = 0
    for char in str(source):
        if char == character:
            t1 += 1
    for char in str(target):
        if char == character:
            t2 += 1
    return abs(t1-t2)

# count the differences in punction characters normalized by the amount of characters (no spaces)
def characterDifferencesNormalized(source, target, character):
    norm = len(target.replace(" ", ""))
    return characterDifferences(source, target, character)/norm


# find mismatches in brackets or accolades
def getMismatch(source, target):
    tgtBrack = 0
    tgtAccolade = 0
    srcBrack = 0
    srcAccolade = 0
    # check for mismatches in target sentence
    for char in str(target):
        if char == '(':
            tgtBrack = 1
        if tgtBrack == 1 and char == ')':
            tgtBrack = 0
        if char == '"':
            tgtAccolade = 1
        if tgtAccolade == 1 and char == '"':
            tgtAccolade = 0
    # if mismatch, check if it was present in source
    if tgtBrack == 1 or tgtAccolade == 1:
        for char in str(source):
            if char == '(':
                srcBrack = 1
            if srcBrack == 1 and char == ')':
                srcBrack = 0
            if char == '"':
                srcAccolade = 1
            if srcAccolade == 1 and char == '"':
                srcAccolade = 0
        # if present in src, mismatch is false
        if srcBrack == 1 and tgtBrack == 1:
            tgtBrack = 0
        if srcAccolade == 1 and tgtAccolade == 1:
            tgtAccolade = 0
    return (tgtBrack+tgtAccolade)




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


# run all in order
def main(source, target, tag):
    tokensSrc, tokensTgt = tokenize(source, target)
    tokensSrc, tokensTgt = equalizeTokens(tokensSrc, tokensTgt)
    dif = compareTokens(tokensSrc, tokensTgt, tag)
    return dif
