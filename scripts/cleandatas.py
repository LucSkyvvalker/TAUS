import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

def droptaglist(dt, tag, taglist):
    a = list(dt[tag])
    b = []
    for n in range(len(a)):
        if str(a[n]) not in taglist:
            b.append(n)
    dt = dt.drop(b)
    dt = dt.reset_index(drop=True)
    return dt
        
def gettaglist(dt, tag, taglist):
    a = list(dt[tag])
    b = []
    for n in range(len(a)):
        if str(a[n]) in taglist:
            b.append(n)
    dt = dt.drop(b)
    dt = dt.reset_index(drop=True)
    return dt

def charin(dt, tag, charlist):
    a = list(dt[tag])
    b = []
    for n in range(len(a)):
        notfound = True
        m = 0
        sentence = str(a[n])
        while notfound and m < len(sentence):
            if sentence[m] in charlist:
                notfound = False
                b.append(n)
            m += 1
    dt = dt.drop(b)
    dt = dt.reset_index(drop=True)
    return dt

def charout(dt, tag, charlist):
    a = list(dt[tag])
    b = []
    for n in range(len(a)):
        notfound = True
        m = 0
        sentence = str(a[n])
        while notfound and m < len(sentence):
            if sentence[m] in charlist:
                notfound = False
            m += 1
        if m == len(sentence):
            b.append(n)
    dt = dt.drop(b)
    dt = dt.reset_index(drop=True)
    return dt

def dropstring(dt, tag, string):
    d = list(dt[tag])
    b = []
    start = len(string) - 1
    for a in range(len(d)):
        n = start
        wf = True
        sentence = str(d[a])
        while n < len(sentence) and wf: 
            m = 0
            wl = True
            while m < len(string) and wl:
                if sentence[n - m] != string[-1 - m]:
                    wl = False
                m += 1
                if m == len(string) and wl:
                    wf = False
                    b.append(a)
            n += 1
    dt = dt.drop(b)
    dt = dt.reset_index(drop=True)
    return dt

def getstring(dt, tag, string):
    d = list(dt[tag])
    b = []
    start = len(string) - 1
    for a in range(len(d)):
        n = start
        wf = True
        sentence = str(d[a])
        while n < len(sentence) and wf: 
            m = 0
            wl = True
            while m < len(string) and wl:
                if sentence[n - m] != string[-1 - m]:
                    wl = False
                m += 1
                if m == len(string) and wl:
                    wf = False
            n += 1
        if wf:
            b.append(a)
    dt = dt.drop(b)
    dt = dt.reset_index(drop=True)
    return dt

def cleandata(dt, indexin=[], indexout=[], origin=["TM", "MT"], MR=70, MRO=False, 
              maxSWC=70, maxTWC=70, maxEWC=70, editdismin=-1, editdismax=-1, 
              content=[], industry=[], engine=[], charins=[], charexs=["<", ">"],
             charint=[], charext=[], wordin=False, wordout=False):
    #
    # Automatically uses recommended setting to clean data
    #
    # dt - The data that will be cleaned
    # indexin - Removes all indexes not in indexin
    # indexout - Removes all indexes in indexout not compatible with indexin
    # origin - Removes all rows with row[origin] not in origin
    # MR - Removes rows with match_rate values lower than MR if origin is TM
    # MRO - Removes rows with match_rate values lower than MR if origin is TM or MRO
    # maxSWC - Removes all rows with higher source word count than maxSWC
    # maxTWC - Removes all rows with higher target word count than maxTWC
    # maxEWC - Removes all rows with higher edited word count than maxEWC
    # editdismin - Removes all rows with lower edit_distance. 
    # Negative values for editdismin won't do anything
    # editdismax - Removes all rows with Higher edit_distance. 
    # Negative values for editdismax won't do anything
    # content - Removes all rows with row[content] not in content
    # industry - Removes all rows with row[industry] not in industry
    # engine - Removes all rows with row[engine] not in engine
    # charins - Removes all rows not containing a char in charins in row[source]
    # charexs - Removes all rows containing a char in charexs in row[source]
    # charint - Removes all rows not containing a char in charins in row[target]
    # charext - Removes all rows containing a char in charexs in row[target]
    # wordin - Removes all rows not containing string wordin in row[source]
    # wordinout - Removes all rows containing string wordout in row[source]
    #

    if indexin:
        b = []
        for n in range(len(dt.index)):
            if dt.index[n] not in indexin:
                b.append(dt.index[n])
        dt = dt.drop(b)
        dt = dt.reset_index(drop=True)

    elif indexout:
        dt = dt.drop(indexout)
        dt = dt.reset_index(drop=True)

    if origin:
        dt = droptaglist(dt, "origin", origin)

    if MR:
        p = (dt["match_rate"] >= MR)
        if MRO:
            q = (dt["origin"] != "TM" and dt["origin"] != MRO)
        else:
            q = (dt["origin"] != "TM")
        dt = dt[p|q]
        dt = dt.reset_index(drop=True)

    if maxSWC and maxTWC and maxEWC:
        swc = (dt["source_wc"] <= maxSWC)
        twc = (dt["target_wc"] <= maxTWC)
        ewc = (dt["edit_wc"] <= maxEWC)
        dt = dt[swc&twc&ewc]
        dt = dt.reset_index(drop=True)

    if editdismin >= 0:
        dt = dt[dt["edit_distance"] >= editdismin]
        dt = dt.reset_index(drop=True)

    if editdismax >= 0:
        dt = dt[dt["edit_distance"] <= editdismax]
        dt = dt.reset_index(drop=True)

    if content:
        dt = droptaglist(dt, "content_type", content)

    if industry:
        dt = droptaglist(dt, "industry", industry)

    if engine:
        dt = droptaglist(dt, "mt_engine", engine)

    if charins:
        dt = charout(dt, "source", charins)

    if charexs:
        dt = charin(dt, "source", charexs)

    if charint:
        dt = charout(dt, "target", charins)

    if charext:
        dt = charin(dt, "target", charexs)

    if wordin:
        dt = getstring(dt, "source", wordin)

    if wordout:
        dt = dropstring(dt, "source", wordout)


    return dt