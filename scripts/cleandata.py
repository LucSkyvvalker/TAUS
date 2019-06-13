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