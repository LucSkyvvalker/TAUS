import pandas as pd
import math
import numpy as np
import nltk
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from scipy.special import logit
from scipy.optimize import minimize
from scipy import optimize
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
from joblib import dump
from joblib import load

x = [[0,0], [0,1], [0,2], [1, 0], [1,1], [1, 2], [2, 0], [2,1], [2, 2]]
y = [0, 0, 0, 1, 1, 1, 1, 1, 1]
points = [[0, 3], [1, 3], [2, 3]]

def trainSVM(x, y):
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(x, y)
    return clf

def trainMLP(x, y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(x, y)
    return clf

def trainLR(x, y):
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial', max_iter=10000)
    clf.fit(x, y)
    return clf

def classCLF(clf, points):
    c = clf.predict(points)
    p = clf.predict_proba(points)
    rp = []
    for pp in p:
        rp.append(max(pp))
    return c, rp

def trainmean(x, y):
    meanlist = []
    a = set(y)
    alist = list(a)
    for n in a:
        count = 0
        s = np.array([0] * len(x[0]))
        for m in range(len(y)):
            if y[m] == n:
                count += 1
                s += np.array(x[m])
        meanlist.append(s/count)
    return meanlist, alist

def classmean(meanlist, alist, points, alpha):
    problist = []
    classlist = []
    for point in points:
        dislist = []
        probl = []
        for mean in meanlist:
            dis = np.sqrt(np.dot(mean - point, mean - point))
            dislist.append(dis)
        classlist.append(alist[np.argmin(dislist)])
        dislist = np.array(dislist)
        dislist += np.array([alpha] * len(alist))
        blist = dislist ** 2
        dislist /= blist
        tot = np.sum(dislist)
        for dis in dislist:
            probl.append(dis/tot)
        problist.append(probl)
    return classlist, problist

def minvec(w, x, y):
    wv = np.array(w)
    xm = np.matrix(x)
    xy = np.array(y)
    return np.sum((np.matmul(x, w) - y)**2)

def trainreg(x, y):
    fit = optimize.minimize(minvec, [1.84, 1.25], method='Nelder-Mead', args=(x, y))
    pm = np.transpose((points))
    w = fit.x
    return w

def classreg(w, points, alpha):
    classlist = []
    problist = []
    a = set(y)
    alist = list(a)
    pm = np.transpose((points))
    scorelist = np.dot(w, pm)
    for score in scorelist:
        dislist = []
        probl = []
        for uv in alist:
            dislist.append(abs(score - uv))
        classlist.append(alist[np.argmin(dislist)])
        dislist = np.array(dislist)
        dislist += np.array([alpha] * len(alist))
        blist = dislist ** 2
        dislist /= blist
        tot = np.sum(dislist)
        for dis in dislist:
            probl.append(dis/tot)
        problist.append(probl)
    return classlist, problist

def quartilezer(x):
    npx = np.array(x)
    nx = []
    ml = []
    sl = []
    for n in range(len(npx[0])):
        ncol = []
        col = npx[:, n]
        cm = np.mean(col)
        cs = np.std(col)
        ml.append(cm)
        sl.append(cs)
        for m in col:
            if m > cm + cs:
                ncol.append(3)
            elif m < cm - cs:
                ncol.append(0)
            elif m >= cm:
                ncol.append(2)
            else:
                ncol.append(1)
        nx.append(ncol)
    return np.transpose(nx), ml, sl

def quartilizems(x, ml, sl):
    npx = np.array(x)
    nx = []
    for n in range(len(npx[0])):
        ncol = []
        col = npx[:, n]
        cm = ml[n]
        cs = sl[n]
        for m in col:
            if m > cm + cs:
                ncol.append(3)
            elif m < cm - cs:
                ncol.append(0)
            elif m >= cm:
                ncol.append(2)
            else:
                ncol.append(1)
        nx.append(ncol)
    return np.transpose(nx)

def nbayes(x, y):
    npx = np.array(x)
    probdic = {}
    cor = np.sum(y)
    fal = len(y) - cor
    p1 = cor/len(y)
    p0 = 1-p1
    for n in range(len(npx[0])):
        fdic = {}
        ydic = {}
        ndic = {}
        fx = npx[:, n]
        for num in range(4):
            ydic[num] = 0
            ndic[num] = 0
        for fn in range(len(fx)):
            if y[fn] == 0:
                ndic[fx[fn]] += 1/fal
            else:
                ydic[fx[fn]] += 1/cor
        fdic[1] = ydic
        fdic[0] = ndic
        probdic[n] = fdic
    return probdic

class NBC:
    def __init__(self):
        probdic = {}
        ml = []
        sl = []

    def trainnb(self, x, y):
        nx, ml, sl = quartilezer(x)
        probdic = nbayes(nx, y)
        self.probdic = probdic
        self.ml = ml
        self.sl = sl
        return probdic, ml ,sl

    def classnb(self ,points):
        prd = self.probdic
        ml = self.ml
        sl = self.sl
        problist = []
        classlist = []
        np = quartilizems(points, ml, sl)
        for point in np:
            yprob = 1
            nprob = 1
            for n in range(len(point)):
                yprob *= prd[n][1][point[n]]
                nprob *= prd[n][0][point[n]]
            if nprob > yprob:
                classlist.append(0)
                problist.append(nprob/(nprob + yprob))
            else:
                classlist.append(1)
                problist.append(yprob/(nprob + yprob))
        return classlist, problist

def savemodel(model, filename):
    dump(model, filename)

def loadmodel(filename):
    return load(filename)
