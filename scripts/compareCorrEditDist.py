import getScores as gs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compareCorrEditDist():
    fscore, method, scores, crossent = gs.getScores()
    trainEditDist = np.array(pd.read_csv('../data/trainEditDist.csv'), dtype=int)
    testEditDist = np.array(pd.read_csv('../data/testEditDist.csv'), dtype=int)
    temp = []
    for i in range(len(scores[0])):
        if scores[0][i] == 0:
            temp.append(scores[1][i])
        else:
            temp.append(1- scores[1][i])
    temp1 = []
    temp2 = []
    for x in range(101):
        temp1.append([])
        temp2.append([])
    for i in range(len(temp)):
        t0 = int(testEditDist[i])
        temp1[t0].append(temp[i])
    for i in range(len(temp1)):
        temp2[i] = np.mean(temp1[i])
        temp1[i] = np.std(temp1[i])
    x = np.arange(101)
    fig1 = plt.figure(1)
    plt.scatter(x, temp1)
    plt.title('std of confidence score per edit distance')
    fig2 = plt.figure(2)
    plt.scatter(x, temp2)
    plt.title('Mean confidence score per edit distance')
    plt.show()

if __name__ == "__main__":
    compareCorrEditDist()
