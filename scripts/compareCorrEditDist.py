import getScores as gs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# compareCorrEditDist() gets the scores for a given model and finds the correlation
# between edit distance and confidence scores.
def compareCorrEditDist():
    fscore, method, scores, crossent = gs.getScores()
    try:
        trainEditDist = np.array(pd.read_csv('../data/trainEditDist.csv'), dtype=int)
        testEditDist = np.array(pd.read_csv('../data/testEditDist.csv'), dtype=int)
    except:
        print("No valid datasets were found")
    # create bins
    temp = []
    temp1 = []
    temp2 = []
    # fill the bins
    for i in range(len(scores[0])):
        if scores[0][i] == 0:
            temp.append(scores[1][i])
        else:
            temp.append(1- scores[1][i])
    for x in range(101):
        temp1.append([])
        temp2.append([])
    for i in range(len(temp)):
        t0 = int(testEditDist[i])
        temp1[t0].append(temp[i])
    for i in range(len(temp1)):
        temp2[i] = np.mean(temp1[i])
        temp1[i] = np.std(temp1[i])
    # create x-axis
    x = np.arange(101)
    
    fig1 = plt.figure(1)
    plt.scatter(x, temp1)
    plt.title('std of confidence score per edit distance')
    plt.ylabel('Confidence Score')
    plt.xlabel('edit_distance')

    fig2 = plt.figure(2)
    plt.scatter(x, temp2)
    plt.ylabel('Confidence Score')
    plt.xlabel('edit_distance')
    plt.title('Mean confidence score per edit distance')

    plt.show()

if __name__ == "__main__":
    compareCorrEditDist()
