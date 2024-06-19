import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

ratingsfile = 'ratings.dat'
columnnames = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratingsdata = pd.read_csv(ratingsfile, delimiter='::', engine='python', names=columnnames)

totalusers = 6040
totalitems = 3952
useritemmatrix = csr_matrix((ratingsdata['Rating'], (ratingsdata['UserID'] - 1, ratingsdata['MovieID'] - 1)), shape=(totalusers, totalitems)).toarray()

useritemmatrix[np.isnan(useritemmatrix)] = 0

kmeanscluster = KMeans(n_clusters=3, random_state=42)
clusterlabels = kmeanscluster.fit_predict(useritemmatrix)

usersbycluster = {cluster: useritemmatrix[clusterlabels == cluster] for cluster in range(3)}

def getrecommendations(matrix, method):
    if method == 'TotalSum':
        return np.sum(matrix, axis=0)
    elif method == 'MeanScore':
        return np.mean(matrix, axis=0)
    elif method == 'CountRatings':
        return np.count_nonzero(matrix, axis=0)
    elif method == 'HighApproval':
        return np.sum(matrix >= 4, axis=0)
    elif method == 'Borda':
        return bordascore(matrix)
    elif method == 'Copeland':
        return copelandscore(matrix)
    else:
        raise ValueError("Unknown method")

def bordascore(matrix):
    scores = np.zeros(totalitems)
    for ratings in matrix:
        ranks = np.argsort(ratings)[::-1]
        for idx, item in enumerate(ranks):
            scores[item] += len(ratings) - idx - 1
    return scores

def copelandscore(matrix):
    copelandscores = np.zeros(totalitems)
    for itemx in range(totalitems):
        for itemy in range(totalitems):
            if itemx != itemy:
                wins = np.sum(matrix[:, itemx] > matrix[:, itemy])
                losses = np.sum(matrix[:, itemx] < matrix[:, itemy])
                copelandscores[itemx] += (wins - losses)
    return copelandscores

aggregationstrategies = ['TotalSum', 'MeanScore', 'CountRatings', 'HighApproval', 'Borda', 'Copeland']

toprecommendations = {}
for cluster, usermatrix in usersbycluster.items():
    toprecommendations[cluster] = {}
    for strategy in aggregationstrategies:
        itemscores = getrecommendations(usermatrix, strategy)
        topitems = np.argsort(itemscores)[::-1][:10]
        toprecommendations[cluster][strategy] = topitems

for cluster, recommendations in toprecommendations.items():
    print(f"Group {cluster} Recommendations:")
    for strategy, items in recommendations.items():
        print(f"  {strategy}: {items}")
