import random
from collections import Counter
from LinearUtils.Vectors import distance
from StatisticsUtils.Statistics import mean
import matplotlib.pyplot as plt


def rawMajorityVote(labels):
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

def majorityVote(labels):
    voteCounts = Counter(labels)
    winner, winnerCount = voteCounts.most_common(1)[0]
    nWinners = len([count
                    for count in voteCounts.values()
                    if count == winnerCount])

    if nWinners == 1:
        return winner
    else:
        return majorityVote(labels[:-1])

def knnClassifier(k, labeledPoints, newPoint):
    """
    :param k:
    :param labeledPoints: pair of (point, label)
    :param newPoint:
    :return:
    """

    byDistance = sorted(labeledPoints,
                        key=lambda pointlabel: distance(pointlabel[0], newPoint))

    kNearestLabels = [label for _, label in byDistance[:k]]
    return majorityVote(kNearestLabels)




"""
Tests
"""

"""
def randomPoint(dim):
    return [random.random() for _ in range(dim)]

def randomDistance(dim, numPairs):
    return [distance(randomPoint(dim), randomPoint(dim))
                     for _ in range(numPairs)]

dimensions = range(1, 101)
avgDistances = []
minDistances = []

random.seed(0)

for dim in dimensions:
    distances = randomDistance(dim, 10000)
    avgDistances.append(mean(distances))
    minDistances.append(min(distances))


plt.plot(dimensions, avgDistances, color="green", label="Average Distances")
plt.plot(dimensions, minDistances, color="blue", label="Min Distances")
plt.legend(loc="best")
plt.xlabel("# of Dimensions")
plt.title("10,000 Random Distances (AKA Curse of Dimentionality)")
plt.show()


minAvgRatio = [minDist / avgDist
               for minDist, avgDist in zip(minDistances, avgDistances)]

plt.plot(dimensions, minAvgRatio, color="blue")
plt.xlabel("# of Dimensions")
plt.title("Min Distance / Avg Distance")
plt.show()
"""