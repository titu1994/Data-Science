from LinearUtils.Vectors import squaredDistance, vectorMean, distance
import random

class KMeans:

    def __init__(self, k):
        self.k = k
        self.means = None

    def classify(self, input):
        """return the index of the fit closest to the input"""
        return min(range(self.k),
                   key=lambda i: squaredDistance(input, self.means[i]))

    def train(self, inputs):
        self.means = random.sample(inputs, self.k)
        assignments = None

        while True:
            newAssignments = list(map(self.classify, inputs))

            if assignments == newAssignments:
                return

            assignments = newAssignments

            for i in range(self.k):
                iPoints = [p for p, a in zip(inputs, assignments) if a == i]
                if iPoints:
                    self.means[i] = vectorMean(iPoints)

def squaredClusteringErrors(inputs, k):
    """finds the total squared error from k-means clustering the inputs"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = list(map(clusterer.classify, inputs))

    return sum(squaredDistance(input,means[cluster])
               for input, cluster in zip(inputs, assignments))

def displayClusters(kmeans : KMeans):
    for cluster in kmeans.means:
        print(cluster)

if __name__ == "__main__":

    buyscomputer = [(0, 2, 0, 1),
                    (0, 2, 0, 0),
                    (1, 2, 0, 1),
                    (2, 1, 0, 1),
                    (2, 0, 1, 1),
                    (2, 0, 1, 0),
                    (1, 0, 1, 0),
                    (0, 1, 0, 1),
                    (0, 0, 1, 1),
                    (2, 1, 1, 1),
                    (0, 1, 1, 0),
                    (1, 1, 0, 0),
                    (1, 1, 1, 1),
                    (2, 1, 0, 0)]

    random.seed(0)
    clusterer = KMeans(k=3)
    clusterer.train(buyscomputer)
    print("3-means:")
    displayClusters(clusterer)
    print("Squared Classification Errot : ", squaredClusteringErrors(buyscomputer, 3))

    random.seed(0)
    clusterer = KMeans(k=5)
    clusterer.train(buyscomputer)
    print("\n5-means:")
    displayClusters(clusterer)
    print("Squared Classification Errot : ", squaredClusteringErrors(buyscomputer, 5))