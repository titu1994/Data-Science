import random

from LinearUtils.Vectors import squaredDistance, vectorMean


class KMeans:

    def __init__(self, k):
        self.k = k
        self.means = None

    def classify(self, input):
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
                    self.means = vectorMean(iPoints)


def squaredClusteringErrors(inputs, k):
    clusters = KMeans(k)
    clusters.train(inputs)
    means = clusters.means
    assignments = list(map(clusters.classify, inputs))

    return sum(squaredDistance(ip, means[cluster])
               for ip, cluster in zip(inputs, assignments))

