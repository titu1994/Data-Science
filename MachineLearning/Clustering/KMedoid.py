import numpy as np
import random as rd

class MeanNodes:

        def __init__(self, x, y, z, w):
            self.x = x
            self.y = y
            self.z = z
            self.w = w

        def euclideanDistance(self, node):
            return np.sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2 + (self.z - node.z) ** 2 + (self.w - node.w) ** 2)

        def __str__(self):
            return "("+str(self.x)+","+str(self.y)+","+str(self.z)+","+str(self.w)+")"

class KMedroid:

    def __init__(self, k = 2, iter = 25):
        self.k = k
        self.medians = []
        self.clusters = [[] for _ in range(k)]
        self.iter = iter

    def fit(self, nodes):
        self.n = len(nodes)
        meanOld = 0
        meanNew = 0

        for _ in range(self.iter):
            # Random selection of unique median node
            tempNode = nodes[np.random.choice(self.n)]
            while tempNode not in self.medians:
                self.medians.append(tempNode)
                tempNode = nodes[np.random.choice(self.n)]
                if len(self.medians) == self.k:
                    break

            min_dist = []
            for node in nodes:
                # Compute euclidean distance of node from each of the clusters
                distances = []
                for m in range(len(self.medians)):
                    distances.append(node.euclideanDistance(self.medians[m]))

                minIndex = distances.index(min(distances))
                min_dist.append(distances[minIndex])
                self.clusters[minIndex].append(node)
            meanNew = np.mean(min_dist)

            print("Mean Old : ", meanOld, " - Mean New : ", meanNew)

            # Update
            if(meanNew - meanOld < 0):
                break
            else:
                meanOld = meanNew
                self.medians.clear()

                for l in self.clusters:
                    l.clear()

    def displayClusters(self):
        for i, cl in enumerate(self.clusters):
            print("Cluster %d:" % (i+1))
            for c in cl:
                print(c)

if __name__ == "__main__":
    nodes = []

    for i in range(10):
        nodes.append(MeanNodes(rd.randint(0, 10), rd.randint(0, 10), rd.randint(0, 10), rd.randint(0, 10)))

    clusterer = KMedroid()
    clusterer.fit(nodes)
    clusterer.displayClusters()


