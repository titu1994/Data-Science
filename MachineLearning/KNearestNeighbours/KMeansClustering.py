from LinearUtils.Vectors import squaredDistance, vectorMean, distance
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, k):
        self.k = k
        self.means = None

    def classify(self, input):
        """return the index of the cluster closest to the input"""
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

def plotSquaredClusteringErrors():

    ks = range(1, len(inputs) + 1)
    errors = [squaredClusteringErrors(inputs, k) for k in ks]

    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("total squared error")
    plt.show()

#
# using clustering to recolor an image
#

def recolorImage(input_file, k=5):

    img = mpimg.imread(input_file)
    pixels = [pixel for row in img
                    for pixel in row]
    clusterer = KMeans(k)
    clusterer.train(pixels)

    def recolor(pixel):
        cluster = clusterer.classify(pixel) # index of the closest cluster
        return clusterer.means[cluster]     # mean of the closest cluster

    new_img = [[recolor(pixel) for pixel in row]
               for row in img]

    plt.imshow(new_img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Random data to test kmeans
    inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

    random.seed(0)
    clusterer = KMeans(k=3)
    clusterer.train(inputs)
    print("3-means:")
    print(clusterer.means)
    print()

    random.seed(0)
    clusterer = KMeans(k=2)
    clusterer.train(inputs)
    print("2-means:")
    print(clusterer.means)
    print()

    print("Errors as a function of k")

    for k in range(1, len(inputs) + 1):
        print(k, squaredClusteringErrors(inputs, k))
    print()

    #print("Clustering image. Will take a large amount of time.")
    #recolorImage(r"C:\Users\Yue\PycharmProjects\Data Science\MachineLearning\KNearestNeighbours\Balance-Circle.jpg",  k = 5)