import math
import random
from collections import Counter
import matplotlib.pyplot as plt
from StatisticsUtils.Probability import inverseNormalCDF
from StatisticsUtils.Statistics import correlation


def __bucketize(point, bucketSize):
    return bucketSize * math.floor(point / bucketSize)

def makeHistogram(points, bucketSize):
    return Counter(__bucketize(point, bucketSize) for point in points)

def plotHistogram(points, bucketSize, title = ""):
    histogram = makeHistogram(points, bucketSize)
    plt.bar(histogram.keys(), histogram.values(), width=bucketSize)
    plt.title(title)
    plt.show()

def randomNormal():
    return inverseNormalCDF(random.random())

# Test 1
"""
random.seed(0)

uniform = [200 * random.random() - 100 for _ in range(10000)]
normal = [57 * inverseNormalCDF(random.random()) for x in range(10000)]

plotHistogram(uniform, 10, "Uniform")
plotHistogram(normal, 10, "Normal")
"""

# Test 2
"""
xs = [randomNormal() for _ in range(10000)]
ys1 = [x + randomNormal() for x in xs]
ys2 = [-x + randomNormal() for x in xs]

plt.scatter(xs, ys1, marker=".", color="black", label="ys1")
plt.scatter(xs, ys2, marker=".", color="gray", label="ys2")
plt.xlabel("xs")
plt.ylabel("ys")
plt.legend(loc=9)
plt.title("Very Different Joint Distributions")
plt.show()

print("Correlation ys1", correlation(xs, ys1))
print("Correlation ys2", correlation(xs, ys2))
"""