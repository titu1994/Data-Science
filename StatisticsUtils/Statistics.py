import math
from collections import Counter

from LinearUtils.Matrices import shape, getCol, generateMatrix
from LinearUtils.Vectors import sumOfSquares, dotProduct


def mean(x):
    return sum(x) / len(x)

def median(v):
    n = len(v)
    sortedV = sorted(v)
    midpoint = n // 2
    if n % 2 == 1:
        return sortedV[midpoint]
    else:
        lo = midpoint - 1
        hi = midpoint
        return (sortedV[lo] + (sortedV[hi] - sortedV[lo]) / 2)

def quartile(x, p):
    pIndex = int(p * len(x))
    return sorted(x)[pIndex]

def mode(x):
    counts = Counter(x)
    maxCount = max(counts.values())
    return [xi
            for xi, count in counts
            if count == maxCount]

def dataRange(x):
    return max(x) - min(x)

def de_mean(x):
    xMean = mean(x)
    return [xi - xMean
            for xi in x]

def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sumOfSquares(deviations) / (n - 1)

def stdDeviation(x):
    return math.sqrt(variance(x))

def interquartileRange(x):
    return quartile(x, 0.75) - quartile(x, 0.25)

def covariance(x, y):
    n = len(x)
    return dotProduct(de_mean(x), de_mean(y)) / (n - 1)

def correlation(x, y):
    stdX = stdDeviation(x)
    stdY = stdDeviation(y)
    if stdX > 0 and stdY > 0:
        return covariance(x, y) / stdX / stdY
    else:
        return 0

def correlationMatrix(data):
    _, numCols = shape(data)

    def matrixEntry(i, j):
        return correlation(getCol(data, i), getCol(data, j))

    return generateMatrix(numCols, numCols, matrixEntry)