import math
from LinearUtils.Matrices import shape


def vectorAdd(v, w):
    return [vi + wi
            for vi, wi in zip(v, w)]

def vectorSubtract(v, w):
    return [vi - wi
            for vi, wi in zip(v, w)]

def vectorSum(vectors):
    result = vectors[0]
    for vec in vectors[1:]:
        result = vectorAdd(result, vec)
    return result

def scalarMultiply(c, v):
    return [c * vi
            for vi in v]

def vectorMean(v):
    n = len(v)
    return scalarMultiply(1 / n, vectorSum(v))

def dotProduct(v, w):
    return sum(vi * wi
               for vi, wi in zip(v, w))

def sumOfSquares(v):
    return dotProduct(v, v)

def magnitude(v):
    return math.sqrt(sumOfSquares(v))

def squaredDistance(v, w):
    return sumOfSquares(vectorSubtract(v, w))

def distance(v, w):
    return magnitude(vectorSubtract(v, w))
