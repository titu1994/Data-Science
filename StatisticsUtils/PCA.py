from functools import partial

from LinearUtils.Matrices import shape, generateMatrix
from LinearUtils.Scaling import scale
from LinearUtils.Vectors import magnitude, dotProduct, vectorSum, scalarMultiply, vectorSubtract
from Optimization.GradientDescent import maximizeBatch
from Optimization.StochasticGradientDescent import maximizeStochastic


def deMeanMatrix(A):
    nr, nc = shape(A)
    colMean, _ = scale(A)
    return generateMatrix(nr, nc, lambda i, j: A[i][j] - colMean[j])

def direction(w):
    mag = magnitude(w)
    return [wi / mag for wi in w]

def __directionalVarianceI(xi, w):
    return dotProduct(xi, direction(w)) ** 2

def directionalVariance(X, w):
    return sum(__directionalVarianceI(xi, w)
               for xi in X)

def __directionalVarianceGradiant(xi, w):
    projectionLength = dotProduct(xi, direction(w))
    return [2 * projectionLength * xij for xij in xi]

def directionalVarianceGradiant(X, w):
    return vectorSum(__directionalVarianceGradiant(xi, w)
                     for xi in X)

def firstPrincipleComponent(X):
    guess = [1 for _ in X[0]]
    unscaledMaximizer = maximizeBatch(
        partial(directionalVariance, X),
        partial(directionalVarianceGradiant, X),
        guess
    )
    return direction(unscaledMaximizer)

def firstPrincipleComponentSGD(X):
    guess = [1 for _ in X[0]]
    unscaledMaximizer = maximizeStochastic(
        lambda x, _, w: __directionalVarianceI(x, w),
        lambda x, _, w: __directionalVarianceGradiant(x, w),
        X,
        [None for _ in X],
        guess
    )
    return direction(unscaledMaximizer)

def project(v, w):
    projectionLength = dotProduct(v, w)
    return scalarMultiply(projectionLength, w)

def removeProjectionsFromVector(v, w):
    return vectorSubtract(v, project(v, w))

def removeProjection(X, w):
    return [removeProjectionsFromVector(xi, w) for xi in X]

def principleComponentAnalysis(X, nComponents):
    components = []
    for _ in range(nComponents):
        component = firstPrincipleComponent(X)
        components.append(component)
        X = removeProjection(X, component)

    return components

def transformVector(v, components):
    return [dotProduct(v, w) for w in components]

def transform(X, components):
    return [transformVector(xi, components) for xi in X]