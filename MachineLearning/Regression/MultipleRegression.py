import random
from functools import partial

from LinearUtils.Vectors import dotProduct, vectorAdd
import Optimization.StochasticGradientDescent as sgd
from MachineLearning.Regression.LinearRegression import totalSumOfSquares



def predict(xi, beta):
    """
    :param xi: Assumes first value is 1
    :param beta:
    :return:
    """
    return dotProduct(xi, beta)

def error(xi, yi, beta):
    return xi -predict(xi, beta)

def squaredError(xi, yi, beta):
    return error(xi, yi, beta) ** 2

def squaredErrorGradient(xi, yi, beta):
    return [-2 * xij * error(xi, yi, beta)
            for xij in xi]

def estimateBeta(x, y):
    betaInitial = [random.random() for xi in x[0]]
    return sgd.minimizeStochastic(squaredError,
                                  squaredErrorGradient,
                                  x, y,
                                  betaInitial,
                                  0.001)

def multipleRSquared(x, y, beta):
    sumOfSquaredErrors = sum(error(xi, yi, beta) ** 2
                             for xi, yi in zip(x, y))
    return sumOfSquaredErrors / totalSumOfSquares(y)

def bootstrapSample(data):
    return [random.choice(data) for _ in data]

def bootstrapStatistics(data, statisticsFn, numSamples):
    return [statisticsFn(bootstrapSample(data)) for _ in range(numSamples)]

def estimateSampleBeta(sample):
    xSample, ySample = zip(*sample)
    return estimateBeta(xSample, ySample)

"""
Regularization
"""

def ridgePenalty(beta, alpha):
    return alpha * dotProduct(beta[1:], beta[1:])

def ridgePenaltyGradiant(beta, alpha):
    return [0] + [2 * alpha * betaj for betaj in beta[1:]]

def squaredErrorRidge(xi, yi, beta, alpha):
    return error(xi, yi, beta) ** 2 + ridgePenalty(beta, alpha)

def squaredErrorRidgeGradient(xi, yi, beta, alpha):
    return vectorAdd(squaredErrorGradient(xi, yi, beta), ridgePenaltyGradiant(beta, alpha))

def estimateBetaRidge(x, y, alpha):
    """
    :param x:
    :param y:
    :param alpha: Hyperparamer to penalize wrong answers
    :return:
    """

    betaInitial = [random.random() for _ in x[0]]
    return sgd.maximizeStochastic(partial(squaredErrorRidge, alpha=alpha),
                                  partial(squaredErrorRidgeGradient, alpha=alpha),
                                  x, y,
                                  betaInitial,
                                  0.001)

