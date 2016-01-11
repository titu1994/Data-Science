from collections import Counter
import matplotlib.pyplot as plt
import math
import random

# Probability Density Function
def uniformPDF(x):
    return 1 if x >= 0 and x < 1 else 0

# Cumulative Density Function
def uniformCDF(x):
    if x < 0: return 0
    elif x < 1: return x
    else: return 1

# Normal Distribution
def normalPDF(x, mu = 0, sigma = 1.0):
    sqrt2Pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x - mu) ** 2 / 2 / sigma ** 2)) / (sqrt2Pi * sigma)

# Normal CDF
def normalCDF(x, mu = 0, sigma = 1.0):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def __displayPDF():
    xs = [x / 10.0
          for x in range(-50, 50)]
    plt.plot(xs, [normalPDF(x, sigma=1) for x in xs], "-", label="mu = 0, sigma = 1")
    plt.plot(xs, [normalPDF(x, sigma=2) for x in xs], "--", label="mu = 0, sigma = 2")
    plt.plot(xs, [normalPDF(x, sigma=0.5) for x in xs], ":", label="mu = 0, sigma = 0.5")
    plt.plot(xs, [normalPDF(x, mu=-1) for x in xs], "-.", label="mu = -1, sigma = 1")
    plt.legend()
    plt.title("Various PDFs")
    plt.show()

# Display Various PDFs
# __displayPDF()

def __displayCDF():
    xs = [x / 10.0
          for x in range(-50, 50)]
    plt.plot(xs, [normalCDF(x, sigma=1) for x in xs], "-", label="mu = 0, sigma = 1")
    plt.plot(xs, [normalCDF(x, sigma=2) for x in xs], "--", label="mu = 0, sigma = 2")
    plt.plot(xs, [normalCDF(x, sigma=0.5) for x in xs], ":", label="mu = 0, sigma = 0.5")
    plt.plot(xs, [normalCDF(x, mu=-1) for x in xs], "-.", label="mu = -1, sigma = 1")
    plt.legend()
    plt.title("Various CDFs")
    plt.show()

# Display various CDFs
# __displayCDF()

# Inverse Normal CDF
def inverseNormalCDF(p, mu = 0, sigma = 1.0, tolerance = 0.00001):
    if mu != 0 or sigma != 1:
        return mu + sigma * inverseNormalCDF(p, tolerance = tolerance)

    lowZ, lowP = -10.0, 0
    highZ, highP = 10.0, 1
    while highZ - lowZ > tolerance:
        midZ = (lowZ + highZ) / 2
        midP = normalCDF(midZ)
        if midP < p:
            lowZ, lowP = midZ, midP
        elif midP > p:
            highZ, highP = midZ, midP
        else:
            break

    return midZ

# Bernouli Trials
def bernoulliTrial(p):
    return 1 if random.random() < p else 0

# Binomial
def binomial(n, p):
    return sum(bernoulliTrial(p)
               for _ in range(n))

def __displayHistogram(p, n, num_points):
    data = [binomial(n, p)
            for _ in range(num_points)]

    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8, color="0.75")

    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    xs = range(min(data), max(data) + 1)
    ys = [normalCDF(i + 0.5, mu, sigma) - normalCDF(i - 0.5, mu, sigma)
          for i in xs]

    plt.plot(xs, ys)
    plt.title("Binomial Distribution vs Normal Approximation")
    plt.show()

# Display Histogram of Binomial Approximation
# __displayHistogram(0.75, 100, 10000)

"""
Hypothesis
"""

def normalApproximationToBinomial(n, p):
    mu = n * p
    sigma = math.sqrt(n * p * (1 - p))
    return mu, sigma

normalProbabilityBelow = normalCDF

def normalProbabilityAbove(lo, mu = 0, sigma = 1.0):
    return 1 - normalCDF(lo, mu, sigma)

def normalProbabilityBetween(lo, hi, mu = 0, sigma = 1.0):
    return normalCDF(hi, mu, sigma) - normalCDF(lo, mu, sigma)

def normalProbabilityOutside(lo, hi, mu = 0, sigma = 1.0):
    return 1 - normalProbabilityBetween(lo, hi, mu, sigma)

def normalUpperBound(p, mu = 0, sigma = 1.0):
    return inverseNormalCDF(p, mu, sigma)

def normalLowerBound(p, mu = 0, sigma = 1.0):
    return inverseNormalCDF(1 - p, mu, sigma)

def normalTwoSidedBounds(p, mu = 0, sigma = 1.0):
    tailP = (1 - p) / 2
    upperBound = normalLowerBound(tailP, mu, sigma)
    lowerBound = normalUpperBound(tailP, mu, sigma)
    return lowerBound, upperBound

# Check if coin is fair
# mu0, sig0 = normalApproximationToBinomial(1000, 0.5)
# print(normalTwoSidedBounds(0.95, mu0, sig0))

# Power of Test with bias p = 0.55
#lo, hi = normalTwoSidedBounds(0.95, mu0, sig0)
#mu1, sig1 = normalApproximationToBinomial(1000, 0.55)
#typeTwoProbability = normalProbabilityBetween(lo, hi, mu1, sig1)
#power = 1 - typeTwoProbability
#print(power)

def twoSidedPValue(x, mu = 0, sigma = 1.0):
    if x >= mu:
        return 2 * normalProbabilityAbove(x, mu, sigma)
    else:
        return 2 * normalProbabilityBelow(x, mu, sigma)

upperPValue = normalProbabilityAbove
lowerPValue = normalProbabilityBelow

# Bayesian Inference
def B(alpha, beta):
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def betaPDF(x, alpha, beta):
    if x < 0 or x > 1:
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)
