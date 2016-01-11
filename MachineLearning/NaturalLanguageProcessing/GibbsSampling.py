import random
from collections import defaultdict


def rollADie():
    return random.choice([1,2,3,4,5,6])

def directSample():
    d1 = rollADie()
    d2 = rollADie()
    return d1, d1 + d2

def randomYGivenX(x):
    """equally likely to be x + 1, x + 2, ... , x + 6"""
    return x + rollADie()

def randomXGivenY(y):
    if y <= 7:
        # if the total is 7 or less, the first die is equally likely to be
        # 1, 2, ..., (total - 1)
        return random.randrange(1, y)
    else:
        # if the total is 7 or more, the first die is equally likely to be
        # (total - 6), (total - 5), ..., 6
        return random.randrange(y - 6, 7)

def gibbsSample(num_iters=100):
    x, y = 1, 2 # can be anything in range [1-6]
    for _ in range(num_iters):
        x = randomXGivenY(y)
        y = randomYGivenX(x)
    return x, y

def compareDistributions(num_samples=1000):
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbsSample()][0] += 1
        counts[directSample()][1] += 1
    return counts
