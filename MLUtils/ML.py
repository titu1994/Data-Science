import random


def splitData(data, prob):
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def trainSetSplit(x, y, testPerc):
    data = zip(x, y)
    train, test = splitData(data, 1 - testPerc)
    xTrain, yTrain = zip(*train) # Unzip trick
    xTest, yTest = zip(*test)
    return xTrain, xTest, yTrain, yTest

def accuracy(truePositive, falsePositive, falseNegative, trueNegative):
    correct = truePositive + trueNegative
    total = truePositive + falsePositive + falseNegative + trueNegative
    return correct / total

def classificationAccuracy(predictedResults, actualResults):
    pairs = zip(predictedResults, actualResults)
    n = len(predictedResults)
    count = 0
    for predicted, actual in pairs:
        if predicted == actual:
            count += 1

    return (count / n)

def classificationError(predictedResults, actualResults):
    return 1 - classificationAccuracy(predictedResults, actualResults)

def precision(truePositive, falsePositive, falseNegative, trueNegative):
    return truePositive / (truePositive + falsePositive)

def recall(truePositive, falsePositive, falseNegative, trueNegative):
    return truePositive / (truePositive + falseNegative)

def f1Score(truePositive, falsePositive, falseNegative, trueNegative):
    p = precision(truePositive, falsePositive, falseNegative, trueNegative)
    r = recall(truePositive, falsePositive, falseNegative, trueNegative)

    return 2 * p * r / (p + r)
