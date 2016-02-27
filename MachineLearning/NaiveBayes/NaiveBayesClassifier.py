from collections import defaultdict, Counter
import math
import numpy as np

class Item:

    def __init__(self, className, label, posProb, negProb, i, fi):
        self.className = className
        self.label = label
        self.posProb = posProb
        self.negProb = negProb
        self.i = i # Feature number Age = 0, Student = 1, Income = 2, Credit = 3
        self.fi = fi # Feature Value number : """  age : y=0, m=1, s=2|  income : l=0, m=1, h=2 | student : n=0, y=1 | credit :  e=0, f=1

    def __str__(self):
        return "Class %s = %s, Probability Positive = %f, Probability Negative = %f" % (self.className, self.label, self.posProb, self.negProb)

    def __eq__(self, other):
        if isinstance(other, Item):
             return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__str__())

class NaiveBayesClassifier:

    def __init__(self, classes, labels, k=0.5 ):
        self.k = k
        self.classes = classes
        self.labels = labels

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        noFeatures = X.shape[1]

        self.computeYProbabilities(y)

        probs = set()

        for i in range(noFeatures):
            uniqueFeatureDict = self._getFeatureUniqueValuePositions(X[:, i])

            for fi, posi in uniqueFeatureDict.items():
                fiPosCount = fiNegCount = 0
                fiNegCount, fiPosCount = self._calculateCounts(fiNegCount, fiPosCount, posi, y)

                posProb = fiPosCount / self.positiveCount
                negProb = fiNegCount / self.negativeCount

                probs.add(Item(self.classes[i], self.labels[i][fi], posProb, negProb, i, fi))

        for rule in probs:
            print(rule)

        self.probabilities = probs

    def predict(self, X):
        X = np.array(X)
        yPred = []
        numFeatures = X.shape[1]

        for xi in X:
            probs = self._getRule(xi, numFeatures)
            probPos = self.posProb
            probNeg = self.negProb

            for p in probs:
                probPos *= p.posProb
                probNeg *= p.negProb

            yPred.append(probPos >= probNeg)

        return yPred

    def _getRule(self, xi, numfeatures) -> list:
        rule = []
        for i in range(numfeatures):
            for p in self.probabilities:
                if p.i == i and p.fi == xi[i]:
                    rule.append(p)

        return rule

    def computeYProbabilities(self, y):
        yCount = Counter(y)
        self.count = yCount[True] + yCount[False]
        self.positiveCount = yCount[True]
        self.negativeCount = yCount[False]
        self.posProb = self.positiveCount / self.count
        self.negProb = self.negativeCount / self.count

    def _getFeatureUniqueValuePositions(self, feature) -> defaultdict:
        setDict = defaultdict(list)

        for i, f in enumerate(feature):
            setDict[f].append(i)

        return setDict

    def _getFeatureUniqueValues(self, feature) -> list:
        ids = set()

        for i, f in enumerate(feature):
            ids.add(i)

        return list(ids)

    def _calculateCounts(self, fiNegCount, fiPosCount, posi, y) -> (int, int):
        for p in posi:
            if y[p]:
                fiPosCount += 1
            else:
                fiNegCount += 1
        return fiNegCount, fiPosCount


if __name__ == "__main__" :
    """
    age : y=0, m=1, s=2
    income : l=0, m=1, h=2
    student : n=0, y=1
    credit :  e=0, f=1
    """
    classes = ["Age", "Income", "Student", "Credit Rating"]

    idlabels = [{0:"Youth", 1: "Middle-Age", 2:"Senior"},
                {0:"Low", 1:"Medium", 2:"High"},
                {0:"Not Student", 1:"Student"},
                {0:"Excellent", 1:"Fair"}]

    buyscomputer = [((0, 2, 0, 1), False),
                    ((0, 2, 0, 0), False),
                    ((1, 2, 0, 1), True),
                    ((2, 1, 0, 1), True),
                    ((2, 0, 1, 1), True),
                    ((2, 0, 1, 0), False),
                    ((1, 0, 1, 0), True),
                    ((0, 1, 0, 1), False),
                    ((0, 0, 1, 1), True),
                    ((2, 1, 1, 1), True),
                    ((0, 1, 1, 0), True),
                    ((1, 1, 0, 0), True),
                    ((1, 1, 1, 1), True),
                    ((2, 1, 0, 0), False)]

    X = []
    y = []
    for i in range(14):
        X.append(buyscomputer[i][0])
        y.append(buyscomputer[i][1])

    clf = NaiveBayesClassifier(classes=classes, labels=idlabels)
    clf.fit(X, y)
    yPred = clf.predict(X)


    tp = tn = fp = fn = 0
    for yi, ypred in zip(y, yPred):
        if yi == 1 and ypred == 1: tp += 1
        elif yi == 0 and ypred == 0: tn += 1
        elif yi == 1 and ypred == 0: fn += 1
        else: fp += 1

    accuracy = tp / ((float) (tp + fp))
    recall = tp / ((float) (tp + fn))

    print("\nFinished Predictions")
    print("True Positive : %d, True Negative : %d, False Positive : %d, False Negative : %d" % (tp, tn, fp, fn))
    print("Accuracy : %f, Recall : %f" % (accuracy, recall))
