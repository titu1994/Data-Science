from collections import Counter, defaultdict
import numpy as np
import operator


class Rule:

    def __init__(self, className, label, outcome, prob, i, fi):
        self.className = className
        self.label = label
        self.outcome = outcome == 1
        self.prob = prob
        self.i = i # Feature number Age = 0, Student = 1, Income = 2, Credit = 3
        self.fi = fi # Feature Value number : """  age : y=0, m=1, s=2|  income : l=0, m=1, h=2 | student : n=0, y=1 | credit :  e=0, f=1

    def __str__(self):
        return "IF %s = %s, THEN RESULT = %r : Probability = %f" % (self.className, self.label, self.outcome, self.prob)

    def __eq__(self, other):
        if isinstance(other, Rule):
             return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__str__())


class OneRClassifier:

    def __init__(self, classes, labels):
        self.rules = None
        self.labels = labels
        self.classes = classes

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        numFeatures = X.shape[1]

        rulelist = set()

        # For each of the 4 features
        for i in range(numFeatures):
            # Get which feature is in which position of the data set
            uniqueFeatureDict = self.__getFeatureUniqueValuePositions(X[:, i])

            # For each feature fi and the list of positions in the data set posi
            for fi, posi in uniqueFeatureDict.items():
                fiPosCount = fiNegCount = 0
                # Calculate the positive and negative counts of each feature fi
                fiNegCount, fiPosCount = self.__calculateCounts(fiNegCount, fiPosCount, posi, y)

                # If count of positive is greater than negative count, then choose winner as Positive result and
                # calculate its probability of correctness
                if fiPosCount > fiNegCount:
                    fiWinner = 1 # Positive winner
                    prob = fiPosCount / (fiPosCount + fiNegCount) # Calculate probability
                else:
                    # Else choose the looser, and calculate its probability
                    fiWinner = 0 # Negative Winner
                    prob = fiNegCount / (fiPosCount + fiNegCount) # Calculate probability

                # Add this new rule to the set of rules
                rulelist.add(Rule(self.classes[i], self.labels[i][fi], fiWinner, prob, i, fi))

        # Sort the rules according to how effective they are in predicting the correct answer
        rulelist = sorted(rulelist, key= operator.attrgetter("prob"), reverse=True)

        # Print all of the rules that the ml learns
        for rule in rulelist:
                print(rule)

        self.rules = rulelist

    def __getFeatureUniqueValuePositions(self, feature) -> defaultdict:
        setDict = defaultdict(list) # Dictionary of lists

        for i, f in enumerate(feature): # Index i, Feature f
            setDict[f].append(i) # For each feature f, add all position i in that dictionary if lists

        return setDict

    def __calculateCounts(self, fiNegCount, fiPosCount, posi, y) -> (int, int):
        for p in posi:
            if y[p]: # If sample at position p has positive outcome
                fiPosCount += 1
            else:
                fiNegCount += 1
        return fiNegCount, fiPosCount

    def predict(self, X):
        X = np.array(X)
        yPred = []
        numFeatures = X.shape[1]

        # For each input xi in list X
        for xi in X:
            # Get the corresponding set of rules for this xi
            rule = self.__getRule(xi, numFeatures)
            probPos = probNeg = 1.0

            # For each rule, calculate the probability of predicting true and false
            for r in rule:
                if r.outcome == True:
                    probPos *= r.prob # Proba of positive class
                    probNeg *= 1 - r.prob # Prob of negative outcome = 1 - prob of positive class
                else:
                    probPos *= 1 - r.prob # Prob of positive outcome = 1 - prob of negative class
                    probNeg *= r.prob # Prob of negative class

            # Use probabilities to determing the winner, aka Yes or No yi output and add it to the corresponding xi
            yPred.append(probPos > probNeg)

        return yPred

    def __getRule(self, xi, numfeatures) -> list:
        rule = []
        for i in range(numfeatures): # For each feature i
            for r in self.rules: # For each rule r
                if r.i == i and r.fi == xi[i]: # If feature r.i == i and feature value r.fi == input class value xi[i]
                    rule.append(r) # Add rule

        return rule

if __name__ == "__main__":
    """
    age : y=0, m=1, s=2
    income : l=0, m=1, h=2
    student : n=0, y=1
    credit :  e=0, f=1
    """
    classes = ["Age", "Income", "Student", "Credit Rating"]

    classLabels = [{0: "Youth", 1: "Middle-Age", 2: "Senior"},
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

    clf = OneRClassifier(classes=classes, labels=classLabels)
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
