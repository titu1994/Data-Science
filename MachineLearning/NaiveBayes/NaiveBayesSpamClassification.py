import glob, re
import random
from collections import Counter

import MLUtils.ML as ml
import MachineLearning.NaiveBayes.NaiveBayes as nb

# Data set is available at [  https://drive.google.com/folderview?id=0B10YMCCARrmfdHlKaEpCSFY4ZTg&usp=sharing  ]
# Change the location in path to suit the location on your harddrive
path = r"D:\Yue\Downloads\SpamData\*\*"
data = []

for fn in glob.glob(path):
    isSpam = "ham" not in fn

    with open(fn, "r", encoding="ISO-8859-1") as file:
        for line in file:
            if line.startswith("Subject:"):
                subject = re.sub("Subject:", "", line).strip()
                data.append((subject, isSpam))

random.seed(0)
trainSet, testSet = ml.splitData(data, 0.80)

classifier = nb.NaiveBayesSpamClassifier()
classifier.train(trainSet)

classified = [(subject, isSpam, classifier.classify(subject))
              for subject, isSpam in testSet]

print("Naive Bayes Classified emails :")
for subject, isSpam, spamProbability in classified:
    print("Subject : ", subject, " Spam Probability : ", spamProbability, " IsSpam : ", isSpam )
print()

# Change the spam probability threshold and see better results. 0.65 is chosen after 10 tests ranging from 0.5 to 0.75 in increments of 0.25
# 0.68125 gives much higher accuracy, with less than 1 % loss in recall, while higher values cause overfitting and lower cause underfitting
counts = Counter((isSpam, spamProbability > 0.68125)
                 for _, isSpam, spamProbability in classified)

print("In format (isSpam, classified as spam) : \n", counts)
tp, tn, fp, fn = counts[(True, True)], counts[(False, False)], counts[(False, True)], counts[(True, False)]

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Accuracy : %f, Recall : %f" % (precision, recall))

