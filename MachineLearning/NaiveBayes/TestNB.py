import glob, re
import random
from collections import Counter

import MLUtils.ML as ml
import MachineLearning.NaiveBayes.NaiveBayes as nb

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
trainSet, testSet = ml.splitData(data, 0.75)

classifier = nb.NaiveBayesSpamClassifier()
classifier.train(trainSet)

classified = [(subject, isSpam, classifier.classify(subject))
              for subject, isSpam in testSet]

print("Naive Bayes Classified emails :")
for subject, isSpam, spamProbability in classified:
    print("Subject : ", subject, " Spam Probability : ", spamProbability, " IsSpam : ", isSpam )
print()

counts = Counter((isSpam, spamProbability > 0.5)
                 for _, isSpam, spamProbability in classified)

print("In format (isSpam, classified as spam) : \n", counts)
