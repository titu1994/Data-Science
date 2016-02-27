from collections import defaultdict
import re
import math

class NaiveBayesSpamClassifier:

    def __init__(self, k = 0.5):
        self.k = k
        self.wordProbs = []

    def train(self, trainingSet):
        nSpams = len([isSpam
                      for message, isSpam in trainingSet
                      if isSpam])
        nNonSpams = len(trainingSet) - nSpams

        wordCounts = _countWords(trainingSet)
        self.wordProbs = _wordProbabilities(wordCounts,
                                            nSpams,
                                            nNonSpams,
                                            self.k)

    def classify(self, message):
        return _spamProbability(self.wordProbs, message)


def _tokenize(message):
    message = message.lower()
    allWords = re.findall("[a-z0-9']+", message)
    return set(allWords)

def _countWords(trainingSet):
    """
    :param trainingSet: pair of (message, isSpam)
    :return:
    """

    counts = defaultdict(lambda: [0, 0])
    for message, isSpam in trainingSet:
        for word in _tokenize(message):
            counts[word][0 if isSpam else 1] += 1
    return counts

def _wordProbabilities(counts, totalSpams, totalNonSpams, k = 0.5):
    return [(w,
             (spam + k) / (totalSpams + 2*k),
             (nonSpam + k) / (totalNonSpams + 2*k))
            for w, (spam, nonSpam) in counts.items()]

def _spamProbability(wordProbs, message):
    messageWords = _tokenize(message)
    logProbIfSpam = logProbIfNotSpam = 0.0

    for word, probIfSpam, probIfNotSpam in wordProbs:
        if word in messageWords:
            logProbIfSpam += math.log(probIfSpam)
            logProbIfNotSpam += math.log(probIfNotSpam)
        else:
            logProbIfSpam += math.log(1 - probIfSpam)
            logProbIfNotSpam += math.log(1 - probIfNotSpam)

    probIfSpam = math.exp(logProbIfSpam)
    probIfNotSpam = math.exp(logProbIfNotSpam)
    return probIfSpam / (probIfSpam + probIfNotSpam)

