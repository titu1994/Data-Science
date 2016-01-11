from collections import Counter

import MachineLearning.DecisionTrees.DecisionTree as decisiontree

def decisionForestClassify(trees, input):
    votes = [decisiontree.classify(tree, input) for tree in trees]
    voteCounts = Counter(votes)

    return voteCounts.most_common(1)[0][0]

