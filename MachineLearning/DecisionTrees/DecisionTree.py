import math
from collections import Counter
from collections import defaultdict
from functools import partial


def entropy(classProbabilities):
    return sum(-p * math.log(p, 2)
               for p in classProbabilities
               if p)

def classProbabilities(labels):
    totalCount = len(labels)
    return [count / totalCount
            for count in Counter(labels).values()]

def dataEntropy(labeledData):
    labels = [label for _, label in labeledData]
    probabilities = classProbabilities(labels)
    return entropy(probabilities)

def partitionEntropy(subsets):
    totalCount = sum(len(subset) for subset in subsets)
    return sum(dataEntropy(subset) * len(subset) / totalCount
               for subset in subsets)

def partitionBy(inputs, attribute):
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    return groups

def partitionEntropyBy(inputs, attribute):
    partition = partitionBy(inputs, attribute)
    return partitionEntropy(partition.values())

def classify(tree, ip):
    if tree in [True, False]:
        return tree

    attribute, subtreeDict = tree
    subtreeKey = ip.get(attribute)

    if subtreeKey not in subtreeDict:
        subtreeKey = None

    subtree = subtreeDict[subtreeKey]
    return classify(subtree, ip)

def buildID3(inputs, splitCandidates = None):
    if splitCandidates is None:
        splitCandidates = inputs[0][0].keys()

    numInputs = len(inputs)
    numTrues = len([label for item, label in inputs if label])
    numFalse = numInputs - numTrues

    if numTrues == 0: return False
    if numFalse == 0: return True

    if not splitCandidates:
        return numTrues >= numFalse

    bestAttribute = min(splitCandidates, key=partial(partitionEntropyBy, inputs))

    partitions = partitionBy(inputs, bestAttribute)
    newCandidates = [candidate for candidate in splitCandidates if candidate != bestAttribute]

    subtrees = {attributeValue : buildID3(subset, newCandidates)
                for attributeValue, subset in partitions.items()}

    subtrees[None] = numTrues > numFalse
    return (bestAttribute, subtrees)

"""
Data Set
inputs = [({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'},    False),    ({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'},   False),    ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'},      True),    ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'},   True),    ({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'},       True),    ({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'},     False),    ({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'},         True),    ({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'},  False),    ({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'},       True),    ({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'},  True),    ({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),    ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'},     True),    ({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'},       True),    ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False) ]

Notes of dataset:

This data set is a classification test, in which an interviewer is given a set of information about the interviewee,such as
what "level" they are in current position (junior, mid level and senior programmers),
what language they prefer to program in,
whether they tweet or not in Twitter (deliberate attempt to provide a weak attribute),
and whether they possess a doctorate or not (attempt to signify a strong attribute).

"""

dataset = [({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, False),
           ({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'},   False),
           ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'},      True),
           ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'},   True),
           ({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'},       True),
           ({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'},     False),
           ({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'},         True),
           ({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'},  False),
           ({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'},       True),
           ({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'},  True),
           ({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),
           ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'},     True),
           ({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'},       True),
           ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False)]

buyscomputer = [({"age":"y", "income":"h", "student":"n", "credit":"f"}, False),
                ({"age":"y", "income":"h", "student":"n", "credit":"e"}, False),
                ({"age":"m", "income":"h", "student":"n", "credit":"f"}, True),
                ({"age":"s", "income":"m", "student":"n", "credit":"f"}, True),
                ({"age":"s", "income":"l", "student":"y", "credit":"f"}, True),
                ({"age":"s", "income":"l", "student":"y", "credit":"e"}, False),
                ({"age":"m", "income":"l", "student":"y", "credit":"e"}, True),
                ({"age":"y", "income":"m", "student":"n", "credit":"f"}, False),
                ({"age":"y", "income":"l", "student":"y", "credit":"f"}, True),
                ({"age":"s", "income":"m", "student":"y", "credit":"f"}, True),
                ({"age":"y", "income":"m", "student":"y", "credit":"e"}, True),
                ({"age":"m", "income":"m", "student":"n", "credit":"e"}, True),
                ({"age":"m", "income":"m", "student":"y", "credit":"f"}, True),
                ({"age":"s", "income":"m", "student":"n", "credit":"e"}, False)]

print("Description of the entropy value of each of the attributes")

"""
for key in ["level", "lang", "tweets", "phd"]:
    print(key, partitionEntropyBy(dataset, key))
"""

for key in ["age", "income", "student", "credit"]:
    print(key, partitionEntropyBy(buyscomputer, key))

#print("\nDescription of the entropy values of all senior personal")

#seniorInputs = [(ip, label) for ip, label in dataset if ip["level"] == "Senior"]

#for key in ["lang", "tweets", "phd"]:
#   print(key, partitionEntropyBy(seniorInputs, key))

#print("\nAs we can see, for a person who is a 'Senior', 'twee' is an irrelevent factor, and 'phd' is a much more important factor\n")

decisionTree = buildID3(buyscomputer)

#query = {"level" : "Junior", "lang" : "Java", "tweets" : "yes", "phd" : "no"}
query = {"age":"y", "income":"m", "student":"y", "credit":"f"}
print("Query : ", query)
#print("Interviewer Decision to hire : ", classify(decisionTree, query))
print("Buys Computer : ", classify(decisionTree, query))