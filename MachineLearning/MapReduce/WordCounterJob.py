from collections import defaultdict, Counter
from MachineLearning.NaiveBayes.NaiveBayes import _tokenize as tokenize

"""
A basic version of the MapReduce algorithm consists of the following steps:

1. Use a mapper function to turn each item into zero or more key-value pairs. (Often this is called the map function)
2. Collect together all the pairs with identical keys.
3. Use a reducer function on each collection of grouped values to produce output values for the corresponding key.

"""

def wordCount_old(documents):
    """word count not using MapReduce"""
    return Counter(word
        for document in documents
        for word in tokenize(document))

def wordCountMapper(document):
    """for each word in the document, emit (word,1)"""
    for word in tokenize(document):
        yield (word, 1)

def wordCountReducer(word, counts):
    """sum up the counts for a word"""
    yield (word, sum(counts))

def wordCount(documents):
    """count the words in the input documents using MapReduce"""

    # place to store grouped values
    collector = defaultdict(list)

    for document in documents:
        for word, count in wordCountMapper(document):
            collector[word].append(count)

    return [output
            for word, counts in collector.items()
            for output in wordCountReducer(word, counts)]

if __name__ == "__main__":

    documents = ["data science", "big data", "science fiction"]

    wc_mapper_results = [result
                         for document in documents
                         for result in wordCountMapper(document)]

    print("wordCountMapper results")
    print(wc_mapper_results)
    print()

    print("wordCount results")
    print(wordCount(documents))
    print()

