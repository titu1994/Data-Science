from collections import defaultdict
from functools import partial

from MachineLearning.MapReduce.WordCounterJob import wordCountMapper, wordCountReducer

"""
A basic version of the MapReduce algorithm consists of the following steps:

1. Use a mapper function to turn each item into zero or more key-value pairs. (Often this is called the map function)
2. Collect together all the pairs with identical keys.
3. Use a reducer function on each collection of grouped values to produce output values for the corresponding key.

"""

def MapReduce(inputs, mapper, reducer):
    """runs MapReduce on the inputs using mapper and reducer"""
    collector = defaultdict(list)

    for input in inputs:
        for key, value in mapper(input):
            collector[key].append(value)

    return [output
            for key, values in collector.items()
            for output in reducer(key,values)]

def reduceWith(aggregation_fn, key, values ):
    """reduces a key-values pair by applying aggregation_fn to the values"""
    yield (key, aggregation_fn(values))

def valuesReducer(aggregation_fn):
    """turns a function (values -> output) into a reducer"""
    return partial(reduceWith, aggregation_fn)

"""
Aggregation Functions
"""

sumReducer = valuesReducer(sum)
maxReducer = valuesReducer(max)
minReducer = valuesReducer(min)
countDistinctReducer = valuesReducer(lambda values: len(set(values)))


if __name__ == "__main__":

    documents = ["data science", "big data", "science fiction"]

    print("Word Count using general MapReduce function")
    print(MapReduce(documents, wordCountMapper, wordCountReducer))
    print()