from functools import partial
from collections import defaultdict
from MachineLearning.MapReduce.MapReduce import *

def matrixMultiplyMapper(m, element):
    """m is the common dimension (columns of A, rows of B)
    element is a tuple (matrix_name, i, j, value)"""
    matrix, i, j, value = element

    if matrix == "A":
        for column in range(m):
            # A_ij is the jth entry in the sum for each C_i_column
            yield((i, column), (j, value))
    else:
        for row in range(m):
            # B_ij is the ith entry in the sum for each C_row_j
            yield((row, j), (i, value))

def matrixMultiplyReducer(m, key, indexed_values):
    resultsByIndex = defaultdict(list)

    for index, value in indexed_values:
        resultsByIndex[index].append(value)

    # sum up all the products of the positions with two results
    sumProduct = sum(results[0] * results[1]
                     for results in resultsByIndex.values()
                     if len(results) == 2)

    if sumProduct != 0.0:
        yield (key, sumProduct)

if __name__ == "__main__":
    entries = [("A", 0, 0, 3), ("A", 0, 1,  2),
           ("B", 0, 0, 4), ("B", 0, 1, -1), ("B", 1, 0, 10)]

    mapper = partial(matrixMultiplyMapper, 3)
    reducer = partial(matrixMultiplyReducer, 3)

    print("MapReduce matrix multiplication")
    print("Entries:", entries)
    print("Result:", MapReduce(entries, mapper, reducer))

