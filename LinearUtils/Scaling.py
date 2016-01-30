from StatisticsUtils.Statistics import mean, stdDeviation
from LinearUtils.Matrices import shape, getCol, generateMatrix


def scale(dataMatrix):
    nRows, nCols = shape(dataMatrix)
    means = [mean(getCol(dataMatrix, j))
             for j in range(nCols)]
    stdevs = [stdDeviation(getCol(dataMatrix, j))
              for j in range(nCols)]
    return means, stdevs

def rescale(dataMatrix):
    means, stdevs = scale(dataMatrix)

    def rescaled(i, j):
        if(stdevs[j] > 0):
            return (dataMatrix[i][j] - means[j]) / stdevs[j]
        else:
            return dataMatrix[i][j]

    nRows, nCols = shape(dataMatrix)
    return generateMatrix(nRows, nCols, rescaled)

