
def shape(A):
    nrows = len(A)
    ncols = len(A[0]) if A else 0
    return (nrows, ncols)

def getRow(A, i):
    return A[i]

def getCol(A, j):
    return [Ai[j]
            for Ai in A]

def generateMatrix(rowcnt, colcnt, generator):
    return [[generator(i, j)
             for j in range(colcnt)]
             for i in range(rowcnt)]

def isDiagnal(i, j):
    return 1 if i == j else 0

def generateIdentity(n):
    return generateMatrix(n, n, isDiagnal)

