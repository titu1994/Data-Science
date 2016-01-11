import StatisticsUtils.Statistics as stat

def predict(alpha, beta, xi):
    return beta * xi + alpha

def error(alpha, beta, xi, yi):
    return yi - predict(alpha, beta, xi)

def sumOfSquaredError(alpha, beta, x, y):
    return sum([error(alpha, beta, xi, yi)
                for xi, yi in zip(x, y)])

def leastSquaresFit(x, y):
    beta = stat.correlation(x, y) * stat.stdDeviation(y) / stat.stdDeviation(x)
    alpha = stat.mean(y) - beta * stat.mean(x)
    return alpha, beta

def totalSumOfSquares(y):
    return sum(v ** 2 for v in stat.de_mean(y))

def rSquared(alpha, beta, x, y):
    return 1 - (sumOfSquaredError(alpha, beta, x, y) / totalSumOfSquares(y))

# Using Gradient Descent to calculate Alpha and Beta

def squaredError(xi, yi, theta):
    alpha, beta = theta
    return error(alpha, beta, xi, yi) ** 2

def squaredErrorGradient(xi, yi, theta):
    alpha, beta = theta
    return [-2 * error(alpha, beta, xi, yi),       # alpha partial derivative
            -2 * error(alpha, beta, xi, yi) * xi]  # beta partial derivative



