import StatisticsUtils.Statistics as stat
import Optimization.StochasticGradientDescent as sgd
import random
import seaborn as sns
sns.set_style("whitegrid")

"""
Simple Linear Regression
"""

def leastSquaresFit(x, y):
    beta = stat.correlation(x, y) * stat.stdDeviation(y) / stat.stdDeviation(x)
    alpha = stat.mean(y) - beta * stat.mean(x)
    return alpha, beta

def predict(alpha, beta, xi):
    return beta * xi + alpha

def printModel(alpha, beta):
    ispositive = beta >= 0
    if ispositive: print("y = %0.3f + (%0.3f * x)" % (alpha, beta))
    else: print("y = %0.3f - (%0.3f * x)" % (alpha, -beta))

"""
Check the "goodness" of the fit
"""

def error(alpha, beta, xi, yi):
    #print("Error : ", yi, " - ", predict(alpha, beta, xi), " = ", (yi - predict(alpha, beta, xi)))
    return yi - predict(alpha, beta, xi)

def sumOfSquaredError(alpha, beta, x, y):
    errors =  [error(alpha, beta, xi, yi)
                for xi, yi in zip(x, y)]
    print("Errors list : ", errors, "\nSum of Errors : ", sum(errors))
    return sum(errors)

def totalSumOfSquares(y):
    return sum([v ** 2 for v in stat.de_mean(y)])

def rSquared(alpha, beta, x, y):
    sumsquared = sumOfSquaredError(alpha, beta, x, y)
    total = totalSumOfSquares(y)
    #print("SumSquared : %f, Total : %f" % (sumsquared, total))
    return 1 - (sumsquared / total)



"""
Linear Regression with advanced methods to reduce error
"""

# Using Gradient Descent to calculate Alpha and Beta

def squaredError(xi, yi, theta):
    alpha, beta = theta
    return error(alpha, beta, xi, yi) ** 2

def squaredErrorGradient(xi, yi, theta):
    alpha, beta = theta
    return [-2 * error(alpha, beta, xi, yi),       # alpha partial derivative
            -2 * error(alpha, beta, xi, yi) * xi]  # beta partial derivative

def gradientDescentFit(X, Y):
    theta = [random.random(), random.random()]
    alpha, beta = sgd.minimizeStochastic(squaredError,
                                         squaredErrorGradient,
                                         X, Y,
                                         theta,
                                         alpha0=0.01)
    return alpha, beta

"""
Test
"""
if __name__ == "__main__":
    print("Randomly generating sample data")

    X = [i * 0.1 for i in range(1, 101)]
    y = [random.gauss(mu=1, sigma=0.5) for _ in X]

    print("Fitting using least squares method")
    alpha, beta = leastSquaresFit(X, y)

    yPred = [predict(alpha, beta, xi) for xi in X]

    printModel(alpha, beta)
    print("Goodness of fit using r^2 value : ", rSquared(alpha, beta, X, y))

    sns.plt.plot(X, y, "bo", X, yPred, "--k")
    sns.plt.title("Plot of sample data with linear regresion slope")
    sns.plt.xlabel("X")
    sns.plt.ylabel("Y")
    sns.plt.show()

    print("\nGenerating sample data using linear line")

    X = [i * 0.1 for i in range(1, 101)]
    y = [x * 5 + (random.random()*10) for x in X]

    print("Fitting using stochastic gradient descent method")
    alpha, beta = gradientDescentFit(X, y)

    yPred = [predict(alpha, beta, xi) for xi in X]

    printModel(alpha, beta)
    print("Goodness of fit using r^2 value : ", rSquared(alpha, beta, X, y))

    sns.plt.plot(X, y, "bo", X, yPred, "--k")
    sns.plt.title("Plot of sample data with linear regresion slope")
    sns.plt.xlabel("X")
    sns.plt.ylabel("Y")
    sns.plt.show()
