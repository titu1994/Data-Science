import random
from LinearUtils.Vectors import vectorSubtract, scalarMultiply
from Optimization.GradientDescent import negate, negateAll


def inRandomOrder(x, y):
    data = zip(x, y)
    indices = [i for i, _ in enumerate(data)]
    random.shuffle(indices)
    for i in indices:
        yield x[i], y[i]

def minimizeStochastic(targetFn, gradientFn, x, y, theta0, alpha0 = 0.01):
    data = zip(x, y)
    theta = theta0
    alpha = alpha0
    minTheta, minValue = None, float("inf")
    iterationsWithNoMovement = 0

    while iterationsWithNoMovement < 100:
        value = sum([targetFn(xi, yi, theta) for xi, yi in data])

        if value < minValue:
            minTheta, minValue = theta, value
            iterationsWithNoMovement = 0
            alpha = alpha0
        else:
            iterationsWithNoMovement += 1
            alpha *= 0.9

        for xi, yi in inRandomOrder(x, y):
            gradientI = gradientFn(xi, yi, theta)
            theta = vectorSubtract(theta, scalarMultiply(alpha, gradientI))

    return minTheta

def maximizeStochastic(targetFn, gradientFn, x, y, theta0, alpha0 = 0.01):
    return minimizeStochastic(negate(targetFn),
                              negateAll(gradientFn),
                              x, y, theta0, alpha0)


def minimizeStochastic_list(targetFn, gradientFn, x, y, theta0, alpha0 = 0.01):
    data = zip(x, y)
    theta = theta0
    alpha = alpha0
    minTheta, minValue = None, float("inf")
    iterationsWithNoMovement = 0

    while iterationsWithNoMovement < 100:
        # Only change from above, remove the [0]
        value = sum([targetFn(xi, yi, theta)[0] for xi, yi in data])

        if value < minValue:
            minTheta, minValue = theta, value
            iterationsWithNoMovement = 0
            alpha = alpha0
        else:
            iterationsWithNoMovement += 1
            alpha *= 0.9

        for xi, yi in inRandomOrder(x, y):
            gradientI = gradientFn(xi, yi, theta)
            theta = vectorSubtract(theta, scalarMultiply(alpha, gradientI))

    return minTheta

def maximizeStochastic_list(targetFn, gradientFn, x, y, theta0, alpha0 = 0.01):
    # Only change from above, uses minimizeStochastic_list instead of minimizeStochastic
    return minimizeStochastic_list(negate(targetFn),
                              negateAll(gradientFn),
                              x, y, theta0, alpha0)
