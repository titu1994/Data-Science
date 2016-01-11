

def sumOfSquares(v):
    return sum(vi ** 2 for vi in v)

def differenceQuotient(f, x, h):
    return (f(x + h) - f(x)) / h

def partialDifferenceQuotient(f, v, i, h):
    w = [vj + (h if i == j else 0)
         for j, vj in enumerate(v)]

    return (f(w) - f(v)) / h

def estimateGradient(f, v, h = 0.00001):
    return [partialDifferenceQuotient(f, v, i, h)
            for i, _ in enumerate(v)]

def step(v, direction, stepSize):
    return [vi + stepSize * directioni
            for vi, directioni in zip(v, direction)]

def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f


# Gradient Descent
def minimizeBatch(targetFn, gradientFn, theta0, tolerance = 0.000001):
    stepsizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    theta = theta0
    targetFn = safe(targetFn)
    value = targetFn(theta)

    while True:
        gradient = gradientFn(theta)
        nextThetas = [step(theta, gradient, stepsize)
                      for stepsize in stepsizes]
        nextTheta = min(nextThetas, key=targetFn)
        nextValue = targetFn(nextTheta)

        if abs(value - nextValue) < tolerance:
            return theta
        else:
            theta, value = nextTheta, nextValue

def negate(f):
    return lambda *args, **kwargs: [-f(*args, **kwargs)]

def negateAll(f):
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def maximizeBatch(targetFn, gradientFn, theta0, tolerance = 0.000001):
    return minimizeBatch(negate(targetFn), negateAll(gradientFn), theta0, tolerance)