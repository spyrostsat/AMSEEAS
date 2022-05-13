def sphere(x):
    fx = 0
    for l in range(0, len(x)):
        fx += x[l] ** 2
    return fx