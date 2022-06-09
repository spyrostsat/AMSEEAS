from math import sin, pi

def levy(x):
    w1 = 1 + (x[0] - 1) / 4
    wd = 1 + (x[len(x) - 1] - 1) / 4
    s = 0
    for l in range(len(x) - 1):
        w = 1 + (x[l] - 1) / 4
        s += ((w - 1) ** 2) * (1 + 10 * (sin(pi * w + 1) ** 2))
    return (sin(pi * w1)) ** 2 + s + (((wd - 1) ** 2) * (1 + (sin(2 * pi * wd)) ** 2))