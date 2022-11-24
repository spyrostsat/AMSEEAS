def zakharov(x):
    sum1 = 0
    sum2 = 0
    for l in range(len(x)):
        sum1 += x[l] ** 2
        sum2 += 0.5 * (l + 1) * x[l]
    return sum1 + sum2 ** 2 + sum2 ** 4