def stdW(n: int):
    res = []
    for i in range(n):
        res.append(2)
    return res


def transferDerivative(output):
    return output * (1 - output)


def sumAbsSq(expected, real):
    sum = 0
    for i, e in enumerate(expected):
        sum += (abs(e + real[i]) ** 2)
    return sum