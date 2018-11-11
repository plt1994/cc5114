import random

def stdW(n: int) -> list:
    res = []
    for i in range(n):
        v = random.random()-0.5
        res.append(v)
    return res

def transferDerivative(output):
    return output * (1.0 - output)

def sumAbsSq(expected, real, index):
    sum = 0
    for i, e in enumerate(expected):
        sum += (e[index] - real[i][index]) ** 2
    return sum

def findD(data:list):
    #find dh and dl, to normalize data
    return max(data), min(data)

def normalize(x,dh,dl,nh, nl):
    return (x-dl)*(nh-nl)/(dh-dl)+nl

def normalizeList(inputs:list, dh, dl, nh = 1, nl = 0):
    r = []
    for i in inputs:
        r.append(normalize(i,dh,dl, nh, nl))
    return r

def denormalize(x,dh,dl,nh, nl):
    return ((dl-dh)*x-(nh*dl)+dh*nl)/(nl-nh)