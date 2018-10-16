def lineal(a,b,intervalo):
    res = []
    for i in intervalo:
        res.append(a*i+b)
    return res

def isYup(a,b,x,y):
    f = a*x + b
    if y>f:
        return 1
    return 0