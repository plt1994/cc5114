from pylab import *
import matplotlib.pyplot as plt
import random
from nnetwrk.perceptron import *
import nnetwrk.curvas as cv


pLinea = Perceptron(1)
pLinea.setW([-2,1])

puntos= []
azulesx, azulesy = [], []
rojosx, rojosy = [], []
expect = []
# recta f(x) = a*x+b
a = 1
b = 0
rango = [-10,10]
n = 10
cantidadDatos = 1000

for i in range(cantidadDatos):
    puntos.append([n * random.random() * 2 - n, n * random.random() * 2 - n])

#aqui va la condicion de la recta
for i,p in enumerate(puntos):
    #si y esta abajo de x, me da 0, en caso contrario me da 1
    expect.append(cv.isYup(a,b,p[0],p[1]))

for i, p in enumerate(puntos):
    if expect[i]==0: #rojo
        rojosx.append(p[0])
        rojosy.append(p[1])
    else:
        azulesx.append(p[0])
        azulesy.append(p[1])

plt.plot(azulesx, azulesy, 'bo')
plt.plot(rojosx, rojosy,'ro')
plt.plot([-10,10],cv.lineal(a,b,[-10,10]),'-')
show()


#entreno mi perceptron con esos datos
def train(times):


    # hago un grafico dandole puntos al perceptron
    projox, projoy = [], []
    pazulx, pazuly = [], []
    puntosnuevos = []
    data = 10000

    for i in range(data):
        puntosnuevos.append([n * random.random() * 2 - n, n * random.random() * 2 - n])

    pres = 0
    for p in puntosnuevos:
        valor = pLinea.calc(p)
        valorEsperado = cv.isYup(a,b,p[0],p[1])
        if(valor == valorEsperado):
            pres+=1
        if valor == 0:  # si el valor es Rojo
            projox.append(p[0])
            projoy.append(p[1])
        else:
            pazulx.append(p[0])
            pazuly.append(p[1])

    # si el perceptron considera que el punto está arriba, me retorne 1 (lo asumo como azul)
    # de otra manera me entrega 0 (el color rojo)


    plt.plot(pazulx, pazuly, 'bo')
    plt.plot(projox, projoy, 'ro')
    plt.plot(rango, cv.lineal(a, b, rango), 'k-')
    show()
    print(pLinea.b, pLinea.w)

    #entreno al final para ver como fue la primera iteracion
    for i in range(times):
        for j, rx in enumerate(rojosx):
            pLinea.train(0, [rx, rojosy[j]])
        for j, ax in enumerate(azulesx):
            pLinea.train(1, [ax, azulesy[j]])

    return pres/data

pr = []
testInterval = range(1,10,1)
for i in testInterval:
    pr.append(train(1))

plt.plot(testInterval, pr)
show()