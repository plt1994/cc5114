from pylab import *
import matplotlib.pyplot as plt
import random
from perceptron import *


pLinea = Perceptron(1)
pLinea.setW([1,1])

puntos= []
azulesx, azulesy = [], []
rojosx, rojosy = [], []
expect = []
# recta f(x) = x

for i in range(100):
    puntos.append([random.random(), random.random()])

#aqui va la condicion de la recta
for i,p in enumerate(puntos):
    #si x>y, estoy bajo la curva y = x
    expect.append((p[0]>p[1])*1) #si esta abajo de x, me da 1, en caso contrario me da cero

for i, p in enumerate(puntos):
    if expect[i]==0: #rojo
        rojosx.append(p[0])
        rojosy.append(p[1])
    else:
        azulesx.append(p[0])
        azulesy.append(p[1])

# plt.plot(azulesx, azulesy, 'bo')
# plt.plot(rojosx, rojosy,'ro')
# plt.plot(linspace(0,1,2),'-')
# show()
#for i, p in enumerate(puntos):
#    print(f'({p[0]},{p[1]}) = {expect[i]}')

#hacer una linea y un test de validacion


#generar un par aleatorio de numeros
#veo si está sobre la curva o bajo la curva
#    ese es mi resultado esperado para ese punto( dos respuestas posibles, 1 o 0)

#entreno mi perceptron con esos datos
for i in range(10000):
    for j, rx in enumerate(rojosx):
        pLinea.train(0,[rx, rojosy[j]])
    for j, ax in enumerate(azulesx):
        pLinea.train(1, [ax, azulesy[j]])


#hago un grafico dandole puntos al perceptron
projox, projoy = [], []
pazulx, pazuly = [], []
puntosnuevos = []
n = 10
for i in range(1000):
    puntosnuevos.append([n*random.random()*2-n, n*random.random()*2-n])

for p in puntosnuevos:
    valor = pLinea.calc(p)
    if valor == 0: #si el valor es Rojo
        projox.append(p[0])
        projoy.append(p[1])
    else:
        pazulx.append(p[0])
        pazuly.append(p[1])


#si el perceptron considera que el punto está arriba, me retorne 1 (lo asumo como azul)
#de otra manera me entrega 0 (el color rojo)

plt.plot(pazulx, pazuly, 'bo')
plt.plot(projox, projoy,'ro')
plt.plot(linspace(-10,10,2),linspace(-10,10,2),'k-')
show()
print(pLinea.b, pLinea.w)