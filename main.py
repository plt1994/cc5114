from pylab import *
import matplotlib.pyplot as plt

import iasubject.neuralnetwork as nn

layerLast = nn.NeuronLayer(1)
layer1 = nn.NeuronLayer(3)
layer2 = nn.NeuronLayer(2)

red = nn.NeuralNetwork(layerLast)
red.addLayer(layer1)
red.addLayer(layer2)
nOfInputs = 2
pesos = [1,-2]
red.setW(pesos)

print(red.calc([0,0]))



setEntrada = [[0, 0], [0, 1], [1, 0], [1, 1]]
expectedOutputs = [0, 1, 1, 1]

#entreno mi red con esos datos
def train(times):
    for i in range(times):
        for j, x in enumerate(setEntrada):
            red.train(x, expectedOutputs[j],15)

    precision = 0
    l = 1
    for k in range(l):
        for i, p in enumerate(setEntrada):
            valor = round(red.calc(p)[0])
            valorEsperado = expectedOutputs[i]
            if (valor == valorEsperado):
                precision += 1
            #print(p, expectedOutputs[i], valor)
    #red.getBias()
    return precision/(4*l)

pr = []
testInterval = range(100)
for i in testInterval:
    pr.append(train(1000))

plt.plot(testInterval, pr)
show()