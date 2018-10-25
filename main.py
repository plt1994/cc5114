from pylab import *
import matplotlib.pyplot as plt

import neuralnetwork.neuralnetwork as nn

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
expectedOutputs = [0, 1, 1, 0]

#entreno mi red con esos datos
def train(times, lr = 0.1):
    # for i in range(times):
    #     for j, x in enumerate(setEntrada):
    #         red.train(x, expectedOutputs[j],lr)
    error = red.networkTrain(setEntrada,expectedOutputs,times)

    precision = 0
    l = 1
    for k in range(l):
        for i, p in enumerate(setEntrada):
            valor = red.calc(p)[0]
            if valor >=0.5:
                valor = 1
            else:
                valor = 0
            valorEsperado = expectedOutputs[i]
            if (valor == valorEsperado):
                precision += 1
            #print(p, expectedOutputs[i], valor)
    #red.getBias()
    #return precision/(4*l)
    return error, precision

# # pr = []
# testInterval = range(times)
# # for i in testInterval:
# #     pr.append(train(100))
# #
# # plt.plot(testInterval, pr)
# # show()

times = 5000
testInterval = range(times)
y, c = train(times)
print(c)
plt.plot(testInterval,y)
plt.show()