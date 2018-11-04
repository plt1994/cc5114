from pylab import *
import matplotlib.pyplot as plt
from nnetwrk.sigmoidneuron import *


neuronor = SigmudNeuron(2)
neuronor.setW([-2, 2])

setEnt = [[0,0],[0,1],[1,0],[1,1]]
expect = [0,1,1,1]

#entreno mi perceptron con esos datos
def train(times):
    for i in range(times):
        for j, x in enumerate(setEnt):
            neuronor.train(expect[j], x)



    pres = 0
    l = 5
    for k in range(l):
        for i, p in enumerate(setEnt):
            valor = round(neuronor.calc(p))
            valorEsperado = expect[i]
            if (valor == valorEsperado):
                pres += 1
            print(p, expect[i], valor)

    print(neuronor.b, neuronor.w)
    return pres/(4*l)

pr = []
testInterval = range(80)
for i in testInterval:
    pr.append(train(1))

plt.plot(testInterval, pr)
show()
