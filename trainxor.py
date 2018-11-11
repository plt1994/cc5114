from pylab import *
import matplotlib.pyplot as plt

import nnetwrk.neuralnetwork as nn
import nnetwrk.perceptron as p

nOfInputs = 2
config = [1,2]
redxor = nn.NeuralNetwork(nOfInputs, config)
pand = p.Perceptron(-1)
pand.setW([1,1])
pesos = [1,-2]
redxor.setW(pesos)

print(redxor.calc([0, 0]))

setEntradaxor = [[0, 0], [0, 1], [1, 0], [1, 1]]
expectedOutputsxor = [[0, 1], [1,1], [1,1], [1,0]]

#entreno mi red con esos datos
def train(times, lr = 0.1):
    error = redxor.networkTrain(setEntradaxor, expectedOutputsxor, times, lr)
    outputs = redxor.networkCalc(setEntradaxor)

    #red.getBias()
    #return precision/(4*l)
    return error, outputs

# # pr = []
# testInterval = range(times)
# # for i in testInterval:
# #     pr.append(train(100))
# #
# # plt.plot(testInterval, pr)
# # show()

times = 2000
testInterval = range(times)
y, c = train(times, 0.3)
print(c)
#y = train(times, 0.1)
for i in c:
    k = []
    for j, el in enumerate(i):
        if i[j] > 0.7:
            k.append(1)
        elif i[j] < 0.3:
            k.append(0)

    print(pand.calc(k))

plt.plot(testInterval,y[0])
plt.show()