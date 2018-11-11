from pylab import *
import matplotlib.pyplot as plt

import nnetwrk.neuralnetwork as nn

nOfInputs = 2
config = [2,2]
red = nn.NeuralNetwork(nOfInputs,config)
pesos = [1,-2]
red.setW(pesos)

print(red.calc([0,0]))

setEntrada = [[0, 0], [0, 1], [1, 0], [1, 1]]
expectedOutputs = [[0, 0], [1,0], [0,1], [0,0]]

#entreno mi red con esos datos
def train(times, lr = 0.1):
    error = red.networkTrain(setEntrada,expectedOutputs,times, lr)
    outputs = red.networkCalc(setEntrada)

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

times = 1000
testInterval = range(times)
y, c = train(times, 0.1)
print(c)
#y = train(times, 0.1)
k = 0
l = 0
for i in c:
    if i[0]<0.5:
        k = 0
    elif i[0]>0.5:
        k = 1
    if i[1]<0.5:
        l = 0
    elif i[1]>0.5:
        l = 1
    print(k+l)

plt.plot(testInterval,y[0])
plt.show()