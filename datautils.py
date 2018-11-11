from matplotlib import pyplot as plt

from nnetwrk import neuralnetwork as nn


def create_nn(nOfInputs, config:list, firstlayerw:list = None):
    red = nn.NeuralNetwork(nOfInputs, config)
    if firstlayerw is not None:
        red.setW(firstlayerw)
    return red


def train(red:nn.NeuralNetwork, setEntrada, expectedOutputs,times:int, lr = 0.1):
    error = red.networkTrain(setEntrada,expectedOutputs,times,lr)
    return error


def calcOutputs(red, setEntrada:list):
    return red.networkCalc(setEntrada)

#la función debe dar el valor 1 si el resultado es el del output
#y 0 en caso contrario.
def get_precision(red, setEntrada:list, setSalida:list, funcion):
    real = calcOutputs(red, setEntrada)
    expected = setSalida
    precision = 0
    for i, r in enumerate(real):
        print('test precision data', i)
        precision += funcion(r, expected[i])
    return precision

def graficar1(datax:list, datay:list):
    plt.plot(datay[0], datax[0])
    plt.tight_layout()
    plt.show()

def graficar4(datax:list, datay:list):
    ax = []
    ax.append(plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1))
    ax.append(plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1))
    ax.append(plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1))
    ax.append(plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1))

    for i in range(4):
        ax[i].plot(datay[i],datax[i])
        ax[i].set_xlabel('Época')
        ax[i].set_ylabel('Error')
        ax[i].grid(True)
        ax[i].set_title('red'+str(i+1))
    plt.tight_layout()
    plt.show()

def graficar8(datax1:list, datax2:list, datay1:list, datay2:list):
    ax = []
    ax.append(plt.subplot2grid((2, 4), (0, 0), rowspan=1, colspan=1))
    ax.append(plt.subplot2grid((2, 4), (0, 1), rowspan=1, colspan=1))
    ax.append(plt.subplot2grid((2, 4), (1, 0), rowspan=1, colspan=1))
    ax.append(plt.subplot2grid((2, 4), (1, 1), rowspan=1, colspan=1))
    ax.append(plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=1))
    ax.append(plt.subplot2grid((2, 4), (0, 3), rowspan=1, colspan=1))
    ax.append(plt.subplot2grid((2, 4), (1, 2), rowspan=1, colspan=1))
    ax.append(plt.subplot2grid((2, 4), (1, 3), rowspan=1, colspan=1))

    for i in range(4):
        ax[i].plot(datay1[i],datax1[i])
        ax[i].set_xlabel('Época')
        ax[i].set_ylabel('Error')
        ax[i].grid(True)
        ax[i].set_title('red'+str(i+1))
    for i in range(4):
        ax[i+4].plot(datay2[i],datax2[i])
        ax[i+4].set_xlabel('Época')
        ax[i+4].set_ylabel('Error')
        ax[i+4].grid(True)
        ax[i+4].set_title('red'+str(i+5))
    plt.tight_layout()
    plt.show()

def gen_datag(redes:list, input, output, times:list, lr:list):
    datax = []
    datay = []
    for i, r in enumerate(redes):
        datax.append(train(r,input,output,times[i],lr[i])[0])
        datay.append(range(times[i]))
    return datax,datay
