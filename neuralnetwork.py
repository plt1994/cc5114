from nnetwrk.neuronlayer import NeuronLayer
from nnetwrk.nnutils import *
import nnetwrk.neuron as n


class NeuralNetwork:
    '''
    Class to make a neural network, to intialize its needed
    to give the number of input to give
    to the first layer and the configuration of the network,
    the first element of the right of config is the number
    of neurons in the output, the rest contains number of
    neurons per layer, the first element of the left contains
    the number of neurons of the first layer (the one who receives
    the input).
    '''
    def __init__(self, nOfinput:int, config:list):
        self.noflayer = len(config)
        self.nofhidden = self.noflayer-1
        self.nOfoutput = config[-1]
        self.nofinput = nOfinput
        #layer that contains the final output
        self.lastLayer = NeuronLayer(self.nOfoutput)
        #the layer that will receive the first input
        self.firstLayer = self.lastLayer
        for i in range(self.nofhidden).__reversed__():
            self.addLayer(NeuronLayer(config[i]))
        self.firstLayer.setWLayer(stdW(self.nofinput))

    #get a neuron from a layer, first layer is layer number 0
    def getNeuron(self, layerindex, neuronindex)->n.Neuron:
        if layerindex<=self.nofhidden:
            layer = self.firstLayer
            for i in range(layerindex):
                layer = layer.next
            return layer.getneuron(neuronindex)

    #add a layer to the current network, this layer will
    #replace the first layer. It means that this new layer
    #will receive the first input when the network is feeded.
    def addLayer(self, layer):
        layer.setNext(self.firstLayer)
        self.firstLayer = layer

    #set the weights of the current first layer
    def setW(self, w:list):
        self.firstLayer.setWLayer(w)

    def getBias(self):
        for i in self.lastLayer.neurons:
            print(i.getB())

    #function that backpropagate the feeding product
    def backProp(self, expectedOutput:list):
        c = 0
        for n in self.lastLayer.neurons:
            error = expectedOutput[c] - n.getOutput()
            n.setDelta(error * transferDerivative(n.getOutput()))
            c+=1
        if self.lastLayer.prev is not None:
            self.lastLayer.prev.backProp()

    #feeding, back propagation and update of the network
    def train(self, input:list, expected:list, lr=0.1):
        self.firstLayer.feed(input)
        self.backProp(expected)
        self.firstLayer.updateW(input, lr)
        return self.lastLayer.getOutputs()

    #same as train, but using data sets and number of epoch
    #this will return the error list to check how the network
    #works with the given data set.
    def networkTrain(self, datasetInput:list, datasetOutput:list, nbEpoch:int, lr = 0.1):
        n_errores = datasetOutput[0].__len__()
        errors = []
        for n_e in range(n_errores):
            errors.append([])
        for i in range(nbEpoch):
            for i, d in enumerate(datasetInput):
                self.train(d, datasetOutput[i], lr)
            calcReal = self.networkCalc(datasetInput)
            for n in range(n_errores):
                errors[n].append(sumAbsSq(datasetOutput, calcReal, n))
        return errors

    #calcule outputs using the current state of the network
    def calc(self, input)->list:
        self.firstLayer.feed(input)
        return self.lastLayer.getOutputs()

    #same as calc, but receiving a data set and returning a
    #list of outputs
    def networkCalc(self, datasetInput:list)->list:
        outputs = []
        for d in datasetInput:
            outputs.append(self.calc(d))
        return outputs
