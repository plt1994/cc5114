from nnetwrk.neuronlayer import NeuronLayer
from nnetwrk.nnutils import transferDerivative, sumAbsSq


class NeuralNetwork:
    def __init__(self, lastLayer: NeuronLayer):
        self.lastLayer = lastLayer
        self.firstLayer = lastLayer

    def addLayer(self, layer):
        layer.setNext(self.firstLayer)
        self.firstLayer = layer

    def setW(self, w):
        self.firstLayer.setWLayer(w)

    def getBias(self):
        for i in self.lastLayer.neurons:
            print(i.getB())

    def backProp(self, expectedOutput):
        for n in self.lastLayer.neurons:
            error = expectedOutput - n.getOutput()
            n.setDelta(error * transferDerivative(n.getOutput()))
        if self.lastLayer.prev is not None:
            self.lastLayer.prev.backProp()

    def train(self, input, expected, lr=0.1):
        self.firstLayer.feedFirst(input)
        self.backProp(expected)
        self.firstLayer.updateW(input, lr)
        return self.lastLayer.getOutputs()

    def networkTrain(self, datasetInput, datasetOutput, nbEpoch):
        errors = []
        for i in range(nbEpoch):
            for i, d in enumerate(datasetInput):
                self.train(d, datasetOutput[i])
            errors.append(sumAbsSq(datasetOutput, self.networkCalc(datasetInput)))
        return errors

    def calc(self, input):
        self.firstLayer.feedFirst(input)
        return self.lastLayer.getOutputs()

    def networkCalc(self, datasetInput):
        outputs = []
        for d in datasetInput:
            outputs.append(self.calc(d)[0])
        return outputs
