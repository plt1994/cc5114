from nnetwrk import sigmoidneuron as sr, neuron as nr
from nnetwrk.nnutils import stdW, transferDerivative


class NeuronLayer:
    'Neuron layer class, its used to build neural networks\
    based in list of lists of neurons, it has too a NextLayer\
    and a PreviousLayer, everytime you add a new layer, they\
    automatically join to each other setting NextLayers weights\
    and setting self weights depending of the number or neurons\
    into the previous layer.'
    def __init__(self, neurons=None, bias=2):
        #Neuron Layer constructor
        self.neurons = []
        if neurons is not None:
            if type(neurons) is int:
                if neurons > 0:
                    self.neurons = []
                    for i in range(neurons):
                        self.addNeuron(sr.SigmudNeuron(bias))
            else:
                self.neurons = neurons
        self.next = None
        self.prev = None

    def addNeuron(self, neuron: nr.Neuron):
        #add a single neuron to the layer
        self.neurons.append(neuron)

    def addAllNeuron(self, neurons):
        #add a list of neurons to the layer
        for n in neurons:
            self.addNeuron(n)

    def updateW(self, input, lr):
        for n in self.neurons:
            n.updateW(input, lr)
            n.updateB(lr)
        if self.next is not None:
            self.next.updateW(self.getOutputs(), lr)

    def getSize(self):
        return self.neurons.__len__()

    def setWLayer(self, w):
        for neuron in self.neurons:
            neuron.setW(w)

    def setNext(self, nextLayer):
        self.next = nextLayer
        nextLayer.prev = self
        # seteo la cantidad de pesos que recibirá cada neurona del nlayer
        nextLayer.setWLayer(stdW(self.getSize()))

    def getOutputs(self):
        outputs = []
        for i in self.neurons:
            outputs.append(i.getOutput())
        return outputs

    def backProp(self):
        for n in self.neurons:
            error = 0
            for nn in self.next.neurons:
                for w in nn.getW():
                    error += w * nn.getDelta()
            n.setDelta(error * transferDerivative(n.getOutput()))
        if self.prev is not None:
            self.prev.backProp()

    def feed(self, values):
        for n in self.neurons:
            n.calc(values)
        if self.next is not None:
            self.next.feed(self.getOutputs())

    def feedFirst(self, values):
        for n in self.neurons:
            n.calc(values)
        if self.next is not None:
            self.next.feed(self.getOutputs())