import iasubject.neuron as nr
import iasubject.sigmoidneuron as sr
import iasubject.perceptron as pr


def stdW(n):
    res = []
    for i in range(n):
        res.append(2)
    return res

def transferDerivative(output):
    return output*(1-output)

class NeuronLayer:
    def __init__(self, elements = None):
        self.elements = []
        if elements != None:
            if type(elements) is int:
                if elements>0:
                    self.elements = []
                    for i in range(elements):
                        self.addNeuron(sr.SigmudNeuron(2))
                        #self.addNeuron(pr.Perceptron(2))
            else:
                self.elements = elements
        self.next = None
        self.prev = None

    def addNeuron(self, neuron):
        self.elements.append(neuron)

    def addAllNeuron(self, neurons):
        for n in neurons:
            self.addNeuron(n)

    def updateW(self, input,lr):
        for n in self.elements:
            n.updateW(input,lr)
            n.updateB(lr)
        if self.next!=None:
            self.next.updateW(self.getOutputs(),lr)

    def getSize(self):
        return self.elements.__len__()


    def setWLayer(self, w):
        for neuron in self.elements:
            neuron.setW(w)

    def setNext(self, nlayer):
        self.next = nlayer
        nlayer.prev = self
        #seteo la cantidad de pesos que recibirá cada neurona del nlayer
        nlayer.setWLayer(stdW(self.getSize()))

    def getOutputs(self):
        outputs = []
        for i in self.elements:
            outputs.append(i.getOutput())
        return outputs

    def backProp(self):
        for n in self.elements:
            error = 0
            for nn in self.next.elements:
                for w in nn.getW():
                    error+=w*nn.getDelta()
            n.setDelta(error*transferDerivative(n.getOutput()))
        if self.prev!=None:
            self.prev.backProp()

    def feed(self, values):
        for n in self.elements:
            n.calc(values)
        if self.next!=None:
            self.next.feed(self.getOutputs())

    def feedFirst(self, values):
        for n in self.elements:
            n.calc(values)
        if self.next!=None:
            self.next.feed(self.getOutputs())

class NeuralNetwork:
    def __init__(self,lastLayer):
        self.last = lastLayer
        self.firstlayer = lastLayer

    def addLayer(self, layer):
        layer.setNext(self.firstlayer)
        self.firstlayer = layer

    def setW(self, w):
        self.firstlayer.setWLayer(w)

    def getBias(self):
        for i in self.last.elements:
            print(i.getB())

    def backProp(self, expValue):
        for n in self.last.elements:
            error = expValue - n.getOutput()
            n.setDelta(error*transferDerivative(n.getOutput()))
        if self.last.prev!=None:
            self.last.prev.backProp()

    def train(self, input, expected, lr = 0.1):
        self.firstlayer.feedFirst(input)
        self.backProp(expected)
        self.firstlayer.updateW(input, lr)
        return self.last.getOutputs()

    def calc(self,input):
        self.firstlayer.feedFirst(input)
        return self.last.getOutputs()






