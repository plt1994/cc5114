import iasubject.neuron as neuron
import math

class SigmudNeuron(neuron.Neuron):
    def __init__(self, b):
        def sigma(z):
            return 1/(1+math.exp(-z))
        super().__init__(b,sigma)

