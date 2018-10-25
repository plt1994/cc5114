import neuralnetwork.neuron as neuron

class Perceptron(neuron.Neuron):
    def __init__(self, b):
        def f(x):
            return (x>0)*1
        super().__init__(b,f)




