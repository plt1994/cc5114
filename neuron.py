class Neuron:
    def __init__(self, b, function):
        self.w = []
        self.b = b
        self.f = function
        self.output = 0
        self.delta = 0

    def setW(self, w):
        self.w = []
        for wi in w:
            self.w.append(wi)

    def setB(self, b):
        self.b = b

    def setOutput(self,output):
        self.output = output

    def setDelta(self,d):
        self.delta = d

    def getOutput(self):
        return self.output

    def getW(self):
        return self.w

    def getB(self):
        return self.b

    def getDelta(self):
        return self.delta

    def updateW(self,input,lr):
        for i, xi in enumerate(input):
            self.w[i] = self.w[i] + (lr * self.getDelta() * xi)

    def updateB(self, lr):
        self.setB(self.getB()+(lr*self.getDelta()))

    def calc(self,x):
        'Calcula un output en base a los pesos y de la neurona, y el input x\
        , x debe ser una lista o similar, pues podria causar un problema sino'
        r = self.b
        for i, xi in enumerate(x):
            r+= xi * self.w[i]
        self.setOutput(self.f(r))
        return self.getOutput()

    def train(self, desired, input, lr = 0.1):
        'deprecado xd'
        #entrego x y le doy el resultado que espero obtener con ese x
        real = self.calc(input) #calculo con los valores actuales de mi perceptron
        diff = desired - real # si el real y el deseado son iguales, no hago cambios
        for i, xi in enumerate(input):
            self.w[i] = self.w[i] + (lr * xi* diff)
        self.setB(self.b + (lr * diff))

