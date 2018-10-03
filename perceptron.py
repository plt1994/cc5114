class Perceptron:
    def __init__(self, b):
        self.w = []
        self.b = b

    def setW(self, w):
        self.w = []
        for wi in w:
            self.w.append(wi)

    def updateB(self, b):
        self.b = b

    def calc(self,x):
        r = self.b
        for i, xi in enumerate(x):
            r+= xi * self.w[i]
        return (r>0)*1

    def train(self, desired, x):
        #entrego x y le doy el resultado que espero obtener con ese x
        real = self.calc(x) #calculo con los valores actuales de mi perceptron
        diff = desired - real # si el real y el deseado son iguales, no hago cambios
        lr = 0.1 #learning rate
        for i, xi in enumerate(x):
            self.w[i] = self.w[i] + (lr * xi* diff)
        self.updateB(self.b + (lr * diff))

class Pnand(Perceptron):
    def __init__(self):
        super().__init__(3)
        self.setW([-2, -2])

class Por(Perceptron):
    def __init__(self):
        super().__init__(0)
        self.setW([1, 1])

class Pand(Perceptron):
    def __init__(self):
        super().__init__(-1)
        self.setW([1, 1])


