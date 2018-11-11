import time
import nnetwrk.datautils
import nnetwrk.nnutils as utils

#importar datos y tranformarlos a datos que pueda leer la red
#se usará un dataset de todas las posibles jugadas del gato
#que hagan a X ganador, asumiremos x=1, o=0, b=blanco=0.5
input = []
#tendremos dos tipos de output para ver si aprende mejor con
#1 output o 2
output = []
output2 = []
archivo = open('datasets/tic-tac-toe.data')
for linea in archivo.readlines():
    aux = (linea.replace('x','1').replace('o','0').replace('b','0.5')).split(',')
    output.append([aux[-1].replace('\n','')])
    input.append(aux[0:8])

#transformamos los caracteres a float del input
for i in input:
    for j, value in enumerate(i):
        i[j] = float(value)
#transformamos los valores de output donde positive = 1, negative = 0
for i, o in enumerate(output):
    for j, value in enumerate(o):
        if value == 'p0sitive':
            output[i] = [1]
            output2.append([1, 0])
        else:
            output[i] = [0]
            output2.append([0, 1])


#creamos varias redes con distintas configuraciones
#usaremos 2 configuraciones de output, con 1 y 2 output
#y para las hiddenlayer usaremos 4 configuraciones
#de un layer y dos layers, con 5 neuronas uno y con 3 neuronas otro
nOfInput = 9
config2 = [5,1]
red = nnetwrk.datautils.create_nn(nOfInput, config2)
for i in range(10):
    a = time.time()
    red.networkTrain(input, output, 100*i)
    print(time.time() - a)




