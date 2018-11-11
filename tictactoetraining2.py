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
#de uno, dos, tres y 4 hiddenlayers.
nOfInput = 9
config1 = [3,1]
red1 = nnetwrk.datautils.create_nn(nOfInput, config1)
config2 = [3,5,1]
red2 = nnetwrk.datautils.create_nn(nOfInput, config2)
config3 = [2,3,5,1]
red3 = nnetwrk.datautils.create_nn(nOfInput, config3)
config4 = [4,7,5,3,1]
red4 = nnetwrk.datautils.create_nn(nOfInput, config4)
config5 = [3,2]
red5 = nnetwrk.datautils.create_nn(nOfInput, config5)
config6 = [3,5,2]
red6 = nnetwrk.datautils.create_nn(nOfInput, config6)
config7 = [2,3,5,2]
red7 = nnetwrk.datautils.create_nn(nOfInput, config7)
config8 = [4,7,5,3,2]
red8 = nnetwrk.datautils.create_nn(nOfInput, config8)

times = 2000
lr = [0.1,0.1,0.1,0.1]
inputdata = []
outputdata = []
outputdata2 = []
for i in range(0,958,10):
    inputdata.append(input[i])
    outputdata.append(output[i])
    outputdata2.append(output2[i])
x1, y1 = nnetwrk.datautils.gen_datag([red1,red2, red3,red4], inputdata, outputdata, [times,times,times,times], lr)
x2, y2 = nnetwrk.datautils.gen_datag([red5,red6, red7,red8], inputdata, outputdata2, [times,times,times,times], lr)
nnetwrk.datautils.graficar8(x1, x2, y1, y2)



