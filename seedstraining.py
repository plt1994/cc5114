import nnetwrk.datautils
import nnetwrk.nnutils as utils
#importar datos del .txt en listas de python
notnormalizeddata = []
output = []
archivo = open('datasets/seeds_dataset.txt','r')
for linea in archivo.readlines():
    dataline = linea.split('\t')
    output.append(dataline[-1].replace('\n',''))
    dataline.pop(-1)
    while(True):
        if '' in dataline:
            dataline.remove('')
        else:
            break
    notnormalizeddata.append(dataline)

# for i in notnormalizeddata:
#     print(i)
# for i in output:
#     print(i)

#dar el formato a los datos
outputfinal = []
for i, n in enumerate(notnormalizeddata):
    for j, nn in enumerate(n):
        notnormalizeddata[i][j] = float(nn)

for i, o in enumerate(output):
    if o == '1':
        outputfinal.append([1,0,0])
    elif o == '2':
        outputfinal.append([0, 1, 0])
    elif o == '3':
        outputfinal.append([0, 0, 1])

# for i in notnormalizeddata:
#     print(i)
# for i in outputfinal:
#     print(i)

print(notnormalizeddata.__len__())
print(outputfinal.__len__())


#normalizar los datos de entrada
normalizeddata = [[],[],[],[],[],[],[]]
for i in range(7):
    auxnorm = []
    for linea in notnormalizeddata:
        auxnorm.append(linea[i])
    # print(auxnorm)
    dh, dl = utils.findD(auxnorm)
    # print(dh,dl)
    auxnorm = utils.normalizeList(auxnorm,dh,dl)
    #
    # print(auxnorm)
    # print('************************************************')
    normalizeddata[i] = auxnorm

# for i in normalizeddata:
#     print(i)

# print('-----------------------------------------------------')
#crear los dataset para entregarlos a la red
normalizeddataset = []
for i in range(210):
    auxdata = []
    for j in normalizeddata:
        auxdata.append(j[i])
    normalizeddataset.append(auxdata)

# for i in normalizeddataset:
#     print(i)
#crear la red

red0 = nnetwrk.datautils.create_nn(7, [5, 3])
red1 = nnetwrk.datautils.create_nn(7, [5, 3])
red2 = nnetwrk.datautils.create_nn(7, [5, 3])
red3 = nnetwrk.datautils.create_nn(7, [5, 3])

#funcion entrenar, que me de el error
times = 2500
#le entrego una parte del dataset, cerca de 30 elementos
inputdata = []
outputdata = []
for i in range(0,210,2):
    inputdata.append(normalizeddataset[i])
    outputdata.append(outputfinal[i])
#x, y = nnetwrk.datautils.gen_datag([red0,red1,red2, red3], inputdata, outputdata, [times,times,times,times], [0.1,0.1,0.1,0.1])
#nnetwrk.datautils.graficar4(x,y)
x,y = nnetwrk.datautils.gen_datag([red0], inputdata, outputdata, [times], [0.1])
def comparar(real:list, expected:list):
    p = 0
    for i, r in enumerate(real):
        print('test',i)
        x = 0
        if r>0.6:
            x = 1
        elif r<0.4:
            x = 0
        if x == expected[i]:
            p = 1
            print('yey!')
        else:
            p = 0
            print('buuuh :c')
            break
    return p

p = nnetwrk.datautils.get_precision(red0, normalizeddataset, outputfinal, comparar)
print('la precision es:', p)
nnetwrk.datautils.graficar1(x,y)

c = 0
for i in red0.networkCalc(normalizeddataset):
    k = 0
    if i[0]> 0.5:
        k = 1
    elif i[1]> 0.5:
        k = 2
    elif i[2] > 0.5:
        k = 3
    print(k, output[c])
    c+=1

#permitir entregar datos para comprobar que funciona

