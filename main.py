from nnetwrk.datautils import create_nn, graficar4, gen_datag

#entreno mi red con los sets de input y output

#graficamos 4 conjuntos de datos

#genera los datos para ser graficados

#creamos diferentes configuraciones de redes,
#considerando 1 output y 2 input
nnet1 = create_nn(2, [2, 1], [-0.5, 0.5])
nnet2 = create_nn(2, [2, 1], [-0.5, 0.5])
nnet3 = create_nn(2, [2, 1])
nnet4 = create_nn(2, [2, 1])
redes = [nnet1, nnet2, nnet3, nnet4]

#generamos los datos
setEntrada = [[0, 0], [0, 1], [1, 0], [1, 1]]
expectedOutputs = [[0], [1], [1], [0]]

x, y = gen_datag(redes, setEntrada, expectedOutputs, [10000, 10000, 10000, 10000], [0.3, 0.3, 0.3, 0.3])
#mostramos los gráficos
graficar4(x, y)

# times = 2000
# testInterval = range(times)
# errorlist1, outputs1 = train(nnet1, times, 0.3)
# #y = train(times)
# print(outputs1)
# print(errorlist1[0][-1])
# for i in outputs1:
#     if i[0]>0.7:
#         print(1)
#     elif i[0]<0.3:
#         print(0)
# plt.plot(testInterval, errorlist1[0])
# plt.show()

# while(True):
# #     test = input('ingrese dos valores separados por un espacio: ')
# #     n = test.split(' ')
# #     print(n)
# #     for i,v in enumerate(n):
# #         n[i] = int(v)
# #     res = red.calc(n)
# #     if res[0]>0.7:
# #         res = 1
# #     elif res[0]<0.3:
# #         res = 0
# #     print(res)