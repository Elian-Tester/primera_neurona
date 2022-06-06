from traceback import print_tb
from matplotlib import units
#import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

def inicializarDatos(datosCsv):
    print(" inicializarDatos() ")
    X = np.array(datosCsv["X"], dtype=float)
    Y = np.array(  datosCsv["Y"] , dtype=float)
    W = np.array([ [3, 5, 8] ])
    N = 0.015

    #pesos = np.random.rand(1, len(celsius))

    print("X: ",len(X), " Y: ", len(Y), " P: ", W[0].shape, "\n")

    print(W[0])
    operacionNeurona(X, Y, W, N)



def operacionNeurona(X, Y, W, N):
    cont = 0
    e_margen_error = 0    
    Etotales = []

    epocas = []
    errorAprendizajeEpoca = []

    while cont < 600:

        """ print("\nOperacion: \n")

        print(X)
        print("\n")
        print(W)
        print("\n") """

        bT = np.transpose(W)
        #print(bT)
        #print("Yc / U: -----------------------------------------------\n")
        Yc = np.dot(X,bT)
        #print("\n")
        
        #print(Yc)
        E = Y - Yc

        """ print("E:\n")
        print(E) """
        Etotales.append(E)

        #print("Yd \n")
        #print(Y)

        #print("Et: \n")
        Et = np.transpose(E)
        #print(Et)

        deltaW_1 = np.dot(Et, X)

        #print("delta w: \n")
        #print(deltaW_1)

        #print("delta w: \n")
        deltaW = deltaW_1*N
        #print(deltaW)

        W = W +deltaW
        #print("W: ")
        #print(W)

        eSumaCuadrada = 0
        #print("e margen error: \n")
        """ print("E totales")
        print(Etotales)
        print("E totales[0]")
        print(Etotales[0]) """

        #print("E totales for >>>>>>>>>>>>>>>")
        """ for En in Etotales:    
            print(En, "--------------------------------------------------------<<<<<<<<<<<<<<<<<<<<<<<<<\n")
            eSumaCuadrada += En[0]**2 """
        eSumaCuadrada = round(math.sqrt(sum(x[0]**2 for x in E)),5)

        #e_margen_error = math.sqrt(eSumaCuadrada)
        #print(e_margen_error)
        errorAprendizajeEpoca.append(eSumaCuadrada)

        epocas.append(cont)
        cont +=1

    
    graficar(epocas, errorAprendizajeEpoca)

def graficar(epocaX, eEpocaY):
    print("Graficar: \n")

    print(len(epocaX), len(eEpocaY))
    print(epocaX)
    print(eEpocaY)

    plt.plot(epocaX,eEpocaY, label='error aprendizaje ')
    plt.legend()            
    plt.show()
    


    

def neurona():
    print("\nNeurona:\n")

    