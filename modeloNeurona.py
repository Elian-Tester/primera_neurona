from pickletools import optimize
from tabnanny import verbose
from traceback import print_tb
from matplotlib import units
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

def inicializarDatos(datosCsv):
    print(" inicializarDatos() ")
    X = np.array(datosCsv["X"], dtype=float)
    Y = np.array(  datosCsv["Y"] , dtype=float)
    W = np.array([ [5, 8, 7, 6, 7] ])
    N = 0.000015
    eP = 0.5    

    #pesos = np.random.rand(1, len(celsius))

    print("X: ",len(X), " Y: ", len(Y), " P: ", W[0].shape, "\n")

    print(W[0])

    operacionNeurona(X, Y, W, N, eP)
    
    tensorFlow1(datosCsv["X"], datosCsv["Y"], eP)

    return "vacio"



def operacionNeurona(X, Y, W, N, eP):
    cont = 0
    e_margen_error = 0    
    Etotales = []

    epocas = []
    errorAprendizajeEpoca = []
    error = 0

    while cont < 100:

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
        #print(eSumaCuadrada, " <--------------> ")
        error = eSumaCuadrada

        #e_margen_error = math.sqrt(eSumaCuadrada)
        #print(e_margen_error)
        errorAprendizajeEpoca.append(eSumaCuadrada)

        epocas.append(cont)
        cont +=1

    print("\nPesos finales:\n")
    print(W)
    
    return graficar(epocas, errorAprendizajeEpoca, eP, cont, W)

def tensorFlow1(X, Y, eP):
    epoca = 100
    print("Tensor:\n")

    """ print("\nX::")
    print(X)
    print("\nY::")
    print(Y) """
    capa = tf.keras.layers.Dense(units=1, input_shape=[5])
        
    modelo = tf.keras.Sequential([capa])    
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.5),
        loss='mean_squared_error'
    )
    
    histor = modelo.fit(np.array(X, float), np.array(Y, float), epochs=epoca, verbose=False)    
    
    grafiTensor = plt.figure(figsize=(8,4))
    plt.plot(histor.history["loss"], label='error tensorflow ')
    errorP1 = [0, epoca]
    errorP2 = [eP ,eP]
    plt.plot(errorP1,errorP2, label='error permisible ')

    puntos = modelo.evaluate(X, Y)
    print(f"Error: {puntos}")
    
    print("Pesos: ", modelo.get_weights())
    plt.title("Tensor")

    plt.show()





def graficar(epocaX, eEpocaY, eP, epoca, W):
    print("Graficar: \n")

    #print(len(epocaX), len(eEpocaY))
    #print(epocaX)
    #print(eEpocaY)

    errorP1 = [0, epoca]
    errorP2 = [eP ,eP]
    grafiLin = plt.figure(figsize=(8,4))

    plt.plot(epocaX,eEpocaY, label='error linalg ')
    plt.plot(errorP1,errorP2, label='error permisible ')
    plt.title("Lin alg")
    plt.legend()            
    plt.show()
    
    return W


    

def neurona():
    print("\nNeurona:\n")

    