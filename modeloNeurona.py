from pickletools import optimize
from tabnanny import verbose
from traceback import print_tb
from matplotlib import units
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt


YC_TENSOR = []

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

    datosGraficar = operacionNeurona(X, Y, W, N, eP)

    #comparacion = {"Epoca": epocas, "Y": Y, "YcFinal": YcFinal}    
    #graficarLinAlg = {"Epocas": epocas, "eEpoca": errorAprendizajeEpoca, "eP":eP, "cont": cont, "W": W}

    """ datosGraficar[0]["Epocas"]
    datosGraficar[0]["Y"]
    datosGraficar[0]["YcFinal"] """
    
    """ datosGraficar[1]["Epocas"]
    datosGraficar[1]["eEpoca"]
    datosGraficar[1]["eP"]
    datosGraficar[1]["cont"]
    datosGraficar[1]["W"] """
    
    ycTensor = tensorFlow1(datosCsv["X"], datosCsv["Y"], eP)
    graficar(datosGraficar[1]["Epocas"] , datosGraficar[1]["eEpoca"], datosGraficar[1]["eP"], datosGraficar[1]["cont"], datosGraficar[1]["W"])
    graficarComparacion(datosGraficar[0]["Epoca"], datosGraficar[0]["Y"] ,datosGraficar[0]["YcFinal"], ycTensor)

    return "vacio"



def operacionNeurona(X, Y, W, N, eP):
    cont = 0
    e_margen_error = 0    
    Etotales = []

    epocas = []
    errorAprendizajeEpoca = []
    error = 0
    YcFinal = []

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
        YcFinal = Yc
        
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
    
    comparacion = {"Epoca": epocas, "Y": Y, "YcFinal": YcFinal}
    #graficarComparacion(epocas, Y ,YcFinal)
    graficarLinAlg = {"Epocas": epocas, "eEpoca": errorAprendizajeEpoca, "eP":eP, "cont": cont, "W": W}
    
    return [comparacion, graficarLinAlg]

def tensorFlow1(X, Y, eP):

    epoca = 100
    print("Tensor:\n")

    capa = tf.keras.layers.Dense(units=1, input_shape=[5])
        
    modelo = tf.keras.Sequential([capa])    
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.5),
        loss='mean_squared_error'
    )
    
    histor = modelo.fit(np.array(X, float), np.array(Y, float), epochs=epoca, verbose=False)    
    
    grafiTensor = plt.figure(figsize=(8,4))
    errorP1 = [0, epoca]
    errorP2 = [eP ,eP]

    puntos = modelo.evaluate(X, Y)
    print(f"e: {puntos}")    
    print("W: ", modelo.get_weights())    
    Yc_TENSOR = modelo.predict(X)
    
    plt.plot(errorP1,errorP2, label='e permisible ')
    plt.plot(histor.history["loss"], label='e tensorflow')
    plt.title("Tensor")

    plt.show()

    return Yc_TENSOR


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


def graficarComparacion(epoca, yD, yCl, yCt): # yCt
    print("Graficar comparacion: \n")

    print(yD)
    print("\n______________\n")
    print(yCl)
    print("\n______________\n")
    print(yCt)

    grafiLin = plt.figure(figsize=(8,4))

    plt.plot(epoca,yCt, label='YC tensor')
    plt.plot(epoca,yCl, label='YC linalg ')
    plt.plot(epoca,yD, label='YD ')

    plt.title("Comparacion")
    plt.legend()            
    plt.show()
    
    #return "xd"


    

def neurona():
    print("\nNeurona:\n")

    