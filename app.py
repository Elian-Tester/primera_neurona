from PyQt5 import uic, QtWidgets
import csv
import matplotlib.pyplot as plt
import sys

import modeloNeurona 

qtCreatorFile = "vistaNeurona.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
           
        self.iniciar_button.clicked.connect(self.leerCsv)


    def leerCsv(self):
        print("Leer csv\n")        
        results = []
        entradas = []
        salida = []
        with open('entrada.csv') as File:
            reader = csv.DictReader(File)
            for row in reader:
                entradas.append( [1, row['x1'], row['x2']])
                salida.append( [ row['y'] ] )

                print(row)
            
            #print (results)        
        datosCsv = {"X": entradas, "Y": salida}
        modeloNeurona.inicializarDatos(datosCsv)
        


if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    app.exec_() #evita cerrar la ventana