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
        with open('201241.csv') as File:
            reader = csv.DictReader(File)
            for row in reader:
                print(row)
                entradas.append( [int(row['x1']), int(row[' x2']), int(row[' x3']), int(row[' x4']), int(row[' x5'])])
                salida.append( [ int(row[' y']) ] )

                print(row)
            
            #print (results)        
        datosCsv = {"X": entradas, "Y": salida}
        finalW = modeloNeurona.inicializarDatos(datosCsv)

        self.labelWfinal.setText( str(finalW) )
        


if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    app.exec_() #evita cerrar la ventana