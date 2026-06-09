import numpy as np
from matplotlib import pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3  
# from matplotlib import gridspec
import math

#  ======= SOM ==========
        
def linkdist(filas, columnas):
    ocultas = filas*columnas
    pasos = np.zeros((ocultas, ocultas))
    
    for n1 in range(ocultas):
        n1_f = filas - int(n1 / columnas) - 1
        n1_c = n1 % columnas
        for n2 in range(ocultas):
            n2_f = filas - int(n2 / columnas) -1
            n2_c = n2 % columnas
            pasos[n1,n2] = abs(n1_f-n2_f) + abs(n1_c-n2_c)
    return pasos

def ubicacion(nroNeurona, filas, columnas):
    n_f = filas - int(nroNeurona / columnas) - 1
    n_c = nroNeurona % columnas
    return(n_f, n_c)

        
def SOM_plot(P, W, pasos, title_fig):
    ocultas = len(pasos)
    # plotear datos
    plt.clf()  # limpia lo que había antes
    x,y = list(P[:,0]), list(P[:,1])
    plt.scatter(x, y, marker="o")
    plt.title(title_fig)
    
    #E dibujar centros 
    x, y= list(W[:,0]), list(W[:,1])
    plt.scatter(x,y, color='red', s=50)    
    
    # dibujar conexiones
    for n1 in range(ocultas):
        for n2 in range(ocultas):
            if (pasos[n1,n2]==1):
                plt.plot([W[n1, 0], W[n2, 0]], [W[n1, 1], W[n2, 1]], color='r')                   
 
    plt.show()
    plt.pause(0.0001)
    
    
def SOM_scatter(entradas2D, T, nomClases):
    ruido = np.random.uniform(-0.15, 0.15, entradas2D.shape)
    marcador = ['o', 'x', '+', 'v', '^', '<', '>', 's', 'd']
    for especie in nomClases:
        ID = np.argmax(nomClases==especie)  # id. de especie
        cuales = (T==ID)
        
        print(especie, " cant = ", cuales.shape[0])
        
        plt.plot(entradas2D[cuales,0]+ruido[cuales,0], \
                 entradas2D[cuales,1]+ruido[cuales,1], marcador[ID],
                 label=especie)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.show() 
    
        plt.ylabel('SOM_1')
        plt.xlabel('SOM_0')       

    
