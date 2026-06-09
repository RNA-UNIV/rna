import numpy as np
import time
from matplotlib import pylab as plt
from IPython import display

def dibuPtos_y_2Rectas(entradas, salida, W, b, ph=0, titulos=[]):
    if (entradas.shape[1]==2):
        if (len(salida.shape)==2):
            salida = salida.T[0]
        if (len(b.shape)==2):
            b = b.T[0]
            
       #--- DIBUJA LOS EJEMPLOS EN EL FONDO -----
   
        plt.axis([min(entradas[:,0])-0.05, max(entradas[:,0])+0.05,min(entradas[:,1])-0.05, max(entradas[:,1])+0.05])
        plt.setp(plt.gca(), autoscale_on=False)
               
        clases=np.unique(salida)
        if len(clases)==2:
            plt.plot(entradas[salida==min(clases),0], entradas[salida==min(clases),1], 'bo')
            plt.plot(entradas[salida==max(clases),0], entradas[salida==max(clases),1], 'ro')
        else:
            plt.plot(entradas[:,0], entradas[:,1], 'ro')
        if (len(titulos)==2):
            plt.xlabel(titulos[0])
            plt.ylabel(titulos[1])
        
       #--- DIBUJA LAS 2 RECTAS ---
        n = 2
        if (ph!=0):
            for r in range(n):
                for p in ph[r]:  #borramos la recta r
                    p.remove()
        X = np.array([min(entradas[:,0]), max(entradas[:,0])])
     
        ph = []
        for r in range(n):
            Y = (-1)*(W[r,0]/W[r,1])*X - (b[r]/W[r,1])
            ph.append(plt.plot(X,np.squeeze(np.asarray(Y))))
            
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.01)
    return(ph)
