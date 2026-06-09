import numpy as np
#from matplotlib import pyplot as plt
from rna.grafica_SOM import *

def calcSilohuette(entradas, centros):
    # ENTRADAS es una matriz de CantEjemplos x nAtrib
    # CENTROS es una matriz de k x nAtrib
    
    CantEjemplos = entradas.shape[0]  #cantidad de filas de ENTRADAS
    nAtrib = entradas.shape[1]
    
    k = centros.shape[0]
    
    #asignar los ejemplos a los centros
    asignaciones = np.zeros(CantEjemplos)
    for e in range(CantEjemplos):
        m = (centros - np.outer(np.ones(k),entradas[e,:]))**2
        dists = np.sqrt(np.sum(m,axis=1))
        asignaciones[e] = np.argmin(dists)

    # se supone que ambas matrices tienen la misma cantidad de columnas
    silhouette =np.zeros(CantEjemplos)
    for e in range(CantEjemplos):
        c = int(asignaciones[e])
            
        miGrupo = entradas[asignaciones==c, :]
        N = miGrupo.shape[0]
        dist_miGrupo = np.mean(np.sqrt(np.sum((miGrupo - np.outer(np.ones(N), entradas[e,:]))**2, axis=1)))
        
        dist = np.argsort(np.sum((centros - np.outer(np.ones(k),centros[c,:]))**2, axis=1))
        masCercano = dist[1]
        
        otroGrupo = entradas[asignaciones==masCercano, :]
        M = otroGrupo.shape[0]
        dist_otroGrupo = np.mean(np.sqrt(np.sum((otroGrupo - np.outer(np.ones(M),entradas[e,:]))**2, axis=1)))
        
        silhouette[e] = (dist_otroGrupo - dist_miGrupo)/max(dist_otroGrupo,dist_miGrupo)
    silhouetteAVG = np.mean(silhouette)

    return(silhouetteAVG)    


def CPN_entrena(X, k, alfa, MAX_ITE, usaF1=1, \
                T=[], beta=0, usaF2=1, MAX_ITE2=300, \
                dibuja=0,titulos=['X1','X2']):
    # k es la cantidad de grupos a formar
    
    (CantEjemplos,nAtrib) = X.shape
    
    # Tomamos al azar k ejemplos como centros iniciales
    mezcla = np.random.permutation(CantEjemplos)
    centros = X[mezcla[0:k],:]
   
    centros_ant = np.zeros(centros.shape)

    # inicialmente todos pertenecen al mismo grupo
    asignaciones = np.ones(CantEjemplos)
    if dibuja:
        ph = dibuPtosColor(1, X, asignaciones, titulos, centros)  

    ite = 0
    factor = 1
    cambioAVG = np.mean((centros-centros_ant)**2)
    while ((cambioAVG>0) and (ite<MAX_ITE)):
        if usaF1:  #-- se reduce la modificación de los pesos ---
            factor = (MAX_ITE-ite) / MAX_ITE
         
        centros_ant = centros.copy()
        #distribuir los ejemplos en los centros
        for e in range(CantEjemplos):
            #-- buscando el centro más cercano --
            dists = np.sqrt(np.sum((centros - X[e,:])**2,axis=1))
            cMin = np.argmin(dists)
            #-- acercamos el centroide más cercano --
            centros[cMin, :] = centros[cMin, :] + factor * alfa * (X[e,:]-centros[cMin, :])
            
            # sóo para pintar
            asignaciones[e] = cMin
#            if dibuja and (e % 30 ==0):
#                ph = dibuPtosColor(1, X, asignaciones, titulos, centros, ph)
            
        ite = ite + 1
        
        cambioAVG = np.mean((centros-centros_ant)**2)
        if dibuja:
            ph = dibuPtosColor(1, X, asignaciones, titulos, centros, ph)            
            print(ite, cambioAVG)
                
    #--- asignacion final de los ejemplos en los centros ---
    asignaciones=[]
    for e in range(CantEjemplos):
        dists = np.sqrt(np.sum((centros - X[e,:])**2,axis=1))
        asignaciones.append(int(np.argmin(dists)))
    
    #=== la capa competitiva ya está entrenada y los ejemplos fueron asignados ===
    if (T!=[]) and (beta>0):  #se indicó la clase    
        ocultas = k
        salidas = T.shape[1]

        W = np.random.uniform(-0.5, 0.5, (ocultas,salidas))
        W_ant = np.zeros((ocultas,salidas))
        
        ErrorAVG = np.mean((W - W_ant)**2)
        ite2=0
        factor=1
        while (ite2<MAX_ITE2):
            # para cada ejemplo calcular la neurona ganadora
            if usaF2:
                factor = (MAX_ITE2-ite2)/MAX_ITE2
            W_ant = W.copy()
            for e in range(CantEjemplos):
                c = asignaciones[e]   
                W[c,:] = W[c,:] + factor * beta * (T[e,:] - W[c,:])
            
            ErrorAVG = np.mean((W - W_ant)**2)    
            ite2 = ite2 + 1
        
        #np.set_printoptions(precision=2, suppress=True)
        #print (W)            
        return(centros,asignaciones, ite, W, ite2) 
    else:
        return(centros,asignaciones, ite) 


def SOM_entrena(P, filas, columnas, alfa, vecindad, ite_reduce, dibuja):
    ocultas = filas * columnas
    
    # Entrenar SOM
    if dibuja:
        plt.figure(figsize=(6,3))
        
    (CantEjemplos,entran) = P.shape   
    
    w_O = np.random.rand(ocultas, entran) 
#    w_O = -10 * np.ones([ocultas,entran])
    pasos = linkdist(filas, columnas)
    
    max_ite = ite_reduce * (vecindad + 2)
    
    print('Iteraciones: ', max_ite)
    ite = 0
    # ver red
    if dibuja:
        SOM_plot(P, w_O, pasos, title_fig= 'Iteración: ' + str(ite)\
             + '-- Vecindad: ' +str(vecindad) )
    
    while (ite < max_ite):
        for p in range(CantEjemplos): 
            distancias = -np.sqrt(np.sum((w_O-P[p,:])**2, axis=1))
            ganadora = np.argmax(distancias)
    
            for n in range(ocultas):
                if (pasos[ganadora, n] <= vecindad):
                       w_O[n,:] = w_O[n,:] + alfa * (P[p, :] - w_O[n,:]) 
                       
    #        if (dibujar and (vecindad==1) and (p<250) and (p % 10 == 0) and ((ite % ite_reduce)==0)):
    #            SOM_plot(P, w_O, pasos, title_fig= 'Iteración: ' + str(ite) \
    #                     + '-- Vecindad: ' +str(vecindad) + '-- Patron: ' +str(p))
            
        ite = ite + 1
        
        if (vecindad >= 1) and ((ite % ite_reduce)==0):
            vecindad = vecindad - 1
    
        if dibuja:
            SOM_plot(P, w_O, pasos, title_fig= 'Iteración: ' + str(ite) \
                     + '-- Vecindad: ' +str(vecindad) )
                
    entradas2D = np.zeros([CantEjemplos, 2])
    for e in range(CantEjemplos):
        nroNeurona = np.argmin(np.sum((w_O-P[e,:])**2,axis=1))
        (fil,col) = ubicacion(nroNeurona,filas,columnas)
        entradas2D[e, 0] = col # columna dentro del mapa
        entradas2D[e, 1] = filas-fil # fila dentro del mapa   
          
    return(w_O, entradas2D)        
