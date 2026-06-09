import numpy as np

# ===== Función de activacion =====
def evaluar(FUN, x):
    if (FUN=='tanh'):
        return (2.0 / (1+np.exp(np.dot(-2,x))) - 1)
    elif (FUN=='sigmoid'):
        return (1.0/(1+np.exp(np.dot(-1,x))))
    elif (FUN=='softmax'):
        return (np.exp(x)/(np.sum(np.exp(x),axis=0).reshape(-1,x.shape[1])))
    else:
        return(x)
    
def evaluarDerivada(FUN,x):
    if (FUN=='tanh'):
        return (1-x**2)
    elif (FUN=='sigmoid'):
        #return (x*(1+np.dot(-1,x)))
        return (x*(1-x))
    elif (FUN=='relu'):
        return ((x>0)*1)
    else:
        return(1)