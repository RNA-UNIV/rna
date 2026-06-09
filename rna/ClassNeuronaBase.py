import numpy as np
import time

from matplotlib import pylab as plt
from IPython import display

import numpy as np
from abc import ABC, abstractmethod


class NeuronaBase(object):
    """
    Clase base semi-abstracta para elementos de computo neuronales.

    Parameters
    ------------
    alpha : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    draw : int
        1 si dibuja - 0 si no
    title : list con 2 elementos
        titulos de los ejes - sólo 2D
    verbose : int
        1 si muestra progreso - 0 si no

    Attributes
    -----------
    w_ : 1d-array o 2d-array
        Pesos despues de entrenar
    b_ : float o 2d-array
        Bias despues de entrenar
    errors_ : list
        Error en cada epoca.
    accuracy_ : list  # NUEVO: agregado para consistencia
        Accuracy en cada epoch.
    """
    def __init__(self, alpha=0.01, n_iter=50,
                 random_state=None, draw=0, title=['X1', 'X2'], verbose=1):
        self.alpha = alpha
        self.n_iter = n_iter
        self.random_state = random_state
        self.draw = draw
        self.title = title
        self.verbose = verbose

        # Atributos que se inicializarán en fit
        self.w_ = None
        self.b_ = None
        self.errors_ = []

    def _show_progress(self, epoca, y_true, y_pred):
        """
        Muestra barra de progreso con el score actual.

        Parameters
        ----------
        epoca : int
            Número de época actual (base 0)
        y_true : array-like
            Valores reales
        y_pred : array-like
            Valores predichos
        """
        if not self.verbose:
            return

        etiqueta, valor = self._score_metric(y_true, y_pred)

        porcentaje = (epoca + 1) / self.n_iter * 100
        barra_len = 30
        progreso = int(barra_len * (epoca + 1) / self.n_iter)
        barra = '█' * progreso + '░' * (barra_len - progreso)
        print(f'\rÉpoca {epoca + 1}/{self.n_iter} |{barra}| {porcentaje:.1f}% - {etiqueta}: {valor:.6f}',
              end='', flush=True)

    def _weights_init(self, n_inputs, n_outputs=1):
        """
        Inicialización unificada de pesos y sesgos.
        """
        rgen = np.random.RandomState(self.random_state)

        if n_outputs == 1:
            self.w_ = rgen.uniform(-0.5, 0.5, size=n_inputs)
            self.b_ = rgen.uniform(-0.5, 0.5)
        else:
            self.w_ = rgen.uniform(-0.5, 0.5, size=[n_outputs, n_inputs])
            self.b_ = rgen.uniform(-0.5, 0.5, size=[n_outputs, 1])

        # ========== MÉTODOS ABSTRACTOS (deben implementar las subclases) ==========

    @abstractmethod
    def fit(self, X, y):
        """Entrenar el clasificador - debe ser implementado."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predecir etiquetas - debe ser implementado."""
        pass

    @abstractmethod
    def net_input(self, X):
        """Calcular entrada neta - debe ser implementado."""
        pass

    @abstractmethod
    def _score_metric(self, y_true, y_pred):
        """
        Calcula la métrica principal del modelo - implementar en subclase

        Parameters
        ----------
        y_true : array-like
            Valores reales
        y_pred : array-like
            Valores predichos

        Returns
        -------
        tuple : (etiqueta, valor)
        """

        raise NotImplementedError("Cada subclase debe implementar su propia métrica")

    def score(self, X, y):
        """Retorna (etiqueta, valor) de la métrica principal"""
        y_pred = self.predict(X)
        return self._score_metric(y, y_pred)

    def save(self, archivo):
        """Guardar pesos, bias y nombre de clase."""
        np.savez(archivo, clase=self.__class__.__name__, w_=self.w_, b_=self.b_)

    def load(self, archivo):
        """Cargar pesos y bias. Verifica compatibilidad de clase."""
        with np.load(archivo) as data:
            # Verificar cantidad de elementos
            if len(data.keys()) != 3:
                raise ValueError(f"Formato incorrecto. Se esperaban 3 elementos, se encontraron {len(data.keys())}")

            # Verificar clase
            if data['clase'] != self.__class__.__name__:
                print(f"ADVERTENCIA: Archivo de clase '{data['clase']}' cargado en '{self.__class__.__name__}'")

            self.w_ = data['w_']
            self.b_ = data['b_']

            
