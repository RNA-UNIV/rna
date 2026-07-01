from rna.grafica import *
from rna.ClassNeuronaBase import NeuronaBase

class NeuronaGradiente(NeuronaBase):
    """
    Parameters
    ------------
    alpha : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    cotaE : float
        minimum error threshold
    FUN : string
        activation function: 'sigmoid', 'tanh', otherwise linear
    random_state : int
        Random number generator seed for random weight initialization.
    draw : int
        1 si dibuja -  0 si no
    title : list con 2 elementos
        titulos de los ejes - sólo 2D
    verbose : int
        1 si muestra progreso - 0 si no

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, alpha=0.01, epochs=50, cotaE=10e-07, FUN='sigmoid', COSTO='ECM', random_state=None, draw=0, title=['X1','X2'], verbose=1):
        self.alpha = alpha
        self.epochs = epochs
        self.cotaE = cotaE
        self.FUN = FUN
        self.COSTO = COSTO
        self.random_state = random_state #-- asignar el valor 1 para fijar la semilla por defecto es aleatorio
        self.draw = draw
        self.title = title
        self.verbose = verbose
        self.accuracy_ = []

    def _check_cost_and_activation(self):
        if self.FUN not in ('sigmoid', 'tanh', 'linear'):
            self.FUN = 'linear'
        # Si es ECM, cualquier activación vale (incluyendo linear)
        if self.COSTO not in ('ECM', 'EC_binaria'):
            self.COSTO = 'ECM'

        if self.COSTO == 'ECM':
            return  # ECM funciona con cualquier activación

        # Si es EC_binaria, solo sigmoid (para valores en 0-1)
        if self.COSTO == 'EC_binaria' and self.FUN == 'sigmoid':
            return

        raise ValueError(f"Combinación no soportada: {self.COSTO} + {self.FUN}")

    def fit(self, X, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.
        Returns
        -------
        self : object
        """

        self._check_cost_and_activation()

        rgen = np.random.RandomState(self.random_state)

        self.w_ = rgen.uniform(-0.5, 0.5, size= X.shape[1])
        self.b_ = rgen.uniform(-0.5, 0.5)
        self.errors_ = []
        self.accuracy_ = []

        ph = 0  # manejador de la recta mientras se dibuja
        ErrorAnt = 0
        ErrorAct = 1

        i = 0
        try:
            while ((i<self.epochs) and (np.absolute(ErrorAnt- ErrorAct) > self.cotaE)):
                ErrorAnt = ErrorAct
                ErrorAct = 0

                for xi, target in zip(X, y):
                    salida = self.predict(xi)
                    errorXi = (target - salida)

                    update = self.alpha * self.deriv_fCosto(target, salida)

                    self.w_ += update * xi
                    self.b_ += update

                    ErrorAct += self.fCosto(target, salida)

                self.errors_.append(ErrorAct)
                self.accuracy_.append(self.accuracy(X, y))

                # graficar la recta
                if (self.draw):
                    ph = dibuPtosRecta(X,y, self.w_, self.b_, self.title, ph)

                # progreso de entrenamiento
                if self.verbose:
                    y_pred = self.predict(X)
                    self._show_progress(i, y, y_pred)

                i = i + 1
        finally:
            if (self.draw):
                waitDibu(ph)

        if self.verbose:
            print()  # Salto de línea al finalizar progreso

        return self

    def fCosto(self, y, y_hat):
        # -- y es el valor esperado e y_hat el valor obtenido (ambos escalares)
        EPS = EPS = np.finfo(float).eps
        if (self.COSTO == 'ECM'):
            return ((y - y_hat) ** 2)
        if (self.COSTO == 'EC_binaria'):
            return (-y * np.log(y_hat + EPS) - (1 - y) * np.log(1 - y_hat + EPS))
        else:
            return (np.absolute(y - y_hat))

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def evaluar(self, x):
        if (self.FUN == 'tanh'):
            return (2.0 / (1 + np.exp(-2 * x)) - 1)
        elif (self.FUN == 'sigmoid'):
            return (1.0 / (1 + np.exp(-x)))
        else:
            return (x)

    def derivar(self, x):
        if (self.FUN == 'tanh'):
            return (1 - x ** 2)
        elif (self.FUN == 'sigmoid'):
            return (x * (1 - x))
        else:
            return (1)

    def deriv_fCosto(self, t, y):
        if (self.COSTO == 'ECM'):
            return (t - y) * self.derivar(y)
        else:
            return (t - y)

    def predict(self, X):
        """Return class label after unit step"""
        return self.evaluar(self.net_input(X))

    def predict_clase(self, X):
        """Retorna un entero con el índice de la clase más probable """
        y_hat = self.predict(X)
        if (self.FUN == 'tanh'):
            return (2 * (y_hat > 0) * 1 - 1)
        elif (self.FUN == 'sigmoid'):
            return ((y_hat > 0.5) * 1)
        else:
            return (X)

    def accuracy(self, X, y):
        y_hat = self.predict_clase(X)
        OK = np.sum(y_hat == y)
        return (OK / X.shape[0])

    def _score_metric(self, y_true, y_pred):
        loss = self.fCosto(y_true, y_pred)
        loss = np.mean(loss.astype(float))

        return (self.COSTO.lower(), loss)