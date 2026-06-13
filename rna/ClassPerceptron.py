from rna.grafica import *
from rna.ClassNeuronaBase import NeuronaBase


class Perceptron(NeuronaBase):
    """Perceptron classifier.
    Parameters
    ------------
    alpha : float
        Learning rate (between 0.0 and 1.0)
    epochs : int
        Passes over the training dataset.
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

    def __init__(self, alpha=0.01, epochs=50, random_state=None, draw=0, title=['X1', 'X2'], verbose=1):
        self.alpha = alpha
        self.epochs = epochs
        self.random_state = random_state  # -- asignar el valor 1 para fijar la semilla por defecto es aleatorio
        self.draw = draw
        self.title = title
        self.verbose = verbose

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

        rgen = np.random.RandomState(self.random_state)

        # self.w_ = rgen.normal(loc=0.0, scale=0.01,size=1 + X.shape[1])

        # self.w_ = rgen.uniform(-0.5, 0.5, size= X.shape[1])
        # self.b_ = rgen.uniform(-0.5, 0.5)
        self._weights_init(X.shape[1])
        self.errors_ = []
        ph = 0  # manejador de la recta mientras se dibuja
        errors = 1
        i = 0
        while ((i < self.epochs) and (errors > 0.0)):
            errors = 0
            for xi, target in zip(X, y):
                update = self.alpha * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

            # graficar la recta
            if (self.draw):
                ph = dibuPtosRecta(X, y, self.w_, self.b_, self.title, ph)

            # progreso de entrenamiento
            if self.verbose:
                y_pred = self.predict(X)
                self._show_progress(i, y, y_pred)

            i = i + 1

        if self.verbose:
            print()  # Salto de línea al finalizar

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def prob_positive_class(self, X):
        """Confidence level for class > 0 """
        netas = self.net_input(X)
        return 1 / (1 + np.exp(-netas))

    def _score_metric(self, y_true, y_pred):
        """
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
        valor = np.mean(y_true == y_pred)
        return ('accuracy', valor)
