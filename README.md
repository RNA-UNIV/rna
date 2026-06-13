# 🧠 RNA - Redes Neuronales Artificiales

**RNA** es una biblioteca educativa en Python para el aprendizaje, experimentación y enseñanza de redes neuronales artificiales.

El proyecto busca ofrecer una alternativa simple y accesible para comprender los fundamentos del aprendizaje automático mediante implementaciones transparentes, documentación didáctica y ejemplos prácticos.

---

## 🚀 Inicio Rápido

Instalar directamente desde GitHub:

```bash
pip install https://github.com/RNA-UNIV/rna/archive/refs/heads/main.zip
```

Luego:[README.md](README.md)

```python
import rna
```

---

## 📖 Demo Principal

La forma más rápida de conocer las capacidades de RNA es ejecutar la demostración principal.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoRNA.ipynb)
[![View Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoRNA.ipynb)


La demo incluye:

* Descarga y gestión de datasets.
* Exploración de datos.
* Preparación de conjuntos de entrenamiento y prueba.
* Entrenamiento de una red neuronal multicapa.
* Evaluación mediante métricas de clasificación.
* Visualización de resultados.


## 📖 Demo Principal

La forma más rápida de conocer las capacidades de RNA es ejecutar la demostración principal.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoRNA.ipynb)
[![View Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoRNA.ipynb)

La demo incluye:

* Descarga y gestión de datasets.
* Exploración de datos.
* Preparación de conjuntos de entrenamiento y prueba.
* Entrenamiento de una red neuronal multicapa.
* Evaluación mediante métricas de clasificación.
* Visualización de resultados.

## 🧪 Demos adicionales

| Demo | Descripción |
|------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoRNA.ipynb) [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoRNA.ipynb) | **DemoRNA** | Demo principal: red neuronal multicapa para clasificación con visualización completa. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoImages.ipynb) [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoImages.ipynb) | **DemoImages** | Clasificación de imágenes usando redes neuronales. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoPerceptron.ipynb) [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoPerceptron.ipynb) | **DemoPerceptron** | Implementación del perceptrón simple para problemas linealmente separables. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoNeuronaLineal.ipynb) [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoNeuronaLineal.ipynb) | **DemoNeuronaLineal** | Neurona lineal (regresión) y aprendizaje con mínimos cuadrados. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoNeuronaGradiente.ipynb) [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoNeuronaGradiente.ipynb) | **DemoNeuronaGradiente** | Neurona con entrenamiento por descenso de gradiente. |
---

## ✨ Características

### 📦 Gestión Integrada de Datasets

RNA incorpora herramientas para:

* Descubrir datasets disponibles.
* Descargar recursos automáticamente.
* Consultar información descriptiva.
* Cargar datos como NumPy o Pandas.

### 🧠 Elementos de Redes Neuronales

Implementación orientada al aprendizaje de:

* Clasificación binaria.
* Regresión lineal.
* Clasificación multiclase.
* Entrenamiento supervisado.


---

## 📚 Ejemplo

```python
from rna import DataLoader
from rna import RNMulticlase

loader = DataLoader()

(X, y) = loader.load_array("iris")

modelo = RNMulticlase()
modelo.fit(X, y)

predicciones = modelo.predict(X)
```

---

## 🏗️ Objetivos del Proyecto

RNA busca proporcionar una plataforma sencilla para comprender:

* Representación de datos.
* Entrenamiento supervisado.
* Propagación hacia adelante.
* Retropropagación del error.
* Evaluación de modelos.

El foco principal está puesto en la claridad conceptual y la experimentación práctica.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
