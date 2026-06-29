# 🧠 RNA - Redes Neuronales Artificiales

**RNA** es una biblioteca educativa en Python para el aprendizaje, experimentación y enseñanza de redes neuronales artificiales.

El proyecto busca ofrecer una alternativa simple y accesible para comprender los fundamentos del aprendizaje automático mediante implementaciones transparentes, documentación didáctica y ejemplos prácticos.

---

## 🚀 Inicio Rápido

Instalar directamente desde GitHub:

```bash
pip install https://github.com/RNA-UNIV/rna/archive/refs/heads/main.zip
```

Luego:

```python
import rna
```

---

## 📖 Demo Principal

La forma más rápida de conocer las capacidades de RNA es ejecutar la demostración principal.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoRNA.ipynb)
[![View Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoRNA.ipynb)

La demo incluye:

* Descarga y gestión de datasets tabulares y de audio.
* Exploración de datos con NumPy y Pandas.
* Preparación de conjuntos de entrenamiento y prueba.
* Entrenamiento de una red neuronal multicapa con Softmax.
* Evaluación mediante métricas de clasificación y matriz de confusión.
* Visualización de resultados.
* Carga y visualización de audios con `rna.visual`.
* Procesamiento de señales de audio con `rna.audio` (recorte de silencios, ajuste de duración y espectrograma Mel).

## 🧪 Demos adicionales

| Abrir |          Demo           | Descripción |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------:|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoRNA.ipynb) [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoRNA.ipynb)                           | **Neurona multiclase**  | Demo principal: red neuronal multicapa con Softmax para clasificación multiclase, carga de audios y procesamiento de señales. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoPerceptron.ipynb) [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoPerceptron.ipynb)             |     **Perceptrón**      | Implementación del perceptrón simple para clasificación binaria. Ejemplo práctico con el dataset Titanic. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoNeuronaLineal.ipynb) [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoNeuronaLineal.ipynb)       |   **Neurona lineal**    | Neurona lineal para regresión con descenso de gradiente. Ejemplo práctico con el dataset Automobile. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoNeuronaGradiente.ipynb) [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoNeuronaGradiente.ipynb) |  **Neurona Gradiente**  | Neurona con entrenamiento por descenso de gradiente. Ejemplo práctico con el dataset Occupancy Detection. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoImages.ipynb) [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoImages.ipynb)                     | **Dataset de imágenes** | Carga, visualización en grilla y Data Augmentation de imágenes usando TensorFlow Dataset. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RNA-UNIV/rna/blob/main/demos/DemoAudios.ipynb) [![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](demos/DemoAudios.ipynb)                     | **Dataset de audios**   | Carga y visualización de audios en grilla con `rna.visual`. Procesamiento de señales: recorte de silencios, ajuste de duración y extracción de espectrograma Mel. |
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
