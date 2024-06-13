# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 06:41:22 2024

@author: vicmmitar
"""

from sklearn import datasets
import numpy as np

# Cargamos el conjunto de datos Iris
iris = datasets.load_iris()

# Las entradas son las características de las flores Iris
entradas = iris.data

# La salida esperada es la especie de cada flor Iris (Setosa, Versicolor, Virginica)
salida_esperada = iris.target

# Convertimos la salida esperada a formato one-hot
salida_esperada_one_hot = np.zeros((salida_esperada.size, salida_esperada.max()+1))
salida_esperada_one_hot[np.arange(salida_esperada.size), salida_esperada] = 1

# Función de activación y su derivada
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    return x * (1 - x)

# Función de activación escalon y su derivada
def funcion_escalon(x):
    return np.where(x >= 0, 1, 0)

def derivada_funcion_escalon(x):
    return np.where(x != 0, 0, np.inf)

# Inicialización de los parámetros de la red neuronal
neuronas_capa_entrada = 4  # número de características en el conjunto de datos Iris
neuronas_capa_oculta = 3  # número de neuronas en la capa oculta
neuronas_capa_salida = 3  # número de clases en el conjunto de datos Iris

pesos_entrada_oculta = np.random.uniform(size=(neuronas_capa_entrada, neuronas_capa_oculta))
pesos_oculta_salida = np.random.uniform(size=(neuronas_capa_oculta, neuronas_capa_salida))

# Bucle de entrenamiento de la red neuronal
for _ in range(1):  # número de iteraciones
    # Propagación hacia adelante
    entrada_capa_oculta = np.dot(entradas, pesos_entrada_oculta)
    activaciones_capa_oculta = sigmoide(entrada_capa_oculta)

    entrada_capa_salida = np.dot(activaciones_capa_oculta, pesos_oculta_salida)
    salida = sigmoide(entrada_capa_salida)

    # Retropropagación
    error = salida_esperada_one_hot - salida
    d_salida = error * derivada_sigmoide(salida)
    
    error_capa_oculta = d_salida.dot(pesos_oculta_salida.T)
    d_capa_oculta = error_capa_oculta * derivada_sigmoide(activaciones_capa_oculta)

    # Actualización de los pesos
    pesos_oculta_salida += activaciones_capa_oculta.T.dot(d_salida) * 0.2
    pesos_entrada_oculta += entradas.T.dot(d_capa_oculta) * 0.2

for i in range(len(salida)):
   salida[i] = np.round(salida[i],1)
    
print(salida)