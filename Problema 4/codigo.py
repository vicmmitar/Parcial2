# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:25:37 2024

@author: vicmmitar
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import random

# Cargar datos
iris = load_iris()
X = iris.data
y = iris.target

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir la función objetivo
def funcion_objetivo(params):
    clf = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=params['learning_rate_init'], max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_test, y_pred)
    return accuracy_score(y_test, y_pred)

# Generar un vecino
def generar_vecino(params):
    vecino = params.copy()
    hl_sizes = params['hidden_layer_sizes']
    lr = params['learning_rate_init']
    
    # Cambiar el número de neuronas en la capa oculta o la tasa de aprendizaje
    if random.random() > 0.5:
        vecino['hidden_layer_sizes'] = (hl_sizes[0] + random.choice([-1, 1]),)
    else:
        vecino['learning_rate_init'] = lr * (1 + random.uniform(-0.1, 0.1))
    
    # Asegurar que los hiperparámetros están dentro de límites razonables
    vecino['hidden_layer_sizes'] = (max(1, vecino['hidden_layer_sizes'][0]),)
    vecino['learning_rate_init'] = max(0.0001, min(0.1, vecino['learning_rate_init']))
    
    return vecino

# Recocido simulado
def recocido_simulado(iteraciones=1000, temp_inicial=10, enfriamiento=0.99):
    solucion_actual = {'hidden_layer_sizes': (random.randint(1, 50),), 'learning_rate_init': random.uniform(0.001, 0.1)}
    mejor_solucion = solucion_actual
    mejor_valor = funcion_objetivo(solucion_actual)
    temp = temp_inicial
    
    for i in range(iteraciones):
        vecino = generar_vecino(solucion_actual)
        valor_vecino = funcion_objetivo(vecino)
        
        if valor_vecino > mejor_valor:
            mejor_solucion = vecino
            mejor_valor = valor_vecino
        
        # Decidir si se acepta el vecino
        if valor_vecino > funcion_objetivo(solucion_actual) or random.uniform(0, 1) < np.exp((valor_vecino - funcion_objetivo(solucion_actual)) / temp):
            solucion_actual = vecino
        
        # Enfriar la temperatura
        temp *= enfriamiento
    
    return mejor_solucion, mejor_valor

# Ejecutar el recocido simulado
mejor_solucion, mejor_valor = recocido_simulado()
print(f"Mejor solución: {mejor_solucion} con precisión: {mejor_valor:.4f}")
