import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Cargar datos
iris = load_iris()
X = iris.data
y = iris.target

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir función objetivo
def funcion_objetivo(params):
    clf = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=params['learning_rate_init'], max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Generar vecinos
def generar_vecinos(params):
    vecinos = []
    hl_sizes = params['hidden_layer_sizes']
    lr = params['learning_rate_init']
    
    # Generar vecinos cambiando el número de neuronas en la capa oculta
    vecinos.append({'hidden_layer_sizes': (hl_sizes[0] + 1,), 'learning_rate_init': lr})
    if hl_sizes[0] > 1:
        vecinos.append({'hidden_layer_sizes': (hl_sizes[0] - 1,), 'learning_rate_init': lr})
    
    # Generar vecinos cambiando la tasa de aprendizaje
    vecinos.append({'hidden_layer_sizes': hl_sizes, 'learning_rate_init': lr * 1.1})
    vecinos.append({'hidden_layer_sizes': hl_sizes, 'learning_rate_init': lr * 0.9})
    
    return vecinos

# Búsqueda local
def busqueda_local(iteraciones=10):
    # Inicializar con un conjunto de hiperparámetros aleatorio
    solucion_actual = {'hidden_layer_sizes': (5,), 'learning_rate_init': 0.01}
    mejor_valor = funcion_objetivo(solucion_actual)
    
    for _ in range(iteraciones):
        vecinos = generar_vecinos(solucion_actual)
        for vecino in vecinos:
            valor_vecino = funcion_objetivo(vecino)
            if valor_vecino > mejor_valor:
                solucion_actual = vecino
                mejor_valor = valor_vecino
                
    return solucion_actual, mejor_valor

# Ejecutar la búsqueda local
mejor_solucion, mejor_valor = busqueda_local()
print(f"Mejor solución: {mejor_solucion} con precisión: {mejor_valor:.4f}")
