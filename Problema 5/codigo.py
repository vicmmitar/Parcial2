# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:37:05 2024

@author: vicmmitar
"""

import numpy as np
import random

# Distancias entre ciudades (matriz de adyacencia)
distancias = [
    [0, 7, 9, 8, 20],
    [7, 0, 10, 4, 11],
    [9, 10, 0, 15, 5],
    [8, 4, 15, 0, 17],
    [20, 11, 5, 17, 0]
]

# Número de ciudades
num_ciudades = len(distancias)

# Función de aptitud
def calcular_distancia(ruta):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        distancia_total += distancias[ruta[i]][ruta[i+1]]
    distancia_total += distancias[ruta[-1]][ruta[0]]  # Regresar al punto de partida
    return distancia_total

# Generar una ruta aleatoria
def generar_ruta_aleatoria():
    ruta = list(range(num_ciudades))
    random.shuffle(ruta)
    return ruta

# Crear la población inicial
def crear_poblacion(tamano_poblacion):
    return [generar_ruta_aleatoria() for _ in range(tamano_poblacion)]

# Selección por torneo
def seleccionar_padres(poblacion, tamano_torneo=3):
    padres = []
    for _ in range(len(poblacion)):
        torneo = random.sample(poblacion, tamano_torneo)
        padres.append(min(torneo, key=calcular_distancia))
    return padres

# Cruce de orden (OX)
def cruzar_ox(padre1, padre2):
    start, end = sorted(random.sample(range(num_ciudades), 2))
    hijo = [None] * num_ciudades
    hijo[start:end] = padre1[start:end]
    
    p2_index = end
    h_index = end
    while None in hijo:
        if padre2[p2_index % num_ciudades] not in hijo:
            hijo[h_index % num_ciudades] = padre2[p2_index % num_ciudades]
            h_index += 1
        p2_index += 1
    return hijo

# Mutación por intercambio
def mutar(ruta, tasa_mutacion=0.01):
    for i in range(num_ciudades):
        if random.random() < tasa_mutacion:
            j = random.randint(0, num_ciudades - 1)
            ruta[i], ruta[j] = ruta[j], ruta[i]
    return ruta

# Algoritmo Genético para TSP
def algoritmo_genetico(tamano_poblacion, generaciones, tasa_mutacion=0.01):
    poblacion = crear_poblacion(tamano_poblacion)
    mejor_ruta = min(poblacion, key=calcular_distancia)
    mejor_distancia = calcular_distancia(mejor_ruta)

    for _ in range(generaciones):
        padres = seleccionar_padres(poblacion)
        nueva_poblacion = []
        
        for i in range(0, len(padres), 2):
            padre1, padre2 = padres[i], padres[i + 1]
            hijo1 = cruzar_ox(padre1, padre2)
            hijo2 = cruzar_ox(padre2, padre1)
            nueva_poblacion.extend([mutar(hijo1, tasa_mutacion), mutar(hijo2, tasa_mutacion)])
        
        poblacion = nueva_poblacion
        ruta_actual = min(poblacion, key=calcular_distancia)
        distancia_actual = calcular_distancia(ruta_actual)
        
        if distancia_actual < mejor_distancia:
            mejor_ruta = ruta_actual
            mejor_distancia = distancia_actual

    return mejor_ruta, mejor_distancia

# ParÃ¡metros del algoritmo genÃ©tico
tamano_poblacion = 100
generaciones = 500
tasa_mutacion = 0.01

# Ejecutar el algoritmo genÃ©tico
mejor_ruta, mejor_distancia = algoritmo_genetico(tamano_poblacion, generaciones, tasa_mutacion)

print("Mejor ruta encontrada:", mejor_ruta)
print("Distancia de la mejor ruta:", mejor_distancia)
