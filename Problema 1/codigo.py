# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:34:44 2024

@author: vicmmitar
"""

from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
X = data.data
y = data.target
print(X, y)

entradas = X
pesos = np.array([1.0,1.0,1.0,1.0])
nuevosPesos = np.zeros((4,1))
ye=y
tasa = 0.4
yc=np.zeros((len(ye),1))
epoca=1
#while(not(yc.transpose()[0]==ye).all()):
while(epoca<=7):
    print("++++++++++EPOCA "+ str(epoca)+" +++++++++++++++++++++++++")
    for i in range(99):
        sumafx = np.dot(entradas[i],pesos)
        #print("sumafx")
        #print(sumafx)
        if sumafx>0:
            yc[i]=1
        else:
            yc[i]=0
        #print("yc")
        #print(yc)
        
        for j in range(len(entradas[i])):    
            nuevosPesos[j]=tasa*entradas[i][j]*(ye[i]-yc[i])
        
        #print("nuevosPesos")
        #print(nuevosPesos)
        pesos[0] = pesos[0] + nuevosPesos[0][0]
        pesos[1] = pesos[1] + nuevosPesos[1][0]
        pesos[2] = pesos[2] + nuevosPesos[2][0]
        pesos[3] = pesos[3] + nuevosPesos[3][0]            
        #print(pesos)
    
    print(ye)
    print(yc.transpose()[0])
    epoca += 1