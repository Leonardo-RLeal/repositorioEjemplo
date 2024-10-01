# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 12:27:16 2024

@author: lrl13
"""

import numpy as np 
import matplotlib.pyplot as plt 
from skimage import data
import time


#Genera data aleatoria de prueba
np.random.seed(42)  # Para reproducibilidad

imagen = data.camera()

# Inicializar una lista para almacenar los valores del histograma
histograma = [0] * 256
#Recorrer cada píxel de la imagen para calcular el histograma 
for fila in range(imagen.shape[0]):
    for columna in range(imagen.shape[1]):
        valor_pixel = imagen[fila,columna]
        histograma[valor_pixel] += 1

data = imagen.reshape(-1, 1)#convierte en vector columna

# Crear una figura con dos subgráficos
plt.figure(figsize=(12, 5))

# Mostrar la imagen original en el primer subgráfico
plt.close('all')
plt.subplot(1, 2, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')  # Ocultar los ejes

# Mostrar el histograma en el segundo subgráfico
plt.subplot(1, 2, 2)
plt.bar(range(256), histograma)
plt.xlabel('Valor de intensidad')
plt.ylabel('Frecuencia')
plt.title('Histograma de la Imagen')

# Mostrar el gráfico
plt.tight_layout()
plt.show()


iter_max = 100 #numero de iteraciones 


#----------------------------------
#Función para calcular distancias 
#----------------------------------
def calc_distancias(data,centros,num_clases):
    """Calcula la distancia euclidiana entre cada dato y cada centro"""
    distancias = np.zeros((data.shape[0],num_clases))
    #a es el número de clases
    for k in range(num_clases):
        distancias[:,k] = np.linalg.norm(data - centros[k],axis = 1)
    #Evitar divisiones por cero reemplazando ceros con un valor muy pequeño
    distancias = np.fmax(distancias, np.finfo(np.float64).eps)
    return distancias 

#----------------------------------
#Función para calcular WCSS 
#----------------------------------
def calc_wcss(data, labels, centros):
    wcss = 0
    for k in range(len(centros)):
        cluster_puntos = data[labels == k]
        wcss += np.sum((cluster_puntos - centros[k])**2)
    return wcss

#-----------------------------------
#Iteraciones del algoritmo k means 
#-----------------------------------
def calc_kmeans(max_clusters):
    inercias = []
   # max_clusters = 10
    for num_clases in range(1,max_clusters+1):
        # Inicializar los centros de manera aleatoria dentro del rango de intensidades
        centros = np.random.uniform(low=data.min(), high=data.max(), size=(num_clases, 1))
        labels = np.zeros(data.shape[0])
    
        for iteration in range(iter_max):
            #Guarda una copia de los centros anteriores para verificar convergencia
            centros_previos = centros.copy()
            
            #Paso 1 : Calcular distancias
            distancias = calc_distancias(data, centros,num_clases)
            #asignar cada punto al centroide más cercano
            
            labels = np.argmin(distancias, axis = 1)
           
            #Paso 3 : Recalcular los centroides
            for k in range(num_clases):
                centros[k] = data[labels == k].mean(axis = 0)
            
            #Paso 4 : Verificar convergencia
            if np.allclose(centros,centros_previos):
                print(f"Convergencia alcanzada en la iteración {iteration}")
                break
        wcss = calc_wcss(data, labels, centros)
        inercias.append(wcss)
    return centros,inercias
#-------------------------------------------------------
 
centrosF, inercias = calc_kmeans(10)
#normaliza el error
codo_norm = (inercias-np.min(inercias))/(np.max(inercias)-np.min(inercias))
# Graficar el método del codo
plt.figure(figsize=(8, 6))
plt.plot(range(1,11), codo_norm, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia (WCSS)')
plt.title('Método del codo para determinar el número óptimo de clusters')
plt.show()
time.sleep(5)



# claseF = int(input('cuantas claseses : '))
# centrosFF, inerciasFF = calc_kmeans(claseF)
# print(f"Centros finales:\n{centrosFF}")


