# -*- coding: utf-8 -*-

#Indicamos si queremos que cada variable categórica se convierta en varias binarias (tantas como categorías), indicamos binarizar = True, o si preferimos que cada variable categórica se convierta simplemente a una numérica (ordinal), binarizar = False.
binarizar = False

'''
Leemos el conjunto de datos. Los valores perdidos notados como '?' se convierten a NaN, si no, se consideraría '?' como una categoría más.
'''

import pandas as pd

carpeta_datos='../../datos/'

if not binarizar:
    house_test = pd.read_csv(carpeta_datos+'test.csv', delimiter=',')
    house_train = pd.read_csv(carpeta_datos+'train.csv', delimiter=',')
else:
    house_test = pd.read_csv(carpeta_datos+'test.csv', na_values="?", delimiter=',')
    house_train = pd.read_csv(carpeta_datos+'train.csv', na_values="?", delimiter=',')

'''
Si el dataset contiene variables categóricas con cadenas, es necesario convertirlas a numéricas
antes de usar 'fit'. Si no las vamos a hacer ordinales (binarizar = True), las convertimos a
variables binarias con get_dummies. Para saber más sobre las opciones para tratar variables
categóricas: http://pbpython.com/categorical-encoding.html
'''
#'''
# devuelve una lista de las características categóricas excluyendo la columna 'SalePrice' (que contiene la clase), para los conjuntos de train y test
lista_categoricas_train = [x for x in house_train.columns if (house_train[x].dtype == object and house_train[x].name != 'SalePrice')]
binarizar = False
if not binarizar:
    house_train_convertida = house_train
else:
    # reemplaza las categóricas por binarias
    house_train_convertida = pd.get_dummies(house_train, columns=lista_categoricas_train)

lista_categoricas_test = [x for x in house_test.columns if (house_test[x].dtype == object and house_test[x].name != 'SalePrice')]
binarizar = False
if not binarizar:
    house_test_convertida = house_test
else:
    # reemplaza las categóricas por binarias
    house_test_convertida = pd.get_dummies(house_test, columns=lista_categoricas_test)
    
#Listas de atributos del dataset.
list(house_train_convertida)
list(house_test_convertida)
#'''

# Transformar las listas en 
'''
Separamos el DataFrame en dos arrays numpy, uno con las características (X) y otro con la clase (y).
En nuestro caso, la última columna es la que contiene la clase, por lo que se puede separar así:
'''
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
#'''
# LabelEncoder codifica los valores originales entre 0 y el número de valores - 1
# Se puede usar para normalizar variables o para transformar variables no-numéricas en numéricas
#'''
X_train = house_train_convertida.values[:,:len(house_train_convertida.columns)-1] #len(house_train_convertida.columns) devuelve 1 # Conjunto de entrenamiento
y_train = house_train_convertida.values[:,len(house_train_convertida.columns)-1]
y_train_bin = le.fit_transform(y_train)

# El conjunto de test no incluye la columna con la clase, por lo que no hace falta aislar la clase.
X_test = house_test_convertida.values

binarizar = False
if not binarizar:
    for i in range(0,X_train.shape[1]):
        for j in range(0,X_train.shape[0]):
            if isinstance(X_train[j,i],str): # Si encontramos una columna con valor de cadena
                X_train[:,i] = le.fit_transform(X_train[:,i]) # Pasamos los valores de la columna a número

    # X_test
    for i in range(0,X_test.shape[1]):
        for j in range(0,X_test.shape[0]):
            if isinstance(X_test[j,i],str): # Si encontramos una columna con valor de cadena
                X_test[:,i] = le.fit_transform(X_test[:,i]) # Pasamos los valores de la columna a número
#'''

# Aplicar algoritmo de árbol de regresión que acepta valores NaN
#'''
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
test_predict = regressor.predict(X_test)
#'''

# Contruir matriz con los resultados de la predicción
matriz_resultado = [[0 for _ in range(2)] for _ in range(1460)] # Matriz resultado de 2 columnas y 1460 filas que inicialmente está rellena con ceros
matriz_resultado[0][0] = 'Id'
matriz_resultado[0][1] = 'SalePrice'

# Rellenar matriz resultado
id_actual = 1461
fila_actual = 1
for prediccion in test_predict:
    matriz_resultado[fila_actual][0] = id_actual
    matriz_resultado[fila_actual][1] = prediccion
    fila_actual += 1
    id_actual += 1

#Exportar la matriz a un archivo .csv de salida
df_matriz_resultado = pd.DataFrame(matriz_resultado)
df_matriz_resultado.to_csv('./prediccion.csv', index=False, header=False)
