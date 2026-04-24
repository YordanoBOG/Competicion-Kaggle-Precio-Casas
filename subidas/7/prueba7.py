# ## Carga de librería
#
# Lo primero es cargar las librerías.
import pandas as pd
import numpy as np

# Ahora leemos los datos.
carpeta_datos='../../datos/'
train = pd.read_csv(carpeta_datos+'train.csv', na_values="NaN") # Definimos na_values para identificar bien los valores perdidos
train.columns

# Vamos a procesar datos:
# - Valores perdidos.
# - Etiquetado.
# Primero quito el Id de train que no me sirve de nada, y complica el etiquetado.
if 'Id' in train:
    train.drop('Id', axis=1, inplace=True)

#  También la quito de test pero antes lo guardo (para el fichero de salida)
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
test = pd.read_csv(carpeta_datos+'test.csv', na_values="NaN")
test_ids = test.Id
test = test.drop('Id', axis=1)

# Concateno la entrada de ambos para los procesos de etiquetado, que aprenda con ambos conjuntos
input_all = pd.concat([train.drop('SalePrice', axis=1), test])
input_all.columns

# Ahora selecciono los atributos de tipo categórico (los que no son numéricos)
col_cat = list(input_all.select_dtypes(exclude=np.number).columns)

# Reemplazo los valores numéricos perdidos por la mediana.
#'''
from sklearn.impute import SimpleImputer
col_num = list(train.select_dtypes(include=np.number).columns)
col_num.remove('SalePrice')
imputer_num = SimpleImputer(strategy="median")
imputer_num.fit(input_all[col_num])
train[col_num] = imputer_num.transform(train[col_num])
test[col_num] = imputer_num.transform(test[col_num])
#'''

# Ahora hago el etiquetado con LabelEncoder, usando un diccionario de LabelEncoder
#from sklearn.preprocessing import LabelEncoder
test_l = test.copy()
train_l = train.copy()

#'''
labelers = {}
for col in col_cat:
    labelers[col] = LabelEncoder().fit(input_all[col])
    test_l[col] = labelers[col].transform(test[col])
    train_l[col] = labelers[col].transform(train[col])
#'''

# ## Ahora preparo los conjuntos de entrenamiento y test
#
# Defino en X_train los valores sin el atributo a predecir, y.
#
# También voy a eliminar el Id de entrenamiento que es problemático, pero lo guardo para el fichero de salida.
y_train = train_l.SalePrice
X_train = train_l.drop('SalePrice', axis=1)

if 'Id' in test_l:
    test_l.drop('Id', axis=1, inplace=True)

X_test = test_l

# Escalar variables
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train) # fit only on training data
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) # apply same transformation to test data

# ## Aplico modelo
#
# Voy a aplicar un modelo de árbol de regresión de bagging
from sklearn.neighbors import KNeighborsRegressor

# Construir un modelo de regresión con los mejores parámetros encontrados
bagged_model = KNeighborsRegressor(n_neighbors=8, weights='distance', algorithm='auto', leaf_size=1)

# Contruir modelo de bagging con 100 modelos a partir del mejor modelo KNN encontrado
from sklearn.ensemble import BaggingRegressor
bagging_model = BaggingRegressor(bagged_model, n_estimators=100)

from sklearn.model_selection import cross_val_score
# Uso la métrica que aplica la competición. Los modelos tienden a maximizar, por eso la métrica de error tiene signo negativo.
import sklearn
# Estas son las métricas posibles
sklearn.metrics.get_scorer_names()
values = cross_val_score(bagging_model, X_train, y_train, scoring='neg_mean_squared_log_error', cv=5)
print()
print("SCORES SOBRE LOS DATOS DE ENTRENAMIENTO:")
print(values)
print(values.mean())

# Ahora vamos a entrenar el modelo de bagging con todo el conjunto de entrenamiento
print("\nEntrenando modelo de bagging...")
bagging_model.fit(X_train, y_train)

# Ahora predigo.
print("\nPrediciendo conjunto de test..")
pred = bagging_model.predict(X_test)

# Guardo el fichero de salida para evaluar:
print("\nEscribiendo resultados...")
ruta_salida = './prediccion.csv'
salida = pd.DataFrame({'Id': test_ids, 'SalePrice': pred})
salida.to_csv(ruta_salida, index=False)
print("\nFIN\n")
#'''
