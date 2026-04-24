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



'''
Validación cruzada con Stacking for regression
'''
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_score
import sklearn

#print(sklearn.__version__)

#------------------------------------------------------------------------------
# FUNCIÓN QUE CALCULA LA MÉTRICA DE LA COMPETICIÓN Y LA MUESTRA POR PANTALLA
#'''
def mostrar_evaluacion_modelo(model, X, y, scoring='neg_mean_squared_log_error', cv=5):
    # Uso la métrica que aplica la competición. Los modelos tienden a maximizar, por eso la métrica de error tiene signo negativo.
    # Estas son las métricas posibles
    sklearn.metrics.get_scorer_names()
    values = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    print()
    print("SCORES SOBRE LOS DATOS DE ENTRENAMIENTO:")
    print(values)
    print(values.mean())
#'''

#------------------------------------------------------------------------------
# FUNCIÓN QUE CALCULA LA MÉTRICA DE LA COMPETICIÓN Y LA DEVUELVE. SE USARÁ PARA VALIDAR LOS MODELOS
def evaluar_modelo(modelo, X, y, scoring='neg_mean_squared_log_error', cv=5):
    scores = cross_val_score(modelo, X, y, scoring=scoring, cv=cv)
    return scores

#------------------------------------------------------------------------------
# FUNCIÓN QUE CONSTRUYE EL MODELO DE XGBOOST CON LOS MEJORES PARÁMETROS OBTENIDOS HASTA EL MOMENTO
def obtener_xgboost():
    xgboost = xgb.XGBRegressor(min_child_weight=3,
                           subsample=0.9, colsample_bytree=0.8,
                           max_depth=5, n_estimators=100)
    bagged_xgb = BaggingRegressor(xgboost, n_estimators=100)
    return bagged_xgb

#------------------------------------------------------------------------------
# FUNCIÓN QUE CONSTRUYE EL MODELO DE ÁRBOL DE REGRESIÓN CON LOS MEJORES PARÁMETROS OBTENIDOS HASTA EL MOMENTO
def obtener_arbol_reg():
    arbol_reg = DecisionTreeRegressor(criterion='absolute_error', min_samples_split=4,
                                       min_samples_leaf=3, max_features=None,
                                       min_impurity_decrease=0.0, ccp_alpha=0.0)
    bagged_arbol_reg = BaggingRegressor(arbol_reg, n_estimators=100)
    return bagged_arbol_reg

#------------------------------------------------------------------------------
# FUNCIÓN QUE CONSTRUYE EL MODELO KNN CON LOS MEJORES PARÁMETROS OBTENIDOS HASTA EL MOMENTO
def obtener_knn():
    knn_reg = KNeighborsRegressor(n_neighbors=8, weights='distance', algorithm='auto', leaf_size=1)
    bagged_knn = BaggingRegressor(knn_reg, n_estimators=100)
    return bagged_knn

#------------------------------------------------------------------------------
# FUNCIÓN QUE CONSTRUYE EL MODELO DE STACKING CON LOS MEJORES MODELOS OBTENIDOS HASTA EL MOMENTO
def obtener_stacking():
    # Definir modelos base
    nivel0 = list()
    nivel0.append(('xgb', obtener_xgboost()))
    nivel0.append(('arbolreg', obtener_arbol_reg()))
    nivel0.append(('knn', obtener_knn()))
    stacking_model = StackingRegressor(estimators=nivel0)
    return stacking_model

#------------------------------------------------------------------------------
# FUNCIÓN QUE LLAMA A TODAS LAS FUNCIONES PARA CONSTRUIR MODELOS Y LOS DEVUELVE EN UN DICCIONARIO
def obtener_modelos():
    modelos = dict()
    #modelos['xgb'] = obtener_xgboost()
    #modelos['arbolreg'] = obtener_arbol_reg()
    #modelos['knn'] = obtener_knn()
    modelos['stacking'] = obtener_stacking()
    return modelos

#------------------------------------------------------------------------------

modelos = obtener_modelos()

# evaluar modelos y obtener resultados
'''
resultados_modelos, nombres_modelos = list(), list()
for nombre, modelo in modelos.items():
    scores = evaluar_modelo(modelo=modelo, X=X_train, y=y_train)
    resultados_modelos.append(scores)
    nombres_modelos.append(nombre)
    print('>%s %.3f (%.3f)' % (nombre, mean(scores), std(scores)))
#'''
    
mostrar_evaluacion_modelo(model=modelos['stacking'], X=X_train, y=y_train)

# Ahora vamos a entrenar el modelo de bagging con todo el conjunto de entrenamiento
print("\nEntrenando modelo...")
modelos['stacking'].fit(X_train, y_train)

# Ahora predigo.
print("\nPrediciendo conjunto de test..")
pred = modelos['stacking'].predict(X_test)

# Guardo el fichero de salida para evaluar:
print("\nEscribiendo resultados...")
ruta_salida = './prediccion.csv'
salida = pd.DataFrame({'Id': test_ids, 'SalePrice': pred})
salida.to_csv(ruta_salida, index=False)
print("\nFIN\n")
#'''
