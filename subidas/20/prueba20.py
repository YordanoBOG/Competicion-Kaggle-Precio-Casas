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

#'''
from sklearn.impute import SimpleImputer
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
#from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
imputer_cat = SimpleImputer(strategy="most_frequent") #IterativeImputer(estimator=VotingClassifier(estimators=4), initial_strategy='most_frequent', max_iter=10, random_state=0)
imputer_cat.fit(input_all[col_cat])
train[col_cat] = imputer_cat.transform(train[col_cat])
test[col_cat] = imputer_cat.transform(test[col_cat])

# Reemplazo los valores numéricos perdidos por la mediana.
#'''
#from sklearn.impute import SimpleImputer
col_num = list(train.select_dtypes(include=np.number).columns)
col_num.remove('SalePrice')
imputer_num = SimpleImputer(strategy="median") #IterativeImputer(estimator=RandomForestRegressor(), initial_strategy='median', max_iter=30, random_state=0)
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


##############################################
#'''
#from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, VotingRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import sklearn

#------------------------------------------------------------------------------
# FUNCIÓN QUE CALCULA LA MÉTRICA DE LA COMPETICIÓN Y LA MUESTRA POR PANTALLA
#
def mostrar_evaluacion_modelo(model, X, y, scoring='neg_mean_squared_log_error', cv=5):
    # Uso la métrica que aplica la competición. Los modelos tienden a maximizar, por eso la métrica de error tiene signo negativo.
    # Estas son las métricas posibles
    sklearn.metrics.get_scorer_names()
    values = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    print()
    print("SCORES SOBRE LOS DATOS DE ENTRENAMIENTO:")
    print(values)
    print(values.mean())

#----------------------------------------------------------------------------
# FUNCIÓN QUE CALCULA LA MÉTRICA DE LA COMPETICIÓN Y LA DEVUELVE. SE USARÁ PARA VALIDAR LOS MODELOS
def evaluar_modelo(modelo, X, y, scoring='neg_mean_squared_log_error', cv=5):
    scores = cross_val_score(modelo, X, y, scoring=scoring, cv=cv)
    return scores

#------------------------------------------------------------------------------
# FUNCIÓN QUE CONSTRUYE EL MODELO DE HIST GRADIENT BOOSTING REGRESSOR CON LOS MEJORES PARÁMETROS OBTENIDOS HASTA EL MOMENTO
def obtener_hist_grad_boosting_reg():
    parameters = {
                    'loss':['squared_error', 'absolute_error', 'poisson', 'quantile'], # 'squared_error', 'absolute_error', 'poisson', 'quantile'
                    'max_iter': [i*10 for i in range(10,15)], # i*10 for i in range(10,15)
                    'max_leaf_nodes': [30, 40, 50],
                    'min_samples_leaf': [20, 30, 40], # 20, 30, 40
                    'max_bins': [255],
                    'n_iter_no_change': [1, 2, 3, 5, 10]
                 }
    #grid_search = GridSearchCV(HistGradientBoostingRegressor(), parameters)
    #print("Ajustando mejores parámetros para el modelo HistGradientBoosting base...")
    #grid_search.fit(X_train, y_train) # Entrenar conjunto de entrenamiento para encontrar mejores parámetros para el modelo base

    # Aislar la mejor combinación de parámetros
    #best_loss = grid_search.best_params_["loss"]
    #best_max_iter = grid_search.best_params_["max_iter"]
    #best_max_leaf_nodes = grid_search.best_params_["max_leaf_nodes"]
    #best_min_samples_leaf = grid_search.best_params_["min_samples_leaf"]
    #best_max_bins = grid_search.best_params_["max_bins"]
    #best_n_iter_no_change = grid_search.best_params_["n_iter_no_change"]
    #print("\n..... MEJORES PARÁMETROS:\n"+str(grid_search.best_params_))

    # Construir un modelo de regresión con los mejores parámetros encontrados
    bagged_model = HistGradientBoostingRegressor(loss='poisson', max_iter=140,
                                                max_leaf_nodes=40, min_samples_leaf=30,
                                                max_bins=255, n_iter_no_change=1)
    bagging_model = BaggingRegressor(bagged_model, n_estimators=100)
    return bagging_model

#------------------------------------------------------------------------------
# FUNCIÓN QUE CONSTRUYE EL MODELO DE GRADIENT BOOSTING CON LOS MEJORES PARÁMETROS OBTENIDOS HASTA EL MOMENTO
def obtener_gradient_boosting_reg():
    parameters = {
                    'loss':['squared_error', 'absolute_error', 'huber', 'quantile'], # 'squared_error', 'absolute_error', 'huber', 'quantile'
                    'criterion': ['friedman_mse', 'squared_error'], # 'friedman_mse', 'squared_error'
                    'min_samples_split': [7, 8, 9, 10, 11, 12, 13],
                    'min_samples_leaf': [1, 2, 3], # 1, 2, 3
                    'max_depth': [6],
                    'max_features': ['auto', 'sqrt', 'log2'], # 'auto', 'sqrt', 'log2'
                 }
    #grid_search = GridSearchCV(GradientBoostingRegressor(), parameters)
    #print("Ajustando mejores parámetros para el modelo Gradient Boosting base...")
    #grid_search.fit(X_train, y_train) # Entrenar conjunto de entrenamiento para encontrar mejores parámetros para el modelo base

    # Aislar la mejor combinación de parámetros
    #best_loss = grid_search.best_params_["loss"]
    #best_criterion = grid_search.best_params_["criterion"]
    #best_min_samples_split = grid_search.best_params_["min_samples_split"]
    #best_min_samples_leaf = grid_search.best_params_["min_samples_leaf"]
    #best_max_depth = grid_search.best_params_["max_depth"]
    #best_max_features = grid_search.best_params_["max_features"]
    #print("\n..... MEJORES PARÁMETROS:\n"+str(grid_search.best_params_))

    # Construir un modelo de regresión con los mejores parámetros encontrados
    bagged_model = GradientBoostingRegressor(loss='squared_error', criterion='squared_error',
                                            min_samples_split=8,
                                            min_samples_leaf=2, max_depth=6,
                                            max_features='log2')
    bagging_model = BaggingRegressor(bagged_model, n_estimators=100)
    return bagging_model

#------------------------------------------------------------------------------
# FUNCIÓN QUE CONSTRUYE EL MODELO DE XGBOOST CON LOS MEJORES PARÁMETROS OBTENIDOS HASTA EL MOMENTO
def obtener_xgboost():
    xgboost = xgb.XGBRegressor(min_child_weight=3,
                           subsample=0.9, colsample_bytree=0.8,
                           max_depth=5, n_estimators=100)
    bagged_xgb = BaggingRegressor(xgboost, n_estimators=100)
    return bagged_xgb

#------------------------------------------------------------------------------
# FUNCIÓN QUE CONSTRUYE EL MODELO DE VOTING CON LOS MEJORES MODELOS OBTENIDOS HASTA EL MOMENTO
def obtener_voting():
    # Definir modelos base
    nivel0 = list()
    nivel0.append(('xgb', obtener_xgboost()))
    print("\nXGBOOST MONTADO.")
    nivel0.append(('histgradboost', obtener_hist_grad_boosting_reg()))
    print("\nHIST GRADIENT BOOSTING MONTADO.")
    nivel0.append(('gradboost', obtener_gradient_boosting_reg()))
    print("\nGRADIENT BOOSTING MONTADO.")
    print("\nMONTANDO VOTING...")
    voting_model = VotingRegressor(estimators=nivel0)
    print("VOTING MONTADO.")
    return voting_model

#------------------------------------------------------------------------------
# FUNCIÓN QUE LLAMA A TODAS LAS FUNCIONES PARA CONSTRUIR MODELOS Y LOS DEVUELVE EN UN DICCIONARIO
def obtener_modelos():
    modelos = dict()
    #modelos['xgb'] = obtener_xgboost()
    #modelos['arbolreg'] = obtener_arbol_reg()
    #modelos['knn'] = obtener_knn()
    #modelos['linearreg'] = obtener_linear_regressor()
    #modelos['svr'] = obtener_svm_reg()
    modelos['voting'] = obtener_voting()
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

mostrar_evaluacion_modelo(model=modelos['voting'], X=X_train, y=y_train)

# Ahora vamos a entrenar el modelo con todo el conjunto de entrenamiento
print("\nEntrenando modelo...")
modelos['voting'].fit(X_train, y_train)

# Ahora predigo.
print("\nPrediciendo conjunto de test..")
pred = modelos['voting'].predict(X_test)

# Guardo el fichero de salida para evaluar:
print("\nEscribiendo resultados...")
ruta_salida = './prediccion.csv'
salida = pd.DataFrame({'Id': test_ids, 'SalePrice': pred})
salida.to_csv(ruta_salida, index=False)
print("\nFIN\n")
#'''
