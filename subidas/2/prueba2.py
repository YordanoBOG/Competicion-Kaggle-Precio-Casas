# ## Carga de librería
#
# Lo primero es cargar las librerías.
import pandas as pd
import numpy as np
import seaborn as sns # Instalar seaborn
from matplotlib import pyplot as plt

# + [markdown] jupyter={"outputs_hidden": false}
# ## Lectura de datos
#
# Ahora leemos los datos.
carpeta_datos='../../datos/'
train = pd.read_csv(carpeta_datos+'train.csv', na_values="NaN") # Definimos na_values para identificar bien los valores perdidos
train.columns

# Vamos a hacer unas ligeras visualizaciones de ejemplo.
sns.displot(data=train, x="SalePrice", aspect=3, kde=True)

# Ahora visualizamos cómo cambia la distribución (usando un boxplot) según otro atributo, como el tipo de calle.
sns.catplot(data=train, y="SalePrice", x="Street", kind="box")

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
col_cat

# Compruebo que son realmente categóricos.
input_all[col_cat].head()

# ## Proceso valores perdidos
test.isnull().sum()
train.isnull().sum()

# Hay valores perdidos tanto en el conjunto de entrenamiento como en el de test.
from sklearn.impute import SimpleImputer

# Voy a reemplazar los valores categóricos por el más frecuente (es mejorable)
imputer_cat = SimpleImputer(strategy="most_frequent")
imputer_cat.fit(input_all[col_cat])
train[col_cat] = imputer_cat.transform(train[col_cat])
test[col_cat] = imputer_cat.transform(test[col_cat])

# Compruebo que la variable objetivo no tenga valores nulos. Si fuese el caso habría que borrar dichas instancias.
train.SalePrice.isnull().sum()

# Ahora reemplazo los valores numéricos por la mediana.
col_num = list(train.select_dtypes(include=np.number).columns)
col_num.remove('SalePrice')
imputer_num = SimpleImputer(strategy="median")
imputer_num.fit(input_all[col_num])
train[col_num] = imputer_num.transform(train[col_num])
test[col_num] = imputer_num.transform(test[col_num])

# ## Hago el etiquetado
#
# Ahora hago el etiquetado con LabelEncoder, usando un diccionario de LabelEncoder
from sklearn.preprocessing import LabelEncoder
labelers = {}
test_l = test.copy()
train_l = train.copy()

for col in col_cat:
    labelers[col] = LabelEncoder().fit(input_all[col])
    test_l[col] = labelers[col].transform(test[col])
    train_l[col] = labelers[col].transform(train[col])

train_l.head()

# Compruebo que estén todos los atributos
assert((train_l.columns == train.columns).all())

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

# ## Aplico modelo
#
# Voy a aplicar un modelo muy sencillo, un árbol de decisión.
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()

from sklearn.model_selection import cross_val_score

# Uso la métrica que aplica la competición. Los modelos tienden a maximizar, por eso la métrica de error tiene signo negativo.
import sklearn
# Estas son las métricas posibles
sklearn.metrics.get_scorer_names()
values = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_log_error', cv=5)
print(values)
print(values.mean())

# Ahora vamos a entrenar con todo el conjunto de entrenamiento
model.fit(X_train, y_train)

# Ahora predigo.
pred = model.predict(X_test)

# Guardo el fichero de salida para evaluar:
ruta_salida = './prediccion.csv'
salida = pd.DataFrame({'Id': test_ids, 'SalePrice': pred})
salida.to_csv(ruta_salida, index=False)
