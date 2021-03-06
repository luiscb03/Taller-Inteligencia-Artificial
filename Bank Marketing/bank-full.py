import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

url = 'bank-full.csv'
data = pd.read_csv(url)

data.drop(['balance', 'day', 'duration', 'campaign', 'pdays', 'poutcome'], axis=1, inplace=True)

#se convierten las edades en rangos
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)

#cambio de categorias en str a int
data.month.replace(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)
data.contact.replace(['unknown', 'cellular', 'telephone'], [0, 1, 2], inplace=True)
data.education.replace(['unknown', 'primary', 'secondary', 'tertiary'], [0, 1, 2, 3], inplace=True)
data.marital.replace(['single', 'married', 'divorced',], [0, 1, 2], inplace=True)
data.job.replace(['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)
data.loan.replace(['no', 'yes'], [0, 1], inplace=True)
data.default.replace(['no', 'yes'], [0, 1], inplace=True)
data.y.replace(['no', 'yes'], [0, 1], inplace=True)
data.housing.replace(['no', 'yes'], [0, 1], inplace=True)

data.dropna(axis=0,how='any', inplace=True)

#dividimos los datos en dos
data_train = data[:22605]
data_test = data[22605:]

x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y) # 0 no acepto, 1 si acepto
# Regresión Logística


# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
logreg.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')

# ARBOL DE DECISIÓN

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')

#REGRESION LINEAL

#seleccionar un modelo
rl = LinearRegression()

#entreamo el modelo
rl.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Regresion Lineal')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {rl.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {rl.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {rl.score(x_test_out, y_test_out)}')

#RANDOM FOREST

#seleccionar un modelo
rf = RandomForestClassifier()

#entreamo el modelo
rf.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Random Forest')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {rf.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {rf.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {rf.score(x_test_out, y_test_out)}')


-------------------------- RESULTADOS ----------------------------------------

- Regresión Logística
accuracy de Entrenamiento de Entrenamiento: 0.9524994470249945
accuracy de Test de Entrenamiento: 0.9493474894934749
accuracy de Validación: 0.814164381137751
**************************************************
- Maquina de soporte vectorial
accuracy de Entrenamiento de Entrenamiento: 0.9524994470249945
accuracy de Test de Entrenamiento: 0.9493474894934749
accuracy de Validación: 0.814164381137751
**************************************************
- Decisión Tree
accuracy de Entrenamiento de Entrenamiento: 0.9563149745631497
accuracy de Test de Entrenamiento: 0.9431541694315417
accuracy de Validación: 0.8044324515615323
**************************************************
- Regresion Lineal
accuracy de Entrenamiento de Entrenamiento: 0.00561185021906796
accuracy de Test de Entrenamiento: 0.001776179178510584
accuracy de Validación: -0.14879329458702584
**************************************************
- Random Forest
accuracy de Entrenamiento de Entrenamiento: 0.9562596770625967
accuracy de Test de Entrenamiento: 0.9460296394602964
accuracy de Validación: 0.8114659824825268



