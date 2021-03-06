import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

url = 'diabetes.csv'
data1 = pd.read_csv(url)
data1.drop(['Pregnancies', 'BloodPressure', 'SkinThickness'], axis=1, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data1.Age = pd.cut(data1.Age, rangos, labels=nombres)

#dividimos los datos en dos
data1_train = data1[:384]
data1_test = data1[384:]

x = np.array(data1_train.drop(['Outcome'], 1))
y = np.array(data1_train.Outcome) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


x_test_out = np.array(data1_test.drop(['Outcome'], 1))
y_test_out = np.array(data1_test.Outcome) # 0 no acepto, 1 si acepto
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
accuracy de Entrenamiento de Entrenamiento: 0.7687296416938111
accuracy de Test de Entrenamiento: 0.7012987012987013
accuracy de Validación: 0.7942708333333334
**************************************************
- Maquina de soporte vectorial
accuracy de Entrenamiento de Entrenamiento: 0.9869706840390879
accuracy de Test de Entrenamiento: 0.6883116883116883
accuracy de Validación: 0.6953125
**************************************************
- Decisión Tree
accuracy de Entrenamiento de Entrenamiento: 1.0
accuracy de Test de Entrenamiento: 0.6753246753246753
accuracy de Validación: 0.6640625
**************************************************
- Regresion Lineal
accuracy de Entrenamiento de Entrenamiento: 0.30897688754473207
accuracy de Test de Entrenamiento: 0.07917379097397892
accuracy de Validación: 0.2854578845401293
**************************************************
- Random Forest
accuracy de Entrenamiento de Entrenamiento: 1.0
accuracy de Test de Entrenamiento: 0.7272727272727273
accuracy de Validación: 0.765625

 