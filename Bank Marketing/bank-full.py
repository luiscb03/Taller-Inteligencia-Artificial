import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
