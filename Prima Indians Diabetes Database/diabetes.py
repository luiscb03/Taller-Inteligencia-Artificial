import pandas as pd
import numpy as np

url = 'diabetes.csv'
data1 = pd.read_csv(url)
data1.drop(['Pregnancies', 'BloodPressure', 'SkinThickness'], axis=1, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data1.Age = pd.cut(data1.Age, rangos, labels=nombres)

#dividimos los datos en dos
data1_train = data1[:384]
data1_test = data1[384:]

 