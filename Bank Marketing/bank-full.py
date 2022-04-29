import numpy as np
import pandas as pd

url = 'bank-full.csv'
data = pd.read_csv(url)

data.drop(['balance', 'day', 'duration', 'campaign', 'pdays', 'poutcome'], axis=1, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)



