import pandas as pd
import numpy as np

url = 'weatherAUS.csv'
data2 = pd.read_csv(url)
data2.drop(['Date', 'Sunshine', 'Evaporation', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm'], axis=1, inplace=True)
data2.drop(['Location', 'MinTemp', 'MaxTemp', 'WindSpeed9am', 'WindSpeed3pm'], axis=1, inplace=True)

data2.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace=True)
data2.RainToday.replace(['No', 'Yes'], [0, 1], inplace=True)
data2.WindGustDir.replace(['no', 'yes'], [0, 1], inplace=True)


