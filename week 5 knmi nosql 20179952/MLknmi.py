import pandas as pd 
import matplotlib.pyplot as pt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('etmgeg_331.csv')
cols = df.columns[df.dtypes.eq('float64')]
df[cols] = df[cols].fillna(0).astype(np.int64)
df.dropna()

# print(df.describe()) 
# print(df.corr() ) 
#De correlatie tussen FXX(hoogste windstoot) en FHX(hoogste uurgemmidelde windsnelheid) is 0.94. De hardste windstoot is afhankelijk van de het hoogste windsnelheid

X = df['FXX'].values.reshape(-1,1)
y = df['FHX']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
lr = LinearRegression()
lr.fit(X_train,y_train)
print("Score train data: " + str(round(lr.score(X_train, y_train),2)))
print("Score test data: " + str(round(lr.score(X_test, y_test),2)))
print(lr.predict([[10]]))

pickle.dump(lr, open('modelSAL.pkl','wb'))