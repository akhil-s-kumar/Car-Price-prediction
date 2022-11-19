#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

#Import Dataset
df = pd.read_csv('car data.csv')

df.dropna(inplace=True)
df.drop_duplicates()

df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
df.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
df.replace({'Seller_Type':{'Individual':0,'Dealer':1}},inplace=True)

y = df[['Selling_Price']]
X = df[['Present_Price', 'Seller_Type', 'Fuel_Type', 'Year', 'Kms_Driven']]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25)

model = LinearRegression()

model = model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

pickle.dump(model, open('model.pkl', 'wb'))

y_preds = pd.DataFrame({'actual': Y_test.squeeze(), 'predicted': y_pred.squeeze()})

accuracy = r2_score(Y_test,y_pred)*100
print("Accuracy of the model is: ", accuracy)
print('Train Score: ', model.score(X_train, Y_train)) 

a = pd.DataFrame({'Present_Price': [400000], 'Seller_Type': [0], 'Fuel_Type': [1], 'Year': [2011], 'Kms_Driven': [42450]})
a

test = model.predict(a)
print(int(test[0]))