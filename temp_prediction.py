#importing libraries
import pandas as pd
import numpy as np


#importing dataset
df = pd.read_excel('Day-wise planets degree and temperature.xlsx')


#changing date into numeric
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df.insert(3, 'Day', df.pop('Day'))
df.insert(4, 'Month', df.pop('Month'))
df.insert(5, 'Year', df.pop('Year'))
#print(df.head())


#selecting features and target variable
#['Sun Degree'(3),'Moon Degree'(4), 'Mars Degree'(5), 'Mercury Degree'(6),
#'Jupitor Degree'(7),'Venus Degree'(8), 'Saturn Degree'(9), 'Rahu Degree'(10),
#'Ketu Degree'(11), 'Day'(-3), 'Month'(-2), 'Year'(-1)]
X = df.iloc[:,3:]
X = np.array(X)
y_max = df.iloc[:,1]
y_max = np.array(y_max)
y_min = df.iloc[:,2]
y_min = np.array(y_min)


#splitting into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train_max, y_test_max = train_test_split(X,y_max,test_size = 0.1, random_state = 0)
X_train, X_test, y_train_min, y_test_min = train_test_split(X,y_min,test_size = 0.1, random_state = 0)


#Training Random Forest Regressor Model on Training set
from sklearn.ensemble import RandomForestRegressor
regressor1 = RandomForestRegressor(n_estimators = 100,random_state = 0)
regressor2 = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor1.fit(X_train, y_train_max)
regressor2.fit(X_train, y_train_min)


#saving model in pickle file
from joblib import dump
pickle_out = open("regressor1.pkl", mode = "wb") 
dump(regressor1, pickle_out) 
pickle_out.close()

from joblib import dump
pickle_out = open("regressor2.pkl", mode = "wb") 
dump(regressor2, pickle_out) 
pickle_out.close()

print("Model saved as pickle file named -> regressor1.pkl and regressor2.pkl ")

