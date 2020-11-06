import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('student-mat.csv', sep=';')
x = dataset.iloc[:, [6, 7, 12, 13, 14, 17, 21, 25, 29, 30, 31]].values
y = dataset.iloc[:, -1].values
le = LabelEncoder()
x[:, 5] = le.fit_transform(x[:, 5])
le1 = LabelEncoder()
x[:, 6] = le1.fit_transform(x[:, 6])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
accuracy = cross_val_score(estimator = regressor, X=x_train, y=y_train, cv=10)
print('Tested Accuracy = ', regressor.score(x_test, y_test)*100)
print('Accuracy: ', accuracy.mean()*100)




