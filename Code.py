import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

dataset = pd.read_csv('student-mat.csv', sep=';')
x = dataset.iloc[:, [6, 7, 12, 13, 14, 17, 21, 25, 29, 30, 31]].values
y = dataset.iloc[:, -1].values
le = LabelEncoder()
x[:, 5] = le.fit_transform(x[:, 5])
x[:, 6] = le.fit_transform(x[:, 6])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

linear_pipeline = Pipeline([('pcalr', PCA(n_components=2)),
                            ('linear_regressor', LinearRegression())])
svr_pipeline = Pipeline([('pcasvr', PCA(n_components=2)),
                         ('supportVector_regressor', SVR(kernel='rbf'))])
decisionTree_pipeline = Pipeline([('pcadtr', PCA(n_components=2)),
                                  ('decisionTree_regressor', DecisionTreeRegressor(random_state=0))])
rf_pipeline = Pipeline([('rf_regressor', RandomForestRegressor(n_estimators=50, random_state=0))])

pipe_dict = {0: 'Linear Regression', 1: 'Support Vector Regression', 2: 'Decision Tree Regression', 3: 'Random Forest Regression'}
pipelines = [linear_pipeline, svr_pipeline, decisionTree_pipeline, rf_pipeline]

for pipeline in pipelines:
    pipeline.fit(x_train, y_train)

for i, model in enumerate(pipelines):
    print("{} Test Accuracy = {} %".format(pipe_dict[i], model.score(x_test, y_test)*100))







