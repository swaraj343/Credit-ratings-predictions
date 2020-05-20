# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:52:01 2020

@author: swara
"""

#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


#Use sklearn.cluster.KMeans to do clustering on the given data set points.csv.
#There are 4 clusters in this data set. Draw a scatter plot for the data and use color
#to indicate their clusters

points_df = pd.read_csv('points.csv')
points_df.head()
points_df.columns

kmeans = KMeans(n_clusters=4, random_state=0).fit(points_df)

kmeans.labels_

kmeans.cluster_centers_

labels = kmeans.labels_

y_kmeans = kmeans.predict(points_df)

y_kmeans

plt.scatter(points_df.x1, points_df.x2, c=y_kmeans, s=50, cmap='viridis')

#Regard the clusters given by your KMeans model as the ground truth labels, randomly 
#split the data set into training data (80%) and testing data (20%). 
#linear SVM classifier and train it on training data set. Use the confusion matrix
#to evaluate its performance on testing data set

points_df['label'] = labels 


points_df.head()
points_df.columns
points_df[['x1','x2']].head()

X_tr, X_ts, y_tr, y_ts = train_test_split(points_df[['x1','x2']], points_df['label'], test_size=0.2, random_state=42)

X_tr.shape
X_ts.shape

y_tr.shape
y_ts.shape

from sklearn.svm import SVC
from sklearn import metrics

model = SVC(kernel='linear')
model.fit(X_tr,y_tr)
predi = model.predict(X_ts)

X_ts.shape
print(metrics.accuracy_score(y_ts,predi))

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_ts,predi))

print(classification_report(y_ts,predi))

labels_df = pd.read_csv('labels.csv')

labels_df.head()

points_labels_df = pd.read_csv('points.csv')
points_labels_df.head()

label= labels_df['labels']  

points_labels_df['label'] = label

points_labels_df.head()

X_trn, X_tst, y_trn, y_tst = train_test_split(points_labels_df[['x1','x2']], points_labels_df['label'], test_size=0.2, random_state=42)

model.fit(X_trn,y_trn)

preds = model.predict(X_tst)

print(confusion_matrix(y_tst,preds))


from tensorflow import keras
import tensorflow as tf

#Neural Network
model = keras.Sequential()

# layers
model.add(keras.layers.Dense(12, input_dim=2, activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))


model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
hist=model.fit(points_labels_df[['x1','x2']], points_labels_df['label'], batch_size = 10, epochs = 150,verbose=0)


y_predic = model.predict(X_ts)
y_final = (y_predic > 0.5).astype(int).reshape(X_ts.shape[0])

print(y_final)
print(confusion_matrix(y_tst,y_final))

plt.plot(hist.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

print(metrics.accuracy_score(y_ts,y_final))

#In this question, we are going to use the diabetes data set. Use sklearn.datasets.load diabetes()
#to load the data and labels

from sklearn import datasets
from sklearn import linear_model

diab = datasets.load_diabetes()

diab.data.shape

X = diab.data[:,:]

y = diab.target


diab.target.shape

#Randomly split the data into training set (80%) and testing set (20%).

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create a linear regression model using sklearn, and fit training data. Evaluate
#your model using test data. Give all the coefficient and R-squared score.

regr_diab = linear_model.LinearRegression()
regr_diab.fit(X_train, y_train)
y_pred = regr_diab.predict(X_test)

# The coefficients
print('Coefficients: \n', regr_diab.coef_)

#intercept
print(regr_diab.intercept_)

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)

from sklearn.model_selection import cross_val_score


#Use 10-fold cross validation to fit and validate your linear regression models on
#the whole data set. Print the scores for each validation



print("The score for each validation are:",cross_val_score(regr_diab, X, y, cv=10))  

# Use sklearn to create RandomForestRegressor model, and fit the
#training data into it.

from sklearn.ensemble import RandomForestRegressor

rf_ = RandomForestRegressor(max_depth=7, random_state=0,n_estimators=100)
rf_.fit(X, y) 

print(rf_.feature_importances_)


rf_predi = rf_.predict(X)
print(rf_predi)


#Use Grid Search to find the optimal hyper-parameters (max depth:{None,
#7, 4} and min samples split: {2, 10, 20}) for RandomForestRegressor

from sklearn.model_selection import GridSearchCV


depth = [None,7,4]
estimators = [10,50,100,100]
min_samples_split = [2,10,20]

rform Grid-Search
diab_grid = GridSearchCV(estimator=RandomForestRegressor(),param_grid={'max_depth': depth,'n_estimators': estimators,'min_samples_split': min_samples_split},
        cv=5, scoring='neg_mean_squared_error', verbose=0,                         n_jobs=-1)
diab_grid_result = diab_grid.fit(X, y)
best_params = diab_grid_result.best_params_

best_params