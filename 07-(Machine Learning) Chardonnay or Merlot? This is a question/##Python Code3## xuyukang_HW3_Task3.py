#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
test = r"C:\Users\xuyuk\Documents\testdata.csv"
train = r"C:\Users\xuyuk\Documents\traindata.csv"
testdata = pd.read_csv(test)
traindata = pd.read_csv(train)
X_test,X_train,y_train,y_test= testdata.iloc[:,1:].values,traindata.iloc[:,1:].values,traindata['Class'].values,testdata['Class'].values


# In[37]:


#Manhattan distance with distance weights
from sklearn.neighbors import KNeighborsClassifier
resultsKNN= pd.DataFrame(columns=['KNN', 'Score for training','Score for testing'])
for knnCount in range (1,18):
    knn = KNeighborsClassifier(n_neighbors=knnCount, p=2, metric='minkowski',weights='distance')
    knn.fit(X_train, y_train)
    scoreTrain = knn.score(X_train, y_train)
    scoreTest = knn.score(X_test, y_test)
    resultsKNN.loc[knnCount] = [knnCount, scoreTrain, scoreTest]
print(resultsKNN.head(11))
resultsKNN.pop("KNN")
ax = resultsKNN.plot()


# In[36]:


#Minkowski distance with distance weights 
from sklearn.neighbors import KNeighborsClassifier
resultsKNN= pd.DataFrame(columns=['KNN', 'Score for training','Score for testing'])
for knnCount in range (1,18):
    knn = KNeighborsClassifier(n_neighbors=knnCount, p=1, metric='minkowski',weights='distance')
    knn.fit(X_train, y_train)
    scoreTrain = knn.score(X_train, y_train)
    scoreTest = knn.score(X_test, y_test)
    resultsKNN.loc[knnCount] = [knnCount, scoreTrain, scoreTest]
print(resultsKNN.head(11))
resultsKNN.pop("KNN")
ax = resultsKNN.plot()


# In[34]:


#Manhattan distance with uniform weights
from sklearn.neighbors import KNeighborsClassifier
resultsKNN= pd.DataFrame(columns=['KNN', 'Score for training','Score for testing'])
for knnCount in range (1,18):
    knn = KNeighborsClassifier(n_neighbors=knnCount, p=2, metric='minkowski')
    knn.fit(X_train, y_train)
    scoreTrain = knn.score(X_train, y_train)
    scoreTest = knn.score(X_test, y_test)
    resultsKNN.loc[knnCount] = [knnCount, scoreTrain, scoreTest]
print(resultsKNN.head(11))
resultsKNN.pop("KNN")
ax = resultsKNN.plot()


# In[35]:


#Minkowski distance with uniform weights 
from sklearn.neighbors import KNeighborsClassifier
resultsKNN= pd.DataFrame(columns=['KNN', 'Score for training','Score for testing'])
for knnCount in range (1,18):
    knn = KNeighborsClassifier(n_neighbors=knnCount, p=1, metric='minkowski')
    knn.fit(X_train, y_train)
    scoreTrain = knn.score(X_train, y_train)
    scoreTest = knn.score(X_test, y_test)
    resultsKNN.loc[knnCount] = [knnCount, scoreTrain, scoreTest]
print(resultsKNN.head(11))
resultsKNN.pop("KNN")
ax = resultsKNN.plot()


# In[ ]:




