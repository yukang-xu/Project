#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[9]:


#drop features
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
filepath = r"C:\Users\xuyuk\.ipython\newwinedata.csv"
wineN = pd.read_csv(filepath)
wineN=wineN.drop(columns=['Proline', 'OD280/OD315 of diluted wines','Hue','Color intensity','Proanthocyanins'])
Y=wineN.pop('Class')#reference—https://stackoverflow.com/questions/36997619/sklearn-stratified-sampling-based-on-a-column
X=wineN
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.333, random_state=6, stratify=Y) 
new_train = X_train
new_test = X_test

resultsEntropy=pd.DataFrame(columns=['Level','Score for Training','Score for Testing'])
for treeDepth in range(1,11):
    dct=DecisionTreeClassifier(criterion='entropy',max_depth=treeDepth,random_state=0)
    dct=dct.fit(new_train,Y_train)
    dct.predict(new_test)
    scoreTest=dct.score(new_test,Y_test)
    scoreTrain=dct.score(new_train,Y_train)
    resultsEntropy.loc[treeDepth]=[treeDepth,scoreTrain,scoreTest]
print(resultsEntropy.head(11))
resultsEntropy.pop('Level')
ax1=resultsEntropy.plot()


# In[10]:


resultsGini=pd.DataFrame(columns=['Level','Score for Training','Score for Testing'])
for treeDepth in range(1,11):
    dct=DecisionTreeClassifier(criterion='gini',max_depth=treeDepth,random_state=0)
    dct=dct.fit(new_train,Y_train)
    dct.predict(new_test)
    scoreTest=dct.score(new_test,Y_test)
    scoreTrain=dct.score(new_train,Y_train)
    resultsGini.loc[treeDepth]=[treeDepth,scoreTrain,scoreTest]
print(resultsGini.head(11))
resultsGini.pop('Level')
ax2=resultsGini.plot()


# In[11]:


resultsKNN2=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=1,metric='minkowski')
    knn.fit(new_train,Y_train)
    scoreTrain=knn.score(new_train,Y_train)
    scoreTest=knn.score(new_test,Y_test)
    resultsKNN2.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN2.head(11))
resultsKNN2.pop('KNN')
ax4=resultsKNN2.plot()

resultsKNN4=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=1,metric='minkowski',weights='distance')
    knn.fit(new_train,Y_train)
    scoreTrain=knn.score(new_train,Y_train)
    scoreTest=knn.score(new_test,Y_test)
    resultsKNN4.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN4.head(11))
resultsKNN4.pop('KNN')
ax6=resultsKNN4.plot()


# In[12]:


resultsKNN2=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=2,metric='minkowski')
    knn.fit(new_train,Y_train)
    scoreTrain=knn.score(new_train,Y_train)
    scoreTest=knn.score(new_test,Y_test)
    resultsKNN2.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN2.head(11))
resultsKNN2.pop('KNN')
ax4=resultsKNN2.plot()

resultsKNN4=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=2,metric='minkowski',weights='distance')
    knn.fit(new_train,Y_train)
    scoreTrain=knn.score(new_train,Y_train)
    scoreTest=knn.score(new_test,Y_test)
    resultsKNN4.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN4.head(11))
resultsKNN4.pop('KNN')
ax6=resultsKNN4.plot()


# In[14]:


randomforest=pd.DataFrame(columns=['RandomForest','Score for Training','Score for Testing'])
for n in range(1,50):
    feat_labels = X.columns[:]
    forest=RandomForestClassifier(n_estimators=n,random_state=6,n_jobs=-1)
    forest.fit(new_train,Y_train)
    score1=forest.score(new_train, Y_train)
    score2=forest.score(new_test, Y_test)
    randomforest.loc[n]=[n,score1,score2]
print(randomforest.head(50))
randomforest.pop('RandomForest')
ax=randomforest.plot()


# In[15]:


new_train["Ash+Alca of Ash"] = new_train["Ash"] + new_train["Alcalinity of ash"] 
new_test["Ash+Alca of Ash"] = new_test["Ash"] + new_test["Alcalinity of ash"] 
new_train["Total Phenols+Non Phenols"] = new_train["Total phenols"] + new_train["Nonflavanoid phenols"]
new_test["Total Phenols+Non Phenols"] = new_test["Total phenols"] + new_test["Nonflavanoid phenols"]
new_train=new_train.drop(columns=['Total phenols', 'Nonflavanoid phenols','Ash','Alcalinity of ash'])
new_test=new_test.drop(columns=['Total phenols', 'Nonflavanoid phenols','Ash','Alcalinity of ash'])


# In[16]:


resultsEntropy=pd.DataFrame(columns=['Level','Score for Training','Score for Testing'])
for treeDepth in range(1,11):
    dct=DecisionTreeClassifier(criterion='entropy',max_depth=treeDepth,random_state=0)
    dct=dct.fit(new_train,Y_train)
    dct.predict(new_test)
    scoreTest=dct.score(new_test,Y_test)
    scoreTrain=dct.score(new_train,Y_train)
    resultsEntropy.loc[treeDepth]=[treeDepth,scoreTrain,scoreTest]
print(resultsEntropy.head(11))
resultsEntropy.pop('Level')
ax1=resultsEntropy.plot()

resultsGini=pd.DataFrame(columns=['Level','Score for Training','Score for Testing'])
for treeDepth in range(1,11):
    dct=DecisionTreeClassifier(criterion='gini',max_depth=treeDepth,random_state=0)
    dct=dct.fit(new_train,Y_train)
    dct.predict(new_test)
    scoreTest=dct.score(new_test,Y_test)
    scoreTrain=dct.score(new_train,Y_train)
    resultsGini.loc[treeDepth]=[treeDepth,scoreTrain,scoreTest]
print(resultsGini.head(11))
resultsGini.pop('Level')
ax2=resultsGini.plot()


# In[18]:


resultsKNN1=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=1,metric='minkowski')
    knn.fit(new_train,Y_train)
    scoreTrain=knn.score(new_train,Y_train)
    scoreTest=knn.score(new_test,Y_test)
    resultsKNN1.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN1.head(11))
resultsKNN1.pop('KNN')
ax3=resultsKNN1.plot()

resultsKNN2=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=1,metric='minkowski',weights='distance')
    knn.fit(new_train,Y_train)
    scoreTrain=knn.score(new_train,Y_train)
    scoreTest=knn.score(new_test,Y_test)
    resultsKNN2.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN2.head(11))
resultsKNN2.pop('KNN')
ax4=resultsKNN2.plot()

resultsKNN3=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=2,metric='minkowski')
    knn.fit(new_train,Y_train)
    scoreTrain=knn.score(new_train,Y_train)
    scoreTest=knn.score(new_test,Y_test)
    resultsKNN3.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN3.head(11))
resultsKNN3.pop('KNN')
ax5=resultsKNN3.plot()

resultsKNN4=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=2,metric='minkowski',weights='distance')
    knn.fit(new_train,Y_train)
    scoreTrain=knn.score(new_train,Y_train)
    scoreTest=knn.score(new_test,Y_test)
    resultsKNN4.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN4.head(11))
resultsKNN4.pop('KNN')
ax6=resultsKNN4.plot()


# In[19]:


randomforest=pd.DataFrame(columns=['RandomForest','Score for Training','Score for Testing'])
for n in range(1,25):
    feat_labels = X.columns[:]
    forest=RandomForestClassifier(n_estimators=n,random_state=6,n_jobs=-1)
    forest.fit(new_train,Y_train)
    score1=forest.score(new_train, Y_train)
    score2=forest.score(new_test, Y_test)
    randomforest.loc[n]=[n,score1,score2]
print(randomforest.head(25))
randomforest.pop('RandomForest')
ax=randomforest.plot()


# In[ ]:




