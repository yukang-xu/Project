#!/usr/bin/env python
# coding: utf-8

# In[47]:


#split dataset and export train,test data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
filepath = r"C:\Users\xuyuk\.ipython\newwinedata.csv"
winedata = pd.read_csv(filepath)
X,y= winedata.iloc[:,1:].values, winedata['Class'].values
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size= 0.33, random_state=0)
df1= pd.DataFrame(data=X_train)
df3= pd.DataFrame(data=y_train)
df2= pd.DataFrame(data=y_test)
df4= pd.DataFrame(data=X_test)
train_df = pd.concat([df3, df1], axis=1)
train_df.columns=winedata.columns
test_df = pd.concat([df2, df4], axis=1)
test_df.columns=winedata.columns
export_csv = train_df.to_csv (r'C:\Users\xuyuk\.ipython\traindata.csv', index = None, header=True) 
export_csv = test_df.to_csv (r'C:\Users\xuyuk\.ipython\testdata.csv', index = None, header=True) 


# In[59]:


#train first classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
#First we use entropy
resultEntropy = pd.DataFrame(columns = ["LevelLimit", "Score for Training", "Score for Testing"])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion = "entropy", max_depth = treeDepth, random_state = 0)
    dct = dct.fit(X_train, y_train)
    dct.predict(X_test)
    scoreTrain = dct.score(X_train, y_train)
    scoreTest = dct.score(X_test, y_test)
    resultEntropy.loc[treeDepth] = [treeDepth, scoreTrain, scoreTest]
print(resultEntropy.head(11))
resultEntropy.pop("LevelLimit")
ax = resultEntropy.plot()


# In[60]:


#second we try to use gini index
resultGini = pd.DataFrame(columns = ["LevelLimit", "Score for Training", "Score for Testing"])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion = "gini", max_depth = treeDepth, random_state = 0)
    dct = dct.fit(X_train, y_train)
    dct.predict(X_test)
    scoreTrain = dct.score(X_train, y_train)
    scoreTest = dct.score(X_test, y_test)
    resultGini.loc[treeDepth] = [treeDepth, scoreTrain, scoreTest]
print(resultGini.head(11))
resultGini.pop("LevelLimit")
ax = resultGini.plot()


# In[45]:


# entropy
from sklearn.tree import export_graphviz
dct1 = DecisionTreeClassifier(criterion = "entropy", max_depth = 3, random_state = 0)
dct1 = dct1.fit(X_train, y_train)
export_graphviz(dct1, out_file = "entropy.dot")
from graphviz import render 
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\graphviz-2.38\Library\bin\graphviz'
render('dot', 'png', 'entropy.dot')  
#predict class to test dataset
dct1.predict(X_test)
#test score
score = dct1.score(X_test, y_test)
print(score)
score = dct1.score(X_train, y_train)
print(score)


# In[49]:


from sklearn.tree import export_graphviz
dct2 = DecisionTreeClassifier(criterion = "gini", max_depth = 3, random_state = 0)
dct2 = dct2.fit(X_train, y_train)
export_graphviz(dct2, out_file = "tree_gini.dot")
from graphviz import render
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\graphviz-2.38\Library\bin\graphviz'
render('dot', 'png', 'tree_gini.dot')  
#predict class to test dataset
dct2.predict(X_test)
#test score
score = dct2.score(X_test, y_test)
print(score)
score = dct2.score(X_train, y_train)
print(score)


# In[58]:


import pandas as pd 
data = {'dataset':['train data', 'test data'], 'proportion':[2/3, 1/3],'instances':[79,39]} 
df = pd.DataFrame(data)  
print(df) 


# In[ ]:




