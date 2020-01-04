#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
test = r"C:\Users\xuyuk\Documents\testdata.csv"
train = r"C:\Users\xuyuk\Documents\traindata.csv"
testdata = pd.read_csv(test)
traindata = pd.read_csv(train)
X_test,X_train,y_train,y_test= testdata.iloc[:,1:].values,traindata.iloc[:,1:].values,traindata['Class'].values,testdata['Class'].values


# In[26]:


from sklearn.ensemble import RandomForestClassifier
results=pd.DataFrame(columns=['Count of Trees','Score for Training','Score for Testing'])
indexR=1
for sizeOfForest in range(1,102,10):
    forest=RandomForestClassifier(criterion='gini',n_estimators=sizeOfForest,random_state=6)
    forest.fit(X_train,y_train)
    scoreTrain=forest.score(X_train,y_train)
    scoreTest=forest.score(X_test,y_test)
    results.loc[indexR]=[sizeOfForest,scoreTrain,scoreTest]
    indexR=indexR+1
print(results.head(16))
results.pop('Count of Trees')
ax=results.plot()   


# In[25]:


feat_labels=traindata.iloc[:,1:].columns
importance=forest.feature_importances_
indices = np.argsort(importance)[::-1]
for i in range(X_train.shape[1]):
    print("(%2d)%-*s%f"%(i+1,30, feat_labels[i], importance[indices[i]]))


# In[18]:


traindata.iloc[:,1:].columns


# In[ ]:




