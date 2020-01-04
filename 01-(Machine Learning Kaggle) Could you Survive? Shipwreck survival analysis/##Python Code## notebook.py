#!/usr/bin/env python
# coding: utf-8

# In[57]:


#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
#1.import data
import numpy as np
import pandas as pd
filepath = r"C:\Users\xuyuk\Documents\test.csv"
test = pd.read_csv(filepath)
filepath2 = r"C:\Users\xuyuk\Documents\train.csv"
train=pd.read_csv(filepath2)


# In[127]:


# explore the dataset 
#survival by sex
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#survival by age
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
#survival by pclass
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
#survival by Embarked 
grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# In[59]:


# indentify the type of data 
test.info()
train.info()
test.describe()
train.describe(include=['object'])
data1 = train.copy(deep = True)
data_cleaner = [data1, test]


# In[60]:


#completing 
for dataset in data_cleaner:    
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
drop_column = ['PassengerId','Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)
print(data1.isnull().sum())
print("-"*10)
print(test.isnull().sum())


# In[63]:


#CREATE
for dataset in data_cleaner:    
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
#cleanup rare title names
#print(data1['Title'].value_counts())
stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (data1['Title'].value_counts() < stat_min) #this will create a true false series with title name as index
#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
print("-"*10)
#preview data again
data1.info()
test.info()
data1.sample(10)


# In[65]:


#CONVERT
#code categorical data
label = LabelEncoder()
for dataset in data_cleaner:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

#define y variable aka target/outcome
Target = ['Survived']
#define x variables for original features aka feature selection
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')
#define x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')
#define x and y variables for dummy features original
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')
data1_dummy.head()


# In[ ]:





# In[116]:


#Correcting
import matplotlib.pyplot as plt
import numpy as np
plt.boxplot(data1["Age"])
plt.show()
plt.boxplot(data1["Fare"])
plt.show()
plt.boxplot(test["Age"])
plt.show()
plt.boxplot(data1["Fare"])
plt.show()
summary1=(data1[['Age', 'Fare']])
summary2=(test[['Age', 'Fare']])
for(name, series) in summary1.iteritems():
    print("\n" + "Analyzed Attribute Name:", name)
    print("** 5 Max Value:", summary1.nlargest(5, [name]))
    print("** 5 Min Value:", summary1.nsmallest(5, [name]) )
for(name, series) in summary2.iteritems():
    print("\n" + "Analyzed Attribute Name:", name)
    print("** 5 Max Value:", summary2.nlargest(5, [name]))
    print("** 5 Min Value:", summary2.nsmallest(5, [name]) )


# In[67]:


#split train and test data 
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)
print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))
train1_x_bin.head()


# In[131]:


## Support Vector Machines
from sklearn.svm import SVC, LinearSVC
svc = SVC()
svc.fit(train1_x, train1_y)
Y_pred = svc.predict(test1_x)
acc_svc = round(svc.score(train1_x, train1_y) * 100, 2)
acc_svc


# In[132]:


# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(train1_x, train1_y)
Y_pred = linear_svc.predict(test1_x)
acc_linear_svc = round(linear_svc.score(train1_x, train1_y) * 100, 2)
acc_linear_svc


# In[133]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines',  'Linear SVC'],
    'Score': [acc_svc,acc_linear_svc]})
models.sort_values(by='Score', ascending=False)


# In[ ]:




