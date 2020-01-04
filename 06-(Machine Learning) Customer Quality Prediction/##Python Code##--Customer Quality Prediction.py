#!/usr/bin/env python
# coding: utf-8

# In[3]:


# loading package
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
filepath = r"C:\Users\xuyuk\Documents\testingCandidate.csv"
test = pd.read_csv(filepath)
filepath2 = r"C:\Users\xuyuk\Documents\training.csv"
train=pd.read_csv(filepath2)
import warnings
warnings.filterwarnings('ignore')
print('-'*25)


# In[8]:


# indentify the type of data 
test.info()
train.info()
test.describe()
train.describe(include=['object'])
data1 = train.copy(deep = True)
data_cleaner = [data1, test]


# In[15]:


#explore the data
data1['responded_Code'] = data1.responded
old = ['yes','no']
new = [1,0]
for i in range(len(old)):
    data1.responded_Code = data1.responded_Code.replace(old[i],new[i])


# In[ ]:


# explore relationship between age and marital
median1=data1.custAge[data1.marital=='single'].median()
median2=data1.custAge[data1.marital=='married'].median()
median3=data1.custAge[data1.marital=='divorced'].median()
mean1=data1.custAge[data1.marital=='single'].mean()
mean2=data1.custAge[data1.marital=='married'].mean()
mean3=data1.custAge[data1.marital=='divorced'].mean()
df = pd.DataFrame ({'mean' : pd.Series([mean1,mean2,mean3], index =['single', 'married', 'divorced' ]), 
      'median' : pd.Series([median1,median2,median3], index =['single', 'married', 'divorced' ])
      })
df


# In[ ]:


#explore default variable
count1=test.custAge[test.default=='yes'].count()
count2=test.custAge[test.default=='no'].count()
df1 = pd.DataFrame ({'default variable' : pd.Series([count1,count2], index =['yes', 'no']), 
      })
df1


# In[ ]:


#explore people who were never contacted
z1=data1.marital[data1.poutcome=='nonexistent'].count()
z2=data1.marital[data1.previous==0].count()
z3=data1.marital[data1.pmonths==999].count()
z4=data1.marital[data1.pdays==999].count()
z5=data1.marital[data1.campaign==0].count()
z6=data1.marital[data1.pastEmail==0].count()
df = pd.DataFrame ({'people who were never contacted ' : pd.Series([z1,z2,z3,z4,z5,z6], index =['poutcome', 'previous', 'pmonths','pdays','campaign','pastEmail' ]), 
      })
df


# In[9]:


#dropping
drop_column = ['schooling','default', 'month','day_of_week','campaign', 'pdays', 'pmonths','pastEmail','id']
data1.drop(drop_column, axis=1, inplace = True)
test.drop(drop_column, axis=1, inplace = True)
data1=data1[data1.profession != 'unknown']
test=test[test.profession != 'unknown']


# In[10]:


#converting 
old = ['unemployed', 'student', 'retired', 'housemaid','blue-collar', 'services', 'technician','admin.', 'entrepreneur', 'management', 'self-employed']
new = ['lowsalary','lowsalary','lowsalary','lowsalary','medsalary','medsalary','medsalary','highsalary','highsalary','highsalary','highsalary']
for i in range(len(old)):
    test.profession = test.profession.replace(old[i],new[i])
    data1.profession = data1.profession.replace(old[i],new[i])
label = LabelEncoder()
data1['poutcome_Code'] = label.fit_transform(data1['poutcome'])
data1['contact_Code'] = label.fit_transform(data1['contact'])
data1['profession_Code'] = label.fit_transform(data1['profession'])
test['poutcome_Code'] = label.fit_transform(test['poutcome'])
test['contact_Code'] = label.fit_transform(test['contact'])
test['profession_Code'] = label.fit_transform(test['profession'])
old = [0, 1, 2,4]
new = [4,0,1,2]
for i in range(len(old)):
    test.profession_Code = test.profession_Code.replace(old[i],new[i])
    data1.profession_Code = data1.profession_Code.replace(old[i],new[i])
old = [0,1,4]
new = [4,0,1]
for i in range(len(old)):
    test.poutcome_Code = test.poutcome_Code.replace(old[i],new[i])
    data1.poutcome_Code = data1.poutcome_Code.replace(old[i],new[i])
data1=data1[data1.marital != 'unknown']
test=test[test.marital != 'unknown']
label = LabelEncoder()
data1['marital_Code'] = label.fit_transform(data1['marital'])
test['marital_Code'] = label.fit_transform(test['marital'])


# In[11]:


#creating+ completing housing and loan variable
loan_dict={'no':0,'yes':1,'unknown':5}
test['loan_help']=test.loan.map(loan_dict)
data1['loan_help']=data1.loan.map(loan_dict)
test['housing_help']=test.housing.map(loan_dict)
data1['housing_help']=data1.housing.map(loan_dict)
data1['loan_Code']=data1['loan_help']+data1['housing_help']
test['loan_Code']=test['loan_help']+test['housing_help']
test.loan_Code = test.loan_Code.replace(2,1)
data1.loan_Code = data1.loan_Code.replace(2,1)
test=test[test.loan_Code != 10]
data1=data1[data1.loan_Code != 10]


# In[12]:


#completing custAge variable
sinmed1=data1.loc[data1['marital'] == 'single', 'custAge'].median()
sinmed2=test.loc[test['marital'] == 'single', 'custAge'].median()
marmed1=data1.loc[data1['marital'] == 'married', 'custAge'].median()
marmed2=test.loc[test['marital'] == 'married', 'custAge'].median()
divmed1=data1.loc[data1['marital'] == 'divorced', 'custAge'].median()
divmed2=test.loc[test['marital'] == 'divorced', 'custAge'].median()
mask = data1['custAge'].isnull()
data1.loc[mask, 'custAge'] = data1.loc[mask, 'marital'].map({'single':sinmed1, 'married':marmed1,'divorced':divmed1})
mask1 = test['custAge'].isnull()
test.loc[mask1, 'custAge'] = test.loc[mask1, 'marital'].map({'single':sinmed2, 'married':marmed2,'divorced':divmed2})


# In[16]:


#define y variable aka target/outcome
Target = ['responded_Code']
#define x variables for original features aka feature selection
data1_x = ['custAge', 'profession', 'marital', 'housing','loan', 'contact', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed'] #pretty name/values for charts
data1_x_calc = ['custAge','previous','emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed', 'poutcome_Code', 'contact_Code','profession_Code', 'marital_Code', 'loan_Code'] #coded for algorithm calculation
data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')
#define x variables for original w/bin features to remove continuous variables
data1_x_bin = ['poutcome_Code', 'contact_Code','profession_Code', 'marital_Code', 'loan_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')
#define x and y variables for dummy features original
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')
data1_dummy.head()
#split data
X_train=data1[data1_x_calc]
Y_train=data1[Target]
X_test=test[data1_x_calc]


# In[17]:


#model development
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[18]:


# evaluation
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred1 = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
indexR=1
results=pd.DataFrame(columns=['Count of Trees','Score for Training','Score for Testing'])
for sizeOfForest in range(1,102,10):
    forest=RandomForestClassifier(criterion='gini',n_estimators=sizeOfForest,random_state=6)
    forest.fit(X_train,Y_train)
    scoreTrain=forest.score(X_train,Y_train)
    scoreTest=forest.score(X_test,Y_pred1)
    results.loc[indexR]=[sizeOfForest,scoreTrain,scoreTest]
    indexR=indexR+1
feat_labels=data1_x_calc
importance=forest.feature_importances_
indices = np.argsort(importance)[::-1]
for i in range(X_train.shape[1]):
    print("(%2d)%-*s%f"%(i+1,30, feat_labels[i], importance[indices[i]]))


# In[19]:


# try to drop loan_Code variable to improve accuracy
drop_column = ['loan_Code']
X_train.drop(drop_column, axis=1, inplace = True)
X_test.drop(drop_column, axis=1, inplace = True)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
#evaluate model again
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[21]:


# submisson
submission=test
submission['responded']=Y_pred
filepath1 = "C:/Users/xuyuk/testingCandidate.csv"
submission.to_csv(filepath1, index= False)


# In[14]:





# In[ ]:




