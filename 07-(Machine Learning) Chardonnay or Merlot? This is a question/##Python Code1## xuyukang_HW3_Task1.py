#!/usr/bin/env python
# coding: utf-8

# In[152]:


#open and read data
import numpy as np
import pandas as pd
filepath = r"C:\Users\xuyuk\wineData.csv"
winedata = pd.read_csv(filepath)
num_cols = winedata._get_numeric_data().columns
num_cols
print(winedata.dtypes)
print(winedata.head())


# In[72]:


##identify class attribute and perform class mapping
winedata["Class"].value_counts().plot(kind = "bar")
print(winedata["Class"].value_counts())
col=winedata.pop('Class')
winedata['Class']=col
class_mapping={label:idx for idx,label in enumerate(np.unique(winedata['Class']))}
print(class_mapping)
winedata['Class']=winedata['Class'].map(class_mapping)


# In[153]:


#normalize by min-max normalization in range (0,3)
from sklearn import preprocessing
import numpy as np
a=winedata[['Class']]
del winedata['Class']
cols = winedata.columns
df = pd.DataFrame(winedata)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 3))
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled, columns = cols)
winedata = pd.concat([a, df_normalized], axis=1)


# In[155]:


#export the final data
export_csv = winedata.to_csv (r'C:\Users\xuyuk\.ipython\newwinedata.csv', index = None, header=True) 


# In[147]:


df_normalized


# In[ ]:




