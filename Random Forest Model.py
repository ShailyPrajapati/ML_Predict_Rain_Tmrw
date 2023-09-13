#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import plotly.express as px 
import plotly.graph_objects as go 
import graphviz


# In[33]:



df=pd.read_csv('weatherAUS.csv', encoding='utf-8')


# In[34]:



df=df[pd.isnull(df['RainTomorrow'])==False]


df=df.fillna(df.mean())


# In[35]:



df['RainTodayFlag']=df['RainToday'].apply(lambda x: 1 if x=='Yes' else 0)
df['RainTomorrowFlag']=df['RainTomorrow'].apply(lambda x: 1 if x=='Yes' else 0)


# In[36]:



df.shape


# In[37]:



X=df[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 
      'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',  
      'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainTodayFlag']]
y=df['RainTomorrowFlag'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[38]:


X_train.shape


# In[40]:



model = RandomForestClassifier(criterion='gini', 
                               bootstrap=True, # enabling bootstrapping
                               random_state=0, # random state for reproducibility
                               max_features='sqrt', # number of random features to use sqrt(n_features)
                               min_samples_leaf=1000, # minimum no of observarions allowed in a leaf
                               max_depth=4, # maximum depth of the tree
                               n_estimators=1000 # how many trees to build
                              )


# In[42]:



clf = model.fit(X_train, y_train)


# In[43]:



pred_labels_tr = model.predict(X_train)

pred_labels_te = model.predict(X_test)


# In[47]:



print('Model Summary')

print("")
print("Evaluation on Test Data")
score_te = model.score(X_test, y_test)
print('Accuracy Score: ', score_te)

print(classification_report(y_test, pred_labels_te))
print("")
print('Evaluation on Training Data')
score_tr = model.score(X_train, y_train)
print('Accuracy Score: ', score_tr)

print(classification_report(y_train, pred_labels_tr))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




