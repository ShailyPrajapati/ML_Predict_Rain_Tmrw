# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 23:26:45 2023

@author: Ruhina
Final Term project
Predict if it will rain tomorrow
Dataset: 145461 x 23
"""
# Load data
import pandas as pd
import numpy as np
import statsmodels.tools.tools as stattools
#For plotting
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn import metrics
#For splitting the data
from sklearn.model_selection import train_test_split

#For data preprocessing
from sklearn.preprocessing import StandardScaler


#For appling LogisticRegression
from sklearn.linear_model import LogisticRegression
#Performance metrices
from sklearn.metrics import roc_auc_score,roc_curve,auc,log_loss,confusion_matrix

#For encoding the features
from sklearn.preprocessing import LabelEncoder,LabelBinarizer


#Import data
weather_full = pd.read_csv("C:/Users/sufiya/Documents/Ruhina/MSBA/QNT-730/Final term/Dataset/weatherAUS.csv")

#EAD
print("Total no.of points = {}".format(weather_full.shape[0]))
weather_full.head(5)

#check for duplicates
weather_full.drop_duplicates(inplace=True)

#check for nulls
weather_full.isnull().any()

# We can see there are many Null values in the data , lets try to fill with proper values.Check for balance data
sns.set(style="whitegrid")
sns.countplot(weather_full.RainTomorrow)
plt.title("Target labels")
plt.show()

# The data is imbalanced.

#Separating the data based on its class label.
data_yes = weather_full[weather_full['RainTomorrow']=='Yes']
data_no = weather_full[weather_full['RainTomorrow']=='No']

#Managing nulls
data_no['MinTemp'] = data_no['MinTemp'].replace(np.nan, data_no['MinTemp'].mode()[0])
data_no['MaxTemp'] = data_no['MaxTemp'].replace(np.nan, data_no['MaxTemp'].mode()[0])
data_no['Rainfall'] = data_no['Rainfall'].replace(np.nan, data_no['Rainfall'].mode()[0])
data_no['Evaporation'] = data_no['Evaporation'].replace(np.nan, data_no['Evaporation'].mode()[0])
data_no['Sunshine'] = data_no['Sunshine'].replace(np.nan, data_no['Sunshine'].mode()[0])
data_no['Pressure9am'] = data_no['Pressure9am'].replace(np.nan, data_no['Pressure9am'].mode()[0])
data_no['Pressure3pm'] = data_no['Pressure3pm'].replace(np.nan, data_no['Pressure3pm'].mode()[0])
data_no['Cloud9am'] = data_no['Cloud9am'].replace(np.nan, data_no['Cloud9am'].mode()[0])
data_no['Cloud3pm'] = data_no['Cloud3pm'].replace(np.nan, data_no['Cloud3pm'].mode()[0])
data_no['Temp9am'] = data_no['Temp9am'].replace(np.nan, data_no['Temp9am'].mode()[0])
data_no['Temp3pm'] = data_no['Temp3pm'].replace(np.nan, data_no['Temp3pm'].mode()[0])
data_no['MinTemp'] = data_no['MinTemp'].replace(np.nan, data_no['MinTemp'].mode()[0])
data_no['Humidity9am'] = data_no['Humidity9am'].replace(np.nan, data_no['Humidity9am'].mode()[0])
data_no['Humidity3pm'] = data_no['Humidity3pm'].replace(np.nan, data_no['Humidity3pm'].mode()[0])
data_no['WindGustSpeed'] = data_no['WindGustSpeed'].replace(np.nan, data_no['WindGustSpeed'].mode()[0])
data_no['WindSpeed9am'] = data_no['WindSpeed9am'].replace(np.nan, data_no['WindSpeed9am'].mode()[0])
data_no['WindSpeed3pm'] = data_no['WindSpeed3pm'].replace(np.nan, data_no['WindSpeed3pm'].mode()[0])

data_yes['MinTemp'] = data_yes['MinTemp'].replace(np.nan, data_yes['MinTemp'].mode()[0])
data_yes['MaxTemp'] = data_yes['MaxTemp'].replace(np.nan, data_yes['MaxTemp'].mode()[0])
data_yes['Rainfall'] = data_yes['Rainfall'].replace(np.nan, data_yes['Rainfall'].mode()[0])
data_yes['Evaporation'] = data_yes['Evaporation'].replace(np.nan, data_yes['Evaporation'].mode()[0])
data_yes['Sunshine'] = data_yes['Sunshine'].replace(np.nan, data_yes['Sunshine'].mode()[0])
data_yes['Pressure9am'] = data_yes['Pressure9am'].replace(np.nan, data_yes['Pressure9am'].median()[0])
data_yes['Pressure3pm'] = data_yes['Pressure3pm'].replace(np.nan, data_yes['Pressure3pm'].median()[0])
data_yes['Cloud9am'] = data_yes['Cloud9am'].replace(np.nan, data_yes['Cloud9am'].mode()[0])
data_yes['Cloud3pm'] = data_yes['Cloud3pm'].replace(np.nan, data_yes['Cloud3pm'].mode()[0])
data_yes['Temp9am'] = data_yes['Temp9am'].replace(np.nan, data_yes['Temp9am'].mode()[0])
data_yes['Temp3pm'] = data_yes['Temp3pm'].replace(np.nan, data_yes['Temp3pm'].mode()[0])
data_yes['MinTemp'] = data_yes['MinTemp'].replace(np.nan, data_yes['MinTemp'].mode()[0])
data_yes['Humidity9am'] = data_yes['Humidity9am'].replace(np.nan, data_yes['Humidity9am'].mode()[0])
data_yes['Humidity3pm'] = data_yes['Humidity3pm'].replace(np.nan, data_yes['Humidity3pm'].mode()[0])
data_yes['WindGustSpeed'] = data_yes['WindGustSpeed'].replace(np.nan, data_yes['WindGustSpeed'].median()[0])
data_yes['WindSpeed9am'] = data_yes['WindSpeed9am'].replace(np.nan, data_yes['WindSpeed9am'].median()[0])
data_yes['WindSpeed3pm'] = data_yes['WindSpeed3pm'].replace(np.nan, data_yes['WindSpeed3pm'].median()[0])

# finally merge the two datasets
data_final = data_yes.append(data_no, ignore_index=True)

data_final.drop(['Date', 'Location'], axis=1, inplace=True)

# For RainToday feature we cannot fill any value, so better to remove the NaN values 
# For RainToday feature we cannot fill any value, so better to remove the NaN values 
data_final = data_final.dropna()
#check for nulls
data_final.isnull().any()
data_final.shape

#box plots

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_final[['MinTemp','MaxTemp','Temp9am','Temp3pm']])

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_final[['WindGustSpeed','WindSpeed9am','WindSpeed3pm']])

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_final[['Humidity9am','Humidity3pm']])

data_final.shape
data_final= data_final[data_final['Humidity3pm']!=0.0]
data_final= data_final[data_final['Humidity9am']!=0.0]
data_final.shape

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_final[['Humidity9am','Humidity3pm']])

sns.set(style="whitegrid")
plt.figure(figsize=(5, 6))
sns.boxplot(data=data_final[['Evaporation','Sunshine']])

sns.set(style="whitegrid")
plt.figure(figsize=(5, 6))
sns.boxplot(data=data_final[['Rainfall']])

# Label Encoding for categorical values

WindGustDir_encode = LabelEncoder()
data_final['WindGustDir']=WindGustDir_encode.fit_transform(data_final['WindGustDir'])

WindDir9am_encode = LabelEncoder()
data_final['WindDir9am']=WindDir9am_encode.fit_transform(data_final['WindDir9am'])

WindDir3pm_encode = LabelEncoder()
data_final['WindDir3pm']=WindDir3pm_encode.fit_transform(data_final['WindDir3pm'])

RainToday_encode = LabelEncoder()
data_final['RainToday']=RainToday_encode.fit_transform(data_final['RainToday'])

RainTomorrow_encode = LabelEncoder()
data_final['RainTomorrow']=RainTomorrow_encode.fit_transform(data_final["RainTomorrow"])

# seperating predictors and class variable
data_final_Y= data_final['RainTomorrow']
data_final_X = data_final.drop(['RainTomorrow'],axis=1)


X_train, X_test, y_train, y_test = train_test_split(data_final_X, data_final_Y, train_size=0.70,random_state = 6)

#Feature scaling
scaler= StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating our model named 'lr'
logR_model_01 = LogisticRegression()

# Training it by using our train data:
logR_model_01.fit(X_train, y_train)

# Predicting values on test data
predictions = logR_model_01.predict(X_test)

# compare with statsmodels
import statsmodels.api as sm
sm_model = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=0)
print(sm_model.pvalues)
sm_model.summary()

# Model accuracy
# Use score method to get accuracy of model
score = logR_model_01.score(X_test, y_test)
print('Testaccuracy of sklearn logistic regression library: {}%'.format(score *100))

#ROC curve
#define metrics

y_pred_proba = logR_model_01.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

#Feature interpretation
feature_weights=sorted(zip(logR_model_01.coef_[0],data_final_X.columns),reverse = True)