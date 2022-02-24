#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries :

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Reading the data :

iris = pd.read_csv("Iris.csv")
iris


#                                             Exploratory Data Analysis

# In[3]:


# Getting info of data :

iris.info()


# In[4]:


print(iris.describe())


# In[5]:


# Dropping unwanted columns :

iris.drop('Id', axis=1, inplace=True)


# In[6]:


# Getting column labels :

iris.columns


# In[7]:


# Fetching unique value from species column :

iris['Species'].unique()


# In[8]:


iris['Species'].value_counts()


# In[9]:


# Checking Null Values :

iris.isnull().sum()


# In[10]:


# Checking zero values :

print((iris==0).sum())


# In[11]:


print("*** obvious errors ***")
print(iris.groupby(['Species']).count())
print("")


# In[12]:


print("*****VARIANCE******")
print(iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].var())


# In[13]:


print("*****STANDARD DEVIATION******")
print(iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].std())


# In[14]:


import utils
print('\n*** Outlier Index ***')
print(utils.OutlierIndex(iris))  


# In[15]:


print('\n*** Outlier Values ***')
print(utils.OutlierValues(iris))  


# In[ ]:





#                                                     Visual Data Analysis

# In[16]:


plt.figure(figsize=(6,8))
sns.lmplot(x='SepalLengthCm',y='SepalWidthCm', data= iris, fit_reg=False , hue='Species' , legend=False, palette ='Set2')
plt.xlabel("Sepal Length (in cm)")
plt.ylabel("Sepal Width (in cm)")
plt.title("-------------------------------------------------------------------------------------------------- \n Comparison of Sepal Length and Sepal Width based on Species type \n --------------------------------------------------------------------------------------------------")
plt.legend(bbox_to_anchor=(1.02,0.15), loc ='center left' , title = "Species")
plt.show()


# In[17]:


plt.figure(figsize=(6,8))
sns.lmplot(x='PetalLengthCm',y='PetalWidthCm', data= iris, fit_reg=False , hue='Species' , legend=False, palette ='Set2')
plt.xlabel("Petal Length (in cm)")
plt.ylabel("Petal Width (in cm)")
plt.title("-------------------------------------------------------------------------------------------------- \n Comparison of Petal Length and Petal Width based on Species type \n --------------------------------------------------------------------------------------------------")
plt.legend(bbox_to_anchor=(1.02,0.15), loc ='center left' , title = "Species")
plt.show()


# In[18]:


plt.figure(figsize=(8,6))
sns.stripplot(y ='SepalLengthCm', x='Species', data= iris, palette='Set1' )
plt.ylabel(" Sepal Length (in cmn)")
plt.title("Sepal Length based on Species ")


# In[19]:


plt.figure(figsize=(8,6))
sns.stripplot(y ='SepalWidthCm', x='Species', data= iris, palette='Set1' )
plt.ylabel(" Sepal Width (in cmn)")
plt.title("Sepal Width based on Species ")


# In[20]:


plt.figure(figsize=(8,6))
sns.stripplot(y ='PetalLengthCm', x='Species', data= iris, palette='Set1' )
plt.ylabel(" Petal Length (in cmn)")
plt.title("Petal Length based on Species ")


# In[21]:


plt.figure(figsize=(8,6))
sns.stripplot(y ='PetalWidthCm', x='Species', data= iris, palette='Set1' )
plt.ylabel(" Petal Width (in cmn)")
plt.title("Petal Width based on Species ")


# In[22]:


iris.hist(bins=20,figsize=(12,8), color='skyblue')


# In[23]:


sns.pairplot(data= iris, hue='Species' ,palette='Set2' ,height=4)


# In[24]:


iris.corr()


# In[25]:


plt.figure(figsize=(7,5))
sns.heatmap( iris.corr(), annot=True , cmap ="PiYG")


# In[26]:


iris.groupby('Species').agg(['mean','median'])


# In[27]:


iris['Species']=iris['Species'].map({'Iris-setosa':0 , 'Iris-versicolor':1, 'Iris-virginica' :2 })


#                                       Machine Learning Algorithms

# In[28]:


# Splitting the dataset :

x = iris.iloc[: ,0:-1]
y = iris.iloc[:,-1]


# In[29]:


x    # Independent Variables


# In[30]:


y     # Dependent Variable


# In[31]:


# Dividing into train and test data :

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest =train_test_split( x, y, test_size=0.3 )


# In[32]:


xtrain.shape,xtest.shape,ytrain.shape,ytest.shape


#                                                  Logistic Regression

# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lr_model= LogisticRegression()


# In[34]:


lr_model.fit(xtrain,ytrain)


# In[35]:


test_predict= lr_model.predict(xtest)


# In[36]:


result = lr_model.predict([[5.1,3.5,1.4,0.2]])


# In[37]:


print(result)


# In[38]:


accuracy_a= lr_model.score(xtest,ytest)
print(accuracy_a * 100 ,"%")


# In[39]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_predict,ytest)
print(accuracy * 100 ,"%")


# In[40]:


from sklearn.metrics import classification_report
print(classification_report(test_predict,ytest))


#                                              Random Forest Classifier

# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

rf_model= RandomForestClassifier(n_estimators = 100)


# In[42]:


rf_model.fit(xtrain,ytrain)


# In[43]:


rf_test_predict= rf_model.predict(xtest)


# In[44]:


result = rf_model.predict([[5.1,3.5,1.4,0.2]])


# In[45]:


result


# In[46]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(rf_test_predict,ytest)
print(accuracy * 100 ,"%")


#                                                 PICKLE FILE

# In[47]:


#
# import pickle
# pickle.dump(lr_model,open('iris.pkl','wb'))


# In[ ]:




