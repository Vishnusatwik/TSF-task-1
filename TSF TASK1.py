#!/usr/bin/env python
# coding: utf-8

# # # Python code for predicting Marks of a Student
# 

# 
# # ## Made By : K.VISHNU SATWIK

# # ### Importing the Libraries
# 

# In[3]:



import pandas as pd


# In[4]:


import numpy as np


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


url="http://bit.ly/w-data"


# In[8]:


s_data=pd.read_csv(url)


# In[9]:


print("Data imported successfully")


# In[10]:


s_data.head(10)


# In[11]:


s_data.plot(x='Hours',y='Scores',style='o')


# In[12]:


plt.title('Hours vs Percentage')


# In[13]:


plt.xlabel('Hours studied')


# In[14]:


plt.ylabel('Pecentage score')


# In[16]:


plt.show()


# In[17]:


X=s_data.iloc[:, :-1].values


# In[18]:


y=s_data.iloc[:, 1].values


# In[20]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[21]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print("Training Complete.")


# In[22]:


line = regressor.coef_*X+regressor.intercept_
plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# In[23]:


print(X_test)
y_pred=regressor.predict(X_test)


# In[24]:


df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df


# In[25]:


hours = 9.25
#X_pred = regressor.predict([hours])
print("No of Hours = {}".format(hours))
print("Predicted Score = ",regressor.predict([[hours]]))


# In[26]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

