#!/usr/bin/env python
# coding: utf-8

# # shivani tiwari 03

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[5]:


df=pd.read_csv('D:/shivani tiwari/bml/placement.csv')


# In[6]:


df.head()
plt.scatter(df['cgpa'],df['package'])


# In[17]:


import seaborn as sns
corr=df.corr()


# In[18]:


sns.heatmap(corr,annot=True)


# In[19]:


x=df['cgpa']
y=df['package']


# In[24]:


x=df.iloc[:,0:1]
y=df.iloc[:,-1]


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=2)


# In[26]:


from sklearn.linear_model import LinearRegression


# In[27]:


lr= LinearRegression()


# In[29]:


lr.fit(x_train,y_train)
lr.predict(x_test.iloc[0].values.reshape(1,1))


# In[30]:


plt.scatter(df['cgpa'],df['package'])
plt.xlabel('cgpa')
plt.ylabel('package')


# In[31]:


m=lr.coef_
m


# In[32]:


b=lr.intercept_
b


# In[33]:


#y=mx+b
m*8.58+b


# In[34]:


m*9.5+b


# In[35]:


m*100+b


# In[ ]:




