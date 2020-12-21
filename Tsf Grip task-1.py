#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:



url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")

df.head()


# In[3]:


df.isnull().sum()


# In[4]:


df.plot(x = 'Hours', y = 'Scores', style = 'o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[5]:


X = df.iloc[:, :-1].values  
Y = df.iloc[:, 1].values


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)


# In[7]:


# X = X.reshape((1,-1))
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, Y_train)

print("Training complete.")


# In[8]:


line =regressor.coef_*X+regressor.intercept_
print(line)


# In[9]:


plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


# In[10]:


print(X_test)                               # Testing data - In Hours
y_pred = regressor.predict(X_test)          # Predicting the scores


# In[11]:


df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})  
df


# In[12]:


hours = np.array(9.25).reshape(1, -1)
own_predt = regressor.predict(hours)
print('If the student reads for %0.3f hours then he will score %0.3f'%(9.25, own_predt[0]))


# In[13]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))


# In[ ]:




