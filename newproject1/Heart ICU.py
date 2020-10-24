#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# In[3]:


df=pd.read_csv('heart.csv')


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X = df.drop("target",axis=1).values
y = df["target"].values


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[12]:


X_train.shape


# In[13]:


X_test.shape


# # Training using various models

# # Neural Network

# In[14]:



from sklearn.metrics import accuracy_score


# In[15]:


from keras.models import Sequential
from keras.layers import Dense


# In[16]:


model = Sequential()
model.add(Dense(20,activation='relu'))

model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[17]:


model.fit(X_train,y_train,epochs=100)


# In[18]:


pred = model.predict_classes(X_test)


# In[19]:


from sklearn.metrics import classification_report, confusion_matrix


# In[20]:


print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))


# In[ ]:





# # Logistic Regression

# In[22]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

Y_pred_lr = lr.predict(X_test)


# In[23]:


print(classification_report(y_test,Y_pred_lr))
print('\n')
print(confusion_matrix(y_test,Y_pred_lr))


# Saving the models

# In[24]:


import pickle


# In[27]:


filename = 'Regression.sav'
pickle.dump(lr, open(filename, 'wb'))


# In[ ]:





# In[28]:


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


# In[34]:


model.save("model.h5")


# In[ ]:




