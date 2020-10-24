#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd


# In[33]:


df=pd.read_csv('heart.csv')


# In[34]:


df=df.drop(['slope','oldpeak','slope','ca','thal'],axis=1)


# # Training

# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X = df.drop("target",axis=1).values
y = df["target"].values


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[38]:


from sklearn.preprocessing import StandardScaler


# In[39]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# # Training using various models

# # Neural Network

# In[40]:


from sklearn.metrics import accuracy_score


# In[41]:


from keras.models import Sequential
from keras.layers import Dense


# In[42]:


model = Sequential()
model.add(Dense(20,activation='relu'))

model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[43]:


model.fit(X_train,y_train,epochs=100)


# In[44]:


pred = model.predict_classes(X_test)


# # Logistic Regression

# In[45]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

Y_pred_lr = lr.predict(X_test)


# Saving the models

# In[46]:


import pickle


# In[47]:


filename = 'Regression.sav'
pickle.dump(lr, open(filename, 'wb'))


# In[ ]:





# In[48]:


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


# In[49]:


model.save("model.h5")


# In[ ]:





# In[ ]:




