
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('heart.csv')

from sklearn.model_selection import train_test_split


# In[8]:


X = df.drop("target",axis=1).values
y = df["target"].values


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

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

