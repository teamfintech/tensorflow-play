import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from tensorflow.keras.datasets import mnist


# In[4]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[5]:


#x_train.shape


# In[6]:


#plt.imshow(x_train[0])


# In[8]:


from tensorflow.keras.utils import to_categorical


# In[10]:


y_cat_train = to_categorical(y_train)


# In[11]:


y_cat_test = to_categorical(y_test)


# In[12]:


y_cat_train[0]


# In[13]:


x_train = x_train/255


# In[14]:


x_test = x_test/255


# In[15]:


x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(10000,28,28,1)


# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# In[18]:


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28,28,1), activation="relu"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[19]:


from tensorflow.keras.callbacks import EarlyStopping


# In[20]:


early_stop = EarlyStopping(monitor="val_accuracy", patience=1)


# In[21]:


model.fit(x_train, y_cat_train, epochs=1000, validation_data=(x_test, y_cat_test), callbacks=[early_stop])


# In[22]:


metrics = pd.DataFrame(model.history.history)


# In[23]:





# In[24]:


metrics[["loss", "val_loss"]].plot()


# In[25]:


metrics[["accuracy", "val_accuracy"]].plot()


# In[26]:


from sklearn.metrics import classification_report, confusion_matrix


# In[27]:


predictions = model.predict_classes(x_test)


# In[28]:





# In[29]:


print(classification_report(y_test, predictions))


# In[31]:


confusion_matrix(y_test, predictions)


# In[39]:


plt.figure(figsize=(15,10))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True)


# In[42]:


model.predict_classes(x_test[0].reshape(1, 28, 28, 1))