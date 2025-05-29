#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import mnist


# In[2]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[3]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[4]:


x_train=x_train.reshape(60000,28*28)
x_test=x_test.reshape(10000,28*28)
x_train=x_train.astype('float64')
x_test=x_test.astype('float64')



# In[5]:


x_train=x_train/255
x_test=x_test/255


# In[6]:


print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')


# In[7]:


print(x_train.shape[1],'train samples')
print(x_test.shape[1],'test samples')


# In[8]:


y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)


# In[9]:


print(y_train.shape)
print(y_test.shape)


# In[10]:


x_train[3]


# In[11]:


y_train[3]


# In[12]:


print('label',y_train[456])
plt.imshow(x_train[456].reshape(28,28), cmap='gray')
plt.show()


# In[13]:


model = Sequential()


# In[14]:


model.add(Dense(128,activation='relu',input_shape=(28*28,)))
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(32,activation='sigmoid'))
model.add(Dense(10,activation='softmax'))


# In[15]:


model


# In[16]:


model.summary()


# In[17]:


w=[]
for i in model.layers:
    weigth=i.get_weights()
    w.append(weigth)
    


# # (input_size, output_size)
# ## for normal dense layer

# In[18]:


layer1=np.array(w[0][0])
print('shape of first layer',layer1.shape)


# # (kernel_height, kernel_width, input_channels, output_channels) 
# ## for cnn
# 

# In[19]:


fig=plt.figure(figsize=(14,12))
col=8
row=int(128/col)
for i in range(1,col*row+1):
    fig.add_subplot(row,col,i)
    plt.imshow(layer1[:,i-1].reshape(28,28),cmap='gray')
plt.show()


# In[20]:


model.compile(metrics=['accuracy'],optimizer='adam',loss='categorical_crossentropy')


# In[24]:


model.fit(x_train,y_train,epochs=5,batch_size=128)


# In[25]:


score=model.evaluate(x_test,y_test)


# In[27]:


print('accuracy',score[1])
print('loss',score[0])


# In[28]:


w=[]
for i in model.layers:
    weigth=i.get_weights()
    w.append(weigth)
layer1=np.array(w[0][0])
fig=plt.figure(figsize=(14,12))
col=8
row=int(128/col)
for i in range(1,col*row+1):
    fig.add_subplot(row,col,i)
    plt.imshow(layer1[:,i-1].reshape(28,28),cmap='gray')
plt.show()


# In[29]:


print('label',y_test[88])
plt.imshow(x_test[88:89].reshape(28,28), cmap='gray')
plt.show()


# In[31]:


prediction = model.predict(x_test[88:89])
prediction = prediction[0]
print('prediction',prediction)
print('threshold op',(prediction>.5)*1)


# In[34]:


print('label',y_test[3])
plt.imshow(x_test[3].reshape(28,28))


# In[38]:


model.predict(x_test[1])


# In[ ]:




