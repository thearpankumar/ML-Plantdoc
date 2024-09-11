#!/usr/bin/env python
# coding: utf-8

# # Plant Disease Prediction

# ## Importing Dataset

# Dataset Link: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

# ## Importing libraries

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ## Data Preprocessing

# ### Training Image preprocessing

# In[8]:


training_set = tf.keras.utils.image_dataset_from_directory(
    "Plantdataset/train",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)


# ### Validation Image Preprocessing

# In[9]:


validation_set = tf.keras.utils.image_dataset_from_directory(
    'Plantdataset/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)


# #### To avoid Overshooting Loss function
# 1. Choose small learning rate default 0.001 here we have taken 0.0001
# 2. There may be chance of underfitting so increase number of neuron
# 3. Add more Convolutional Layer to extract more feature from images there may be possibilty that model unable to capture relevant feature or model is confusing due to lack of feature so feed with more feature

# ## Building Model

# In[10]:


cnn = tf.keras.models.Sequential()


# ### Building Convolution Layer

# In[11]:


cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[12]:


cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[13]:


cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[14]:


cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[15]:


cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[16]:


cnn.add(tf.keras.layers.Dropout(0.25))


# In[17]:


cnn.add(tf.keras.layers.Flatten())


# In[18]:


cnn.add(tf.keras.layers.Dense(units=1500,activation='relu'))


# In[19]:


cnn.add(tf.keras.layers.Dropout(0.4)) #To avoid overfitting


# In[20]:


#Output Layer
cnn.add(tf.keras.layers.Dense(units=38,activation='softmax'))


# ### Compiling and Training Phase

# In[21]:


cnn.compile(optimizer=tf.keras.optimizers.legacy.Adam(
    learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[22]:


cnn.summary()


# In[23]:


training_history = cnn.fit(x=training_set,validation_data=validation_set,epochs=10)


# ## Evaluating Model

# In[24]:


#Training set Accuracy
train_loss, train_acc = cnn.evaluate(training_set)
print('Training accuracy:', train_acc)


# In[25]:


#Validation set Accuracy
val_loss, val_acc = cnn.evaluate(validation_set)
print('Validation accuracy:', val_acc)


# ### Saving Model

# In[26]:


cnn.save('trained_plant_disease_model.keras')


# In[27]:


training_history.history #Return Dictionary of history


# In[28]:


#Recording History in json
import json
with open('training_hist.json','w') as f:
  json.dump(training_history.history,f)


# In[29]:


print(training_history.history.keys())


# ## Accuracy Visualization

# In[30]:


epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='blue',label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Visualization of Accuracy Result')
plt.legend()
plt.show()


# In[ ]:





# ## Some other metrics for model evaluation

# In[31]:


class_name = validation_set.class_names


# In[33]:


test_set = tf.keras.utils.image_dataset_from_directory(
    'Plantdataset/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=1,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)


# In[ ]:





# In[34]:


y_pred = cnn.predict(test_set)
predicted_categories = tf.argmax(y_pred, axis=1)


# In[35]:


true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1)


# In[36]:


Y_true


# In[37]:


predicted_categories


# In[ ]:





# In[38]:


from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(Y_true,predicted_categories)


# In[39]:


# Precision Recall Fscore
print(classification_report(Y_true,predicted_categories,target_names=class_name))


# ### Confusion Matrix Visualization

# In[40]:


plt.figure(figsize=(40, 40))
sns.heatmap(cm,annot=True,annot_kws={"size": 10})

plt.xlabel('Predicted Class',fontsize = 20)
plt.ylabel('Actual Class',fontsize = 20)
plt.title('Plant Disease Prediction Confusion Matrix',fontsize = 25)
plt.show()


# In[ ]:




