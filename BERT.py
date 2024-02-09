#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf


# In[27]:


pip install pandas


# In[2]:


import os


# In[3]:


import pandas as pd
import tensorflow_hub as hub


# In[4]:


import tensorflow_text as text


# In[5]:


male=[]
 for filename in os.listdir("C:\\Users\\cse lab2\\Desktop\\gendermale"):
    with open(os.path.join("C:\\Users\\cse lab2\\Desktop\\gendermale",filename),'r') as f:
        text=f.read()
        male.append(text)


# In[ ]:





# In[6]:


male=pd.DataFrame(male)


# In[7]:


female=[]
for filename in os.listdir("C:\\Users\\cse lab2\\Desktop\\gender female"):
    with open(os.path.join("C:\\Users\cse lab2\\Desktop\\gender female",filename),'r') as f:
        text=f.read()
        female.append(text)


# In[8]:


female=pd.DataFrame(female)


# In[9]:


female


# In[10]:


female.rename(columns = {0:'Reviews'}, inplace = True)


# In[11]:


female


# In[12]:


male.rename(columns = {0:'Reviews'}, inplace = True)


# In[13]:


male


# In[14]:


female['lables']=0
female['category']='female'


# In[15]:


male['lables']=1
male['category']='male'


# In[16]:


df_balanced=pd.concat([male,female],ignore_index=True)


# In[17]:


df_balanced


# In[18]:


!pip install sklearn


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df_balanced['Reviews'],df_balanced['lables'],stratify=df_balanced['lables'])


# In[20]:


bert_encoder=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# In[21]:


bert_preprocesser=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")


# In[22]:


#BERT 


# In[23]:


text_input=tf.keras.layers.Input(shape=(),dtype=tf.string,name="text")


# In[24]:


#bert layers`
preprocessed_text=bert_preprocesser(text_input)
outputs=bert_encoder(preprocessed_text)


# In[25]:


#neural network layers
l=tf.keras.layers.Dropout(0.1,name='dropout')(outputs['pooled_output'])
l=tf.keras.layers.Dense(1,activation='sigmoid',name='output')(l)


# In[26]:


#construct final model
model=tf.keras.Model(inputs=[text_input],outputs=[l])


# In[27]:


METRICS=[
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)


# In[2]:


model.fit(x_train,y_train,epochs=25)


# In[32]:


y_predicted = model.predict(x_test)
y_predicted=y_predicted.flatten()


# In[33]:


import numpy as np
y_predicted=np.where(y_predicted>0.5,1,0)
y_predicted


# In[34]:


from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_predicted)
cm


# In[35]:



print(classification_report(y_test,y_predicted))


# In[36]:


reviews=["Terrific Value The Excelsior isn't fancy but has a lot going for it. It is accross the street from the main train station in Frankfurt and has great breakfasts. The staff was pleasant and gave us a break by letting us into our room well before check-in. The rooms were clean and the price was affordable.","Phew!!!!! lbertsons are a 3min drive away. Can't think of anything the supermarket didn't sell, apart from proper baked beans, but then you can never get them in the states!We ate at The Keg, both the steak the salmon was delicious. We also tried The Bamboo Club - a wide variety of asian dishes on offer. Did not eat at the JW but Roy's looked superb.Very,very hot the entire week but you know before you go so be prepared. Will not be returning - there are too many places out there still to visit, but would recommend both the MCV Arizona."]


# In[37]:


model.predict(reviews)


# In[38]:


reviews[1]






