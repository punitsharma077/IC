#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import keras 
import re
import string
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add


# In[2]:


model=load_model("model_9.h5")


# In[3]:


model_temp = ResNet50(weights="imagenet", input_shape=(224,224,3))


# In[4]:


#this model is part of keras functional api
model_resnet = Model(model_temp.input,model_temp.layers[-2].output)


# In[5]:


#for image to be fed to Resnet
def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    #normalization
    img = preprocess_input(img)
    return img
def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1])
    return feature_vector


# In[6]:




# In[7]:




# In[11]:


with open("word_to_idx.pkl",'rb') as w2i:
    word_to_idx = pickle.load(w2i)
with open("idx_to_word.pkl",'rb') as i2w:
    idx_to_word = pickle.load(i2w)
    


# In[12]:




# In[13]:




# In[14]:


""""
word_to_idx = {}
idx_to_word = {}
total_words=[]

descriptions = None
with open("descriptions.txt","r") as desc:
    descriptions = desc.read()
descriptions = json.loads( descriptions.replace("'", "\""))    #replacing single quotes with backslash
for key in descriptions.keys():
    [total_words.append(i) for des in descriptions[key] for i in des.split()]
print("Total words %d"%len(total_words))

for i,word in enumerate(total_words):
    word_to_idx[word] = i+1
    idx_to_word[i+1] = word
    """


# In[92]:


# Two special words which we need to add 
"""
idx_to_word[1846] = 'startseq'
word_to_idx['startseq']=1846

idx_to_word[1847] = 'endseq'
word_to_idx['endseq']=1847

vocab_size = len(word_to_idx) + 1
print("vocab size",vocab_size)"""


# In[15]:


max_len=35
def predict_caption(photo):
    in_text = "startseq"
    
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word
        
        if word =='endseq':
            break
        
        
    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption


# In[16]:




# In[17]:




# In[18]:


def caption_this_image(image):
  enc = encode_image(image)
  caption = predict_caption(enc)
  return caption


# In[ ]:




