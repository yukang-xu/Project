#!/usr/bin/env python
# coding: utf-8

# In[2]:


####### Identify and Clean data###########
##########################################
import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
filepath1 = r"C:\Users\xuyuk\OneDrive - Georgia State University\Data import\train.csv"
train = pd.read_csv(filepath1)
filepath2 = r"C:\Users\xuyuk\OneDrive - Georgia State University\Data import\test.csv"
test = pd.read_csv(filepath2
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]


# In[3]:


########### data preparation ##########
#######################################
# tokenize our data
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
#  filling the shortfall by zeros and triming the longer ones to the same length(maxlen) as the short ones
maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
# draw a graph and pick maxlen number
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
plt.show()


# In[4]:


######## model development and evaluation ##########
#####################################
# We begin our defining an Input layer that accepts a list of sentences that has a dimension of 200
inp = Input(shape=(maxlen, ))
# Embedding allows us to reduce model size and most importantly the huge dimensions we have to deal with
embed_size = 128
x = Embedding(max_features, embed_size)(inp)
# we feed this Tensor into the LSTM layer
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
# reshape the 3D tensor into a 2D one
x = GlobalMaxPool1D()(x)
# connect the output of drop out layer to a densely connected layer
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
# squash the output between the bounds of 0 and 1.
x = Dense(6, activation="sigmoid")(x)
# configure the learning process
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
#  put our model to the test
batch_size = 32
epochs = 2
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.summary()


# In[ ]:




