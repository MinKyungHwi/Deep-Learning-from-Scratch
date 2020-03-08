#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist


# In[7]:


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True ,one_hot_label=False)
    return x_test, t_test


# In[8]:


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        
    return network


# In[9]:


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    
    return y


# In[10]:


x, t =get_data()
network = init_network()


# In[ ]:




