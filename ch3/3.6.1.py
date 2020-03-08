#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist


# In[2]:


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)


# In[3]:


print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)


# In[ ]:





# In[ ]:




