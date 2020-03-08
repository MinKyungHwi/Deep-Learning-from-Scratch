#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys, os
sys.path.append(os.pardir)
import numpy as np


# In[7]:


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)


# In[8]:


img = x_train[1]
label = t_train[1]
print(label)


# In[9]:


print(img.shape)
img =img.reshape(28,28)
print(img.shape)


# In[10]:


type(x_train)


# In[ ]:


img_show(img)


# In[ ]:




