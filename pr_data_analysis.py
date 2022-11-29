#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# configuring figure/plot params
custom_params = {'figure.figsize':(12,7)} 
sns.set_theme(style="whitegrid", rc=custom_params)


# In[40]:


mnist_data = pd.read_csv('mnist.csv').values
mnist_data.describe


# # Data Exploration

# In[4]:


labels = mnist_data[:,0] # 0 to 9
digits = mnist_data[:, 1:] # 42000 digits
img_size = 28

fig = plt.figure(figsize=(26,13))
columns = 5
rows = 2
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    random_index = random.randrange(0, len(digits))
    plt.imshow(digits[random_index].reshape(img_size, img_size))
    plt.xlabel(str(labels[random_index]))
plt.show()


# In[29]:


unique, counts = np.unique(labels, return_counts=True)
ax = sns.barplot(x=unique,y=counts)
plt.title("Class distribution for edited MNIST dataset")
plt.ylabel("Counts")
plt.xlabel("Numbers")
plt.show()


# ## Model Descriptions
# ### Compare ink vs ink + special feature
# - (Zero mean and SD=1) Multinomial Logit -> Ink Feature
# - (Zero mean and SD=1) MN Logit -> Ink Feature + Our own special feature
# 
# ### Compare both models
# - (Regularized?) MN Logit (w/ LASSO penalty) -> 784 features (all pixel values)
# - Support Vector Machines (SVM)

# In[33]:


# setting up pipeline to facilitate data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale



# # INK Feature

# In[35]:


# create ink feature
ink = np.array([sum(row) for row in digits])
# compute mean for each digit class
ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
# compute standard deviation for each digit class
ink_std = [np.std(ink[labels == i]) for i in range(10)]
print(np.size(ink), np.size(ink_mean), np.size(ink_std))


# In[34]:


scaled_ink = (ink - np.mean(ink))/np.std(ink)
print(scaled_ink)


# In[ ]:


logreg = sklearn.linear_model.LogisticRegression(penalty='l1', c=1,solver='saga')
# saga is the only solver that supports l1 penalty and multi-class problems
#arr = np.array([ink_mean, ink_std])
#print(pd.DataFrame(arr))
#print(df.loc[:, (df.sum() > 0).all()])


# In[ ]:


# special feature:
# go through each row of pixels, count how many times it reaches a non-zero pixel 
# ... with a zero ink pixel separator --> if highest value of this is 2, 
# feature is # of rows value 2 / # of rows value 1
# we do not care about rows with value 0 --> covered by ink feature

# possible issue: does not distinguish between 6 and 9 --> 
# but ink feature also does not distinguish between those in theory

# for part 2, when we need to consider each pixel as an individual feature
# find a way to remove all pixels that always have constant value

