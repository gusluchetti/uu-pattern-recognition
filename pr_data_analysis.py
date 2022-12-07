#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip install opencv-python')


# In[2]:


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


# In[3]:


# configuring figure/plot params
custom_params = {'figure.figsize':(17,9)} 
sns.set_theme(style="whitegrid", rc=custom_params)


# In[4]:


# loading dataset (it's a bit slow)
mnist_data = pd.read_csv('mnist.csv').values


# In[39]:


labels = mnist_data[:, 0]  # 0 to 9
digits = mnist_data[:, 1:] # 42000 digits
print(labels, digits)
mnist_df = pd.DataFrame(mnist_data)
mnist_df.head()
# first column is label, all (784) other columns are pixel values (0-255)


# In[49]:


# randomly print one number from each class
def print_one_from_each(digits, img_size):
    fig = plt.figure(figsize=(26,13))
    cols, rows = 5, 2
    for i in range(1, (cols*rows)+1):
        fig.add_subplot(rows, cols, i)
        index = np.where(labels == i-1)[0][0]
        plt.imshow(digits[index].reshape(img_size, img_size))
        plt.xlabel(str(labels[index]))
    plt.show()


# # Data Exploration / Pre-Processing
# Exploring data and plotting cool stuff

# In[9]:


unique, counts = np.unique(labels, return_counts=True)
ax = sns.barplot(x=unique,y=counts)
ax.bar_label(ax.containers[0])
plt.title("Class distribution for edited MNIST dataset")
plt.ylabel("Counts")
plt.xlabel("Numbers")
plt.show()


# In[10]:


digitsResized = np.zeros((len(digits), 14*14))

for i, d in enumerate(digits):
    _d = np.reshape(d, (28, 28)).astype('float32')
    resized_d = cv2.resize(_d, (14, 14))
    d_ = np.reshape(resized_d, (1, 14*14))
    digitsResized[i] = d_

# print('Digits resized', np.shape(digitsResized))


# In[50]:


print_one_from_each(digits, 28)


# In[52]:


# visually sampling the resized data
print_one_from_each(digitsResized, 14)


# #### Drop useless features
# 
# Useless features are those with constant values across all data points, hence cannot be used to distinguish between data.

# In[12]:


def filterConstantFeature(matrix, idx):
    return False if np.var(matrix[:, idx]) == 0.0 else True


cols_digits = list(range(0, len(digits[0])))
usefulCols_digits = [filterConstantFeature(digits, i) for i in cols_digits]
digitsFiltered = digits[:, usefulCols_digits]

cols_digitsResized = list(range(0, len(digitsResized[0])))
usefulCols_digitsResized = [filterConstantFeature(digitsResized, i) for i in cols_digitsResized]
digitsResizedFiltered = digitsResized[:, usefulCols_digitsResized]     
# DATA: digits -> resized to 14x14 -> dropped constant features.

# print(np.shape(digitsResized))
# print(np.shape(digitsResizedFiltered))


# In[53]:


# given some restriction on later parts of the assignment,
# should we train ALL models with said restrictions for consistency's sake?
# those being only training on 5000 samples and testing on the remaining (37000) samples
# let's see how that works out!


# # Part 1. INK Feature Models
# - Model 1. (Zero mean and SD=1) Multinomial Logit -> Ink Feature
# - Model 2. (Zero mean and SD=1) MN Logit -> Ink Feature + Our own special feature

# In[14]:


def print_feature(feat, feat_mean, feat_std):
    print(f"{feat}\n{feat_mean}\n{feat_std}\n")
    print(f"{np.size(feat)}, {np.size(feat_mean)}, {np.size(feat_std)}")


# In[54]:


# creating ink feature
ink = np.array([sum(row) for row in digits])


# In[55]:


ink_mean = [np.mean(ink[labels == i]) for i in range(10)] # mean for each digit
ink_std = [np.std(ink[labels == i]) for i in range(10)] # std for each digit
scaled_ink = (ink - np.mean(ink)) / np.std(ink)
print_feature(ink, ink_mean, ink_std)


# In[17]:


# our new feature - whitespace between numbers
# not just the opposite of ink, but the whitespace between a digit's
# rows and columns; 0 has a LOT of whitespace, 1 doesn´t have that much
def get_whitespace_feature(digits):
    ws = np.zeros(len(digits))
    
    print(ws)
    return ws


# In[18]:


ws = get_whitespace_feature(digits)
ws_mean = [np.mean(ink[labels == i]) for i in range(10)] # mean for each digit
ws_std = [np.std(ink[labels == i]) for i in range(10)] # sd for each digit
print_feature(ws, ws_mean, ws_std)


# In[57]:


# pipeline setup to facilitate modelling; consolidate training and testing datasets
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = ink.reshape(-1, 1) # reshaping since it's a single feature
y = labels
# I know the instructions mention we don´t need to do this now,
# but I´d rather keep all models (reasonably) consistent
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=42, 
                                                    test_size=37000)

# this pipeline logic is so we don´t leak data from the test set into the training set
scaled_logit = make_pipeline(StandardScaler(), LogisticRegression())
scaled_logit.fit(X_train, y_train)  # apply scaling on training data
scaled_logit.score(X_test, y_test)
y_pred = scaled_logit.predict(X_test)


# In[20]:


from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def show_results(model_desc, y_test, y_pred):
    print(f"\nResults for {model_desc}\n")
    print(classification_report(y_test, y_pred, zero_division=0)) # hiding zero division warn
    cm = confusion_matrix(y_test, y_pred, labels=scaled_logit.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=scaled_logit.classes_)
    disp.plot()
    plt.show()


# In[21]:


show_results("MN LOGIT - INK FEATURE", y_test, y_pred)


# # Part 2. All Pixel Values Models
# _**NOTE: Both with 196 (14*14 pixels) features (all resized pixel values)**_
# 
# - Model 3. (Regularized?) MN Logit (w/ LASSO penalty) 
# - Model 4. Support Vector Machines (SVM)

# ### MN Logit (w/ LASSO penalty)

# In[22]:


# find a way to remove all pixels that always have constant value

logreg = sklearn.linear_model.LogisticRegression(penalty='l1', c=1,solver='saga')
# saga is the only solver that supports l1 penalty and multi-class problems
#arr = np.array([ink_mean, ink_std])
#print(pd.DataFrame(arr))
#print(df.loc[:, (df.sum() > 0).all()])


# ### SVM + Grid Search

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm


# === Note on kernels and params ===

# rbf: gamma
# linear: x, x'
# sigmoid: coef0
# poly: degree, coef0
#   but coef0 can be safely left unchanged "in most cases" according to 
# https://stackoverflow.com/questions/21390570/scikit-learn-svc-coef0-parameter-range

paramGrid = {
    'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
    'C': [0.1, 0.5, 0.9, 1.5, 2, 2.5],
    'degree': [0.5, 1, 2, 5],
    'gamma': ['auto', 'scale'],
}

grid = GridSearchCV(SVC(), paramGrid, refit=True, verbose=3)
grid.fit(X_train2_std, y_train2)

