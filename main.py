#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip install opencv-python')


# In[3]:


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


# In[31]:


# configuring figure/plot params
custom_params = {'figure.figsize':(16, 9)} 
sns.set_theme(style="whitegrid", font_scale=1.3, rc=custom_params)


# In[4]:


# loading dataset (it's a bit slow)
mnist_data = pd.read_csv('mnist.csv').values


# In[5]:


labels = mnist_data[:, 0]  # 0 to 9
digits = mnist_data[:, 1:] # 42000 digits
print(labels, len(labels))
print(digits, len(digits[0]))
mnist_df = pd.DataFrame(mnist_data)
mnist_df.head()
# first column is label, all (784) other columns are pixel values (0-255)


# # Data Exploration / Pre-Processing
# Exploring data and plotting cool stuff

# In[32]:


unique, counts = np.unique(labels, return_counts=True)
ax = sns.barplot(x=unique,y=counts)
ax.bar_label(ax.containers[0])
plt.title("Class distribution for edited MNIST dataset")
plt.ylabel("Counts")
plt.xlabel("Numbers")
plt.show()

fig = ax.get_figure()
fig.savefig("class_dist.png", dpi=300) 
# TODO: maybe plot mean and std in this plot?


# In[8]:


digitsResized = np.zeros((len(digits), 14*14))

for i, d in enumerate(digits):
    _d = np.reshape(d, (28, 28)).astype('float32')
    resized_d = cv2.resize(_d, (14, 14))
    d_ = np.reshape(resized_d, (1, 14*14))
    digitsResized[i] = d_

# print('Digits resized', np.shape(digitsResized))


# In[27]:


# randomly print one number from each class
def print_one_from_each(filename, digits, img_size):
    fig = plt.figure(figsize=(16,8))
    cols, rows = 5, 2
    for i in range(1, (cols*rows)+1):
        fig.add_subplot(rows, cols, i)
        index = np.where(labels == i-1)[0][0]
        plt.imshow(digits[index].reshape(img_size, img_size))
        plt.xlabel(str(labels[index]))
    plt.show()
    fig.savefig(f"{filename}.png", dpi=300)


# In[28]:


print_one_from_each("28_digits", digits, 28)


# In[29]:


# visually sampling the resized data
print_one_from_each("14_digits", digitsResized, 14)


# #### Drop useless features
# 
# Useless features are those with constant values across all data points, hence cannot be used to distinguish between data.

# In[11]:


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


# In[12]:


# given some restriction on later parts of the assignment,
# should we train ALL models with said restrictions for consistency's sake?
# those being only training on 5000 samples and testing on the remaining (37000) samples


# # Part 1. INK Feature Models
# - Model 1. (Zero mean and SD=1) Multinomial Logit -> Ink Feature
# - Model 2. (Zero mean and SD=1) MN Logit -> Ink Feature + Our own special feature

# In[13]:


# preparing to build both features
def print_feature(feat, feat_mean, feat_std):
    print(f"{feat}\n{feat_mean}\n{feat_std}\n")
    print(f"{np.size(feat)}, {np.size(feat_mean)}, {np.size(feat_std)}")


# In[64]:


# creating ink feature
ink = np.array([sum(row) for row in digits])
print(f"Ink Feature:\n {ink}")


# In[25]:


ink_mean = [np.mean(ink[labels == i]) for i in range(10)] # mean for each digit
ink_std = [np.std(ink[labels == i]) for i in range(10)] # std for each digit
print_feature(ink, ink_mean, ink_std)


# In[10]:


# our new feature - number of lines (horizontal and vertically)
# how many one-line pixels are enveloped by zero pixels
# rows and columns; 0 has a LOT of whitespace, 1 doesn´t have that much
def build_line_feature(digits, max_count=6, img_size=28):
    """
    building line counter feature
    digits: array for all digits and all pixel values
    max_count: max number of lines being counted
    img_size: size of the image (default is 28x28)
    """ 
    num_samples = len(digits)
    df_lines = pd.DataFrame()
    # generating empty dataset
    for i in range(max_count):
        directions = ['h', 'v']
        for direction in directions:
            new_col = f"{i}{direction}_line_ratio"
            df_lines[new_col]=0
    # print(df_lines)
    
    diagonal_arr = np.arange(start=0, stop=num_samples+1, step=img_size+1)
    for row in digits: # 784 length rows
        for index in diagonal_arr: # top_left-bot_right diagonal
            print(index)
    
    
    return df_lines


# In[11]:


lines = build_line_feature(digits)
# lines_mean = [np.mean(lines[labels == i]) for i in range(10)] # mean for each digit
# lines_std = [np.std(lines[labels == i]) for i in range(10)] # sd for each digit
# print_feature(lines, lines_mean, lines_std)


# ## Model 1. MN Logit (only INK feature)

# In[24]:


# pipeline setup to facilitate modelling; consolidate training and testing datasets
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

l1f_features = ink.reshape(-1, 1) # reshaping since it's a single feature
# I know the instructions mention we don´t need to do this now,
# but I´d rather keep all models (reasonably) consistent
l1f_train, l1f_test, y_l1f_train, y_l1f_test = train_test_split(l1f_features, labels, 
                                                    random_state=42, 
                                                    test_size=0.2)

# this pipeline logic is so we don´t leak data from the test set into the training set
scaled_logit = make_pipeline(StandardScaler(), LogisticRegression())
scaled_logit.fit(l1f_train, y_l1f_train)  # apply scaling on training data
scaled_logit.score(l1f_test, y_l1f_test)
y_l1f_pred = scaled_logit.predict(l1f_test)


# In[62]:


from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def show_results(filename, y_test, y_pred):
    print(f"\nResults for {filename}\n")
    print(classification_report(y_test, y_pred, zero_division=0)) # hiding zero division warn
    cm = confusion_matrix(y_test, y_pred, labels=scaled_logit.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=scaled_logit.classes_)
    disp.plot()
    plt.grid(visible=None)
    plt.show()
    # TODO: how to save this as pic?
    disp.figure_.savefig(f"{filename}.png", dpi=300)


# In[63]:


show_results("mn_logit-ink_feature", y_l1f_test, y_l1f_pred)


# ## Model 2. MN Logit (INK feature + Our Feature)

# In[ ]:


# TODO: complete this
l2f_features = ink.reshape(-1, 1)
l2f_train, l2f_test, y_l2F_train, y_l2f_test = train_test_split(l2f_features, labels, 
                                                    random_state=42, 
                                                    test_size=0.2)

# this pipeline logic is so we don´t leak data from the test set into the training set
# scaled_logit = make_pipeline(StandardScaler(), LogisticRegression())
scaled_logit.fit(l2f_train, y_l2f_train)  # apply scaling on training data
scaled_logit.score(l2f_test, y_l2f_test)
y_l2f_pred = scaled_logit.predict(l2f_test)


# In[ ]:


show_results("MN LOGIT - BOTH FEATURES", y_l2f_test, y_l2f_pred)


# # Part 2. All Pixel Values Models
# _**NOTE: Both with 196 (14*14 pixels) features (all resized pixel values)**_
# 
# - Model 3. (Regularized?) MN Logit (w/ LASSO penalty) 
# - Model 4. Support Vector Machines (SVM)

# In[ ]:


# separating training and test samples
# specific requirements here (5k train and 37k test split - total 42k)
# p2_features = resized all digits
p2_train, p2_test, y_p2_train, y_p2_test = train_test_split(p2_features, labels, 
                                                    random_state=42, 
                                                    test_size=37000)


# ### MN Logit (w/ LASSO penalty)

# In[ ]:


from sklearn.linear_model import LogisticRegression

# saga is the only solver that supports l1 penalty and multi-class problems
# arr = np.array([ink_mean, ink_std])
# print(pd.DataFrame(arr))
# print(df.loc[:, (df.sum() > 0).all()])

p2_logit = make_pipeline(
    StandardScaler(), 
    LogisticRegression(penalty='l1', C=0.5, solver='saga')
)

p2_logit.fit(p2_train, y_p2_train)
p2_logit.score(p2_test, y_p2_test)
p2_logit_pred = p2_logit.predict(p2_test)


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

