#!/usr/bin/env python
# coding: utf-8

# ## Part 1: preparing data
# 
# Load pixel data from CSV file.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


img_size = 28
mnist_data = pd.read_csv('./mnist.csv').values
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]      # DATA: raw digits data


# In[17]:


# Visually sampling the data

img_size = 28
columns = 5
rows = 2
fig = plt.figure(figsize=(26,13))

for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    random_index = random.randrange(0, len(digits))
    plt.imshow(digits[random_index].reshape(img_size, img_size))
    plt.xlabel(str(labels[random_index]))

plt.show()


# #### Resizing the image
# 
# After multiple attempts at fitting the model, it was observed that the required time for the model to converge was too expensive (in one instance, it does not converge even after 2 hours). Hence it was decided to resize the number of features down to 196 (14 x 14 pixel image).

# In[2]:





# In[19]:





# 

# In[3]:





# #### Splitting and scaling dataset

# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train0, X_test0, y_train0, y_test0 = train_test_split(
    digitsFiltered, 
    labels, 
    train_size=5000, 
    test_size=37000, 
    random_state=1
)
X_train0_std = scaler.fit_transform(X_train0)       # Digits, filtered and scaled.
X_test0_std = scaler.transform(X_test0)


scaler = StandardScaler()
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    digitsResized, 
    labels, 
    train_size=5000, 
    test_size=37000, 
    random_state=1
)
X_train1_std = scaler.fit_transform(X_train1)       # Digits, resized and scaled.
X_test1_std = scaler.transform(X_test1)


scaler = StandardScaler()
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    digitsResizedFiltered, 
    labels, 
    train_size=5000,
    test_size=37000, 
    random_state=1
)
X_train2_std = scaler.fit_transform(X_train2)       # Digits, resized filtered and scaled.
X_test2_std = scaler.transform(X_test2)


# #### End of part 1
# 
# We have prepared 3 sets of data:
# - Set 0 has its constant features removed.
# - Set 1 has its image dimension reduced by half.
# - Set 2 has both treatments from set 0 and 1.
# 
# All sets are scaled. We are going to work mainly with set 2. Other sets are supplementary.

# ## Part 2: SVM parameter tuning using grid search
# 
# We are going to tune the parameters for SVC with polynomial, linear, and RBF kernels.

# In[13]:





# In[15]:


from sklearn.metrics import classification_report, confusion_matrix 


print('Best estimator:', grid.best_estimator_)
print('Best params:', grid.best_params_)
print('Best score from cross-validation:', grid.best_score_)

print('\nPerformance on unseen data:')
gridPredictions = grid.predict(X_test2_std)
print(classification_report(y_test2, gridPredictions))
# print(confusion_matrix(y_test2, gridPredictions))


# In[16]:


m = SVC(kernel='poly', C=2.5, degree=2)
m.fit(X_train2_std, y_train2)
print('Accuracy on unseen data:', m.score(X_test2_std, y_test2))


# ## Part 3: Logistic Regression
# 
# Parameter tuning for Logistic Regression.
# 
# TODO: the Logreg model does not converge after running for more than 2 hours.

# In[ ]:


from sklearn.linear_model import LogisticRegressionCV


paramC = [0.1, 0.5, 0.9, 1.5, 2, 2.5]

mlogreg = LogisticRegressionCV(Cs=paramC, penalty='l1', solver='saga', cv=5, random_state=0, max_iter=1000, verbose=3)
mlogreg.fit(X_train2_std, y_train2)
accLogreg = mlogreg.score(X_test2_std, y_test2)
print(f'Accuracy: {accLogreg}')
print('Model\'s parameters are as follow:')
print(mlogreg.get_params())

# Refer to https://www.kaggle.com/code/joparga3/2-tuning-parameters-for-logistic-regression 
# to do parameter tuning manually.


# In[ ]:


from sklearn.linear_model import LogisticRegressionCV


accuracyA = LogisticRegressionCV(
    penalty='l1', 
    solver='saga', 
    random_state=0, 
    max_iter=1000
).fit(X_train1_std, y_train1).score(X_test1_std, y_test1)
print('Accuracy with all features (scaled and resized):', accuracyA)

accuracyB = LogisticRegressionCV(
    penalty='l1', 
    solver='saga', 
    random_state=0, 
    max_iter=1000
).fit(X_train2_std, y_train2).score(X_test2_std, y_test2)
print('Accuracy with subset of features (scaled and resized, constant features dropped):', accuracyB)

