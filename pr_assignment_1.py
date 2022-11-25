
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
from sklearn.model_selection import cross_val_score

mnist_data = pd.read_csv('mnist.csv').values

labels = mnist_data[:, 0]       # the true number written by the user
digits = mnist_data[:, 1:]      # the pixel value of the image 


# img_size = 28
# plt.imshow(digits[0].reshape(img_size, img_size))
# plt.show()


# create ink feature

ink = np.array([sum(row) for row in digits])

# compute mean for each digit class
ink_mean = [np.mean(ink[labels == i]) for i in range(10)]

# compute standard deviation for each digit class
ink_std = [np.std(ink[labels == i]) for i in range(10)]

#print(np.size(ink), np.size(ink_mean), np.size(ink_std))

scaled_ink = (ink - np.mean(ink))/np.std(ink)
print(scaled_ink)

# [TODO]perform cross-validation and replace value of c in the logistic regression

logreg = sklearn.linear_model.LogisticRegression(penalty='l1', c=1,solver='saga')
# saga is the only solver that supports l1 penalty and multi-class problems

#arr = np.array([ink_mean, ink_std])
#print(pd.DataFrame(arr))


#print(df.loc[:, (df.sum() > 0).all()])

# other feature, go through each row of pixels, count how many times it reaches a non-zero pixel with a zero ink pixel separator --> if highest value of this is 2, feature is # of rows value 2 / # of rows value 1
# we do not care about rows with value 0 --> covered by ink feature
# problem: does not distinguish between 6 and 9 --> ink feature also does not distinguish between those in theory



# for part 2, when we need to consider each pixel as an individual feature

#df.info()

#print(df[400].sum())
#print(df.sum())

# find a way to remove all pixels that always have constant value